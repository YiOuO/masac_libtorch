#pragma once
#include <Eigen/Core>
#include <torch/torch.h>
#include <vector>
#include <tuple>
#include <cstring>
#include <cmath>

enum MAStatus
{
    MA_PLAYING = 0,
    MA_DONE = 1,
    MA_RESETTING = 2
};

/**
 * A simple cooperative grid environment without physical interactions.
 * - Each agent i has a position and a goal: (pos_x, pos_y) -> (goal_x, goal_y).
 * - Each step applies a bounded displacement proportional to the action.
 * - Rewards are PER-AGENT (vector): distance reduction bonus + terminal bonus/penalty.
 * - Done is shared: episode ends if all agents reach their goals OR any agent goes out-of-bounds.
 */
class MultiAgentEnvironment
{
public:
    explicit MultiAgentEnvironment(int numberOfAgents,
                                   const std::vector<Eigen::Vector2d> &initialGoals)
        : numberOfAgents_(numberOfAgents),
          positions_(numberOfAgents),
          goals_(initialGoals),
          reached_(numberOfAgents_, 0)
    {
        for (int i = 0; i < numberOfAgents_; ++i)
            positions_[i].setZero();
    }

private:
    int numberOfAgents_;
    int statedim_ = 4; // [pos_x, pos_y, goal_x, goal_y]
    std::vector<uint8_t> reached_; double reach_radius_=0.6, leave_radius_=0.8;
public:
    std::vector<Eigen::Vector2d> positions_;
    std::vector<Eigen::Vector2d> goals_;

    torch::Tensor getLocalObservation(int agentIndex)
    {
        torch::Tensor obs = torch::zeros({1, 4}, torch::kF64);
        obs[0][0] = positions_[agentIndex][0];
        obs[0][1] = positions_[agentIndex][1];
        obs[0][2] = goals_[agentIndex][0];
        obs[0][3] = goals_[agentIndex][1];
        return obs;
    }

    // Global state: concatenation of all agents' local observations in fixed order
    torch::Tensor getGlobalState()
    {
        torch::Tensor state = torch::zeros({1, statedim_ * numberOfAgents_}, torch::kF64);
        for (int i = 0; i < numberOfAgents_; ++i)
        {
            state[0][statedim_ * i + 0] = positions_[i][0];
            state[0][statedim_ * i + 1] = positions_[i][1];
            state[0][statedim_ * i + 2] = goals_[i][0];
            state[0][statedim_ * i + 3] = goals_[i][1];
        }
        return state;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int>
    Step(const torch::Tensor &jointAction)
    {
        // ===== 1) 记录旧距离 =====
        std::vector<double> old_dist_(numberOfAgents_, 0.0);
        for (int i = 0; i < numberOfAgents_; ++i)
            old_dist_[i] = distance(positions_[i], goals_[i]);

        // ===== 2) 施加动作 =====
        const double maxStepLength = 0.1;
        auto action = jointAction.squeeze().to(torch::kF64);
        std::vector<double> act_energy_i(numberOfAgents_, 0.0);
        for (int i = 0; i < numberOfAgents_; ++i)
        {
            const double ax = action[2 * i].item<double>();
            const double ay = action[2 * i + 1].item<double>();
            act_energy_i[i] = ax * ax + ay * ay; // used for reward
            positions_[i][0] += maxStepLength * ax;
            positions_[i][1] += maxStepLength * ay;
        }

        // ===== 3) 计算当前距离、团队统计量 =====
        std::vector<double> distNow(numberOfAgents_, 0.0);
        double sum_d = 0.0;
        for (int i = 0; i < numberOfAgents_; ++i)
        {
            distNow[i] = distance(positions_[i], goals_[i]);
            sum_d += distNow[i];
        }
        const double mean_d = sum_d / std::max(1, numberOfAgents_);
        double var_d = 0.0;
        for (int i = 0; i < numberOfAgents_; ++i)
        {
            const double t = distNow[i] - mean_d;
            var_d += t * t;
        }
        var_d /= std::max(1, numberOfAgents_);

        
        // ===== 4) 逐体奖励（一次性终点 + 到达后不再累计） =====
        torch::Tensor rewardVector = torch::zeros({1, numberOfAgents_}, torch::kF64);

        // 可调权重（按需微调/置零以关闭团队项）
        const double w_ind = 10.0;  // 个体距离缩短
        const double w_goal = 20.0; // 一次性终点奖
        const double w_oob = 20.0;  // 越界惩罚
        const double w_mean = 0.0;  // 团队平均更近（负号在下面体现）
        const double w_sync = 0.0;  // 团队同步（方差惩罚）
        const double w_coll = 0.0;  // 碰撞惩罚（阈值见下）
        const double w_act = 0.1;  // 动作能耗惩罚（共享）

        const double collision_dmin = 0.5; // 碰撞阈值

        // 简单碰撞惩罚分配到各体
        std::vector<double> coll_pen_i(numberOfAgents_, 0.0);
        for (int i = 0; i < numberOfAgents_; ++i)
        {
            for (int j = i + 1; j < numberOfAgents_; ++j)
            {
                const double dij = (positions_[i] - positions_[j]).norm();
                if (dij < collision_dmin)
                {
                    coll_pen_i[i] -= w_coll;
                    coll_pen_i[j] -= w_coll;
                }
            }
        }

        int reachedCount = 0;
        int outOfBoundsCount = 0;

        for (int i = 0; i < numberOfAgents_; ++i)
        {
            const double d_old = old_dist_[i];
            const double d_now = distNow[i];

            const bool was_reached = (reached_[i] != 0);
            const bool now_reached = (d_now < reach_radius_);

            // 迟滞：到达后只要没远离到 leave_radius_ 就视为“保持到达”
            if (!was_reached && now_reached)
            {
                reached_[i] = 1; // 第一次到达
            }
            else if (was_reached && d_now > leave_radius_)
            {
                reached_[i] = 0; // 明显离开目标区
            }

            double r_i = 0.0;

            // 个体 shaped 奖励：仅在“尚未到达”阶段累计
            if (!was_reached)
            {
                const double delta = d_old - d_now; // 更近为正
                // 若不希望因为抖动出现负分，可启用半波整流：
                // const double delta_pos = std::max(0.0, delta);
                r_i += w_ind * (d_old - d_now);
            }

            // 一次性终点奖：只在首次进入 reach_radius_ 时发放
            if (!was_reached && now_reached)
            {
                r_i += w_goal;
            }

            // 越界惩罚
            if (d_now > 10.0)
            {
                r_i -= w_oob;
                outOfBoundsCount++;
            }

            // 团队项（每个体都能感知团队信号）
            r_i += -w_mean * mean_d;       // 团队平均越小越好
            r_i += -w_sync * var_d;        // 距离越均衡越好（同步推进）
            r_i += coll_pen_i[i];          // 碰撞惩罚
            
            if (!reached_[i]) {
                r_i += -w_act * act_energy_i[i];
            }

            if (reached_[i])
                reachedCount++;

            rewardVector[0][i] = r_i;
        }

        // ===== 5) 共享终止条件 =====
        const bool all_reached = (reachedCount == numberOfAgents_);
        const bool any_oob = (outOfBoundsCount > 0);

        int status = MA_PLAYING;
        bool done = false;
        if (all_reached)
        {
            status = MA_DONE;
            done = true;
            printf("team win\n");
        }
        else if (any_oob)
        {
            status = MA_DONE;
            done = true;
            printf("team lose\n");
        }

        // ===== 6) 打包返回 =====
        auto nextState = getGlobalState();
        auto doneTensor = torch::full({1, 1}, done ? 1.0 : 0.0, torch::kF64);
        return {nextState, rewardVector, doneTensor, status};
    }

    // Compute distance between a & b
    double distance(const Eigen::Vector2d &a, const Eigen::Vector2d &b)
    {
        return (a - b).norm();
    }

    void reset()
    {
        for (auto &p : positions_)
            {p.setZero();}
        std::fill(reached_.begin(), reached_.end(), 0);
    }
};
