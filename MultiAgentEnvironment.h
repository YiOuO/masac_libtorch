#pragma once
#include <Eigen/Core>
#include <torch/torch.h>
#include <vector>
#include <tuple>
#include <cstring>
#include <cmath>

enum MAStatus { MA_PLAYING = 0, MA_DONE = 1, MA_RESETTING = 2 };

/**
 * A simple cooperative grid environment without physical interactions.
 * - Each agent i has a position and a goal: (pos_x, pos_y) -> (goal_x, goal_y).
 * - Each step applies a bounded displacement proportional to the action.
 * - Rewards are PER-AGENT (vector): distance reduction bonus + terminal bonus/penalty.
 * - Done is shared: episode ends if all agents reach their goals OR any agent goes out-of-bounds.
 */
class MultiAgentEnvironment {
public:
    explicit MultiAgentEnvironment(int numberOfAgents,
                                   const std::vector<Eigen::Vector2d>& initialGoals)
        : numberOfAgents_(numberOfAgents),
          positions_(numberOfAgents),
          goals_(initialGoals) {
        for (int i = 0; i < numberOfAgents_; ++i) positions_[i].setZero();
    }
	private:
		int numberOfAgents_;
		int statedim_ = 4; // [pos_x, pos_y, goal_x, goal_y]
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
	torch::Tensor getGlobalState() {
		torch::Tensor state = torch::zeros({1, statedim_ * numberOfAgents_}, torch::kF64);
		for (int i = 0; i < numberOfAgents_; ++i) {
			state[0][statedim_ * i + 0] = positions_[i][0];
			state[0][statedim_ * i + 1] = positions_[i][1];
			state[0][statedim_ * i + 2] = goals_[i][0];
			state[0][statedim_ * i + 3] = goals_[i][1];
		}
		return state;
	}
	

	
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int> 
	Act(const torch::Tensor& jointAction)
	{
		// Record previous distances to compute shaped reward
		std::vector<double> old_dist_(numberOfAgents_, 0.0);
        for (int i = 0; i < numberOfAgents_; ++i) {
            old_dist_[i] = distance(positions_[i], goals_[i]);
        }

		// Apply actions
		const double maxStepLength = 0.1;
		auto action = jointAction.squeeze().to(torch::kF64);
		for (int i = 0; i < numberOfAgents_; ++i) {
            const double ax = action[2 * i].item<double>();
            const double ay = action[2 * i + 1].item<double>();
            positions_[i][0] += maxStepLength * ax;
            positions_[i][1] += maxStepLength * ay;			
		}

        // Compute per-agent rewards
        torch::Tensor rewardVector = torch::zeros({1, numberOfAgents_}, torch::kF64);
        int reachedCount = 0;
        int outOfBoundsCount = 0;
	
        for (int i = 0; i < numberOfAgents_; ++i) {
            const double distNow = distance(positions_[i], goals_[i]);
            double reward_i = 10.0 * (old_dist_[i] - distNow);  // distance reduction

            if (distNow < 0.6) {   // reached nearby goal
                reward_i += 100.0;
                reachedCount++;
            }
            if (distNow > 10.0) {  // out-of-bounds penalty
                reward_i -= 100.0;
                outOfBoundsCount++;
            }
            rewardVector[0][i] = reward_i;
        }	
        // Episode termination condition (shared)
        int status = MA_PLAYING;
        bool done = false;
        if (reachedCount == numberOfAgents_) { status = MA_DONE; done = true; }
        if (outOfBoundsCount > 0)            { status = MA_DONE; done = true; }

        auto nextState = getGlobalState();
        auto doneTensor = torch::full({1, 1}, done ? 1.0 : 0.0, torch::kF64);
        return {nextState, rewardVector, doneTensor, status};
	}

	// Compute distance between a & b
    double distance(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
        return (a - b).norm();
    }

	void reset() {
        for (auto& p : positions_) p.setZero();
    }

};
