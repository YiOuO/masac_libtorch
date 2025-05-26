#include <fstream>
#include <Eigen/Core>
#include <torch/torch.h>
#include <random>
#include "SoftActorCritic.h"
#include "Models.h"
#include "TestEnvironment.h"
#include "ReplayBuffer.h"

int main()
{
    // Random engine.
    std::random_device rd;
    std::mt19937 re(rd());
    std::uniform_int_distribution<> dist(-5, 5); 

    // Environment.
    double x = double(dist(re)); // goal x pos
    double y = double(dist(re)); // goal y pos
    TestEnvironment env(x,y);

    // SAC Model.
    uint n_in = 4;
    uint n_out = 2;
    double std = 1e-2;
    float lr = 3e-4, alpha_lr = 1e-3, alpha = 0.1, gamma = 0.99, tau=0.005;
    SAC sac(n_in,n_out,alpha,gamma,tau,lr);
    
    // Replay buffer
    ReplayBuffer buffer(100000);

    // Training loop.
    uint n_iter = 10000;
    uint n_epochs = 30;
    uint batch_size = 2048;
    uint mini_batch_size = 512;

    // Output.
    std::ofstream out;
    out.open("../data/data.csv");

    // episode, agent_x, agent_y, goal_x, goal_y, STATUS=(PLAYING, WON, LOST, RESETTING)
    out << 1 << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << RESETTING << "\n";

    // Counter.
    uint c = 0;

    // Average reward.
    double best_avg_reward = 0.;
    double avg_reward = 0.;

    for (uint e=1;e<=n_epochs;e++)
    {
        printf("epoch %u/%u\n", e, n_epochs);
        for (auto i=0; i<n_iter; i++)
        {
            // std::cout<<"i "<<i<<std::endl;
            // Sate of env.
            auto state = env.State();

            // Play
            auto [action, log_prob] = sac.actor->sample(state);
            auto action_cpu = action.squeeze().cpu();
            double x_act = action_cpu[0].item<double>();
            double y_act = action_cpu[1].item<double>();    
            auto sd = env.Act(x_act,y_act);
            // episode, agent_x, agent_y, goal_x, goal_y, AGENT=(PLAYING, WON, LOST, RESETTING)
            out << e << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << std::get<1>(sd) << "\n";
            // Reward
            auto reward = env.Reward(std::get<1>(sd));
            auto done = std::get<2>(sd);
            avg_reward += reward.item<double>()/n_iter;

            // New state
            auto next_state = env.State();
            // std::cout<<"next state "<<next_state<<std::endl;
            // Add to buffer 
            buffer.add(state,action,reward,next_state,done);

            // Update
            if(buffer.size() > batch_size)
            {
                // Train NN
                sac.train_step(buffer, mini_batch_size);
                
            }

            if(done.item<double>() == 1.)
            {
                // Set new goal.
                double x_new = double(dist(re)); 
                double y_new = double(dist(re));
                env.SetGoal(x_new, y_new);
                // Reset the position of the agent.
                env.Reset();
                // episode, agent_x, agent_y, goal_x, goal_y, STATUS=(PLAYING, WON, LOST, RESETTING)
                out << e << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << RESETTING << "\n";               
            }

        }

        // Save the best net.
        if (avg_reward > best_avg_reward) {

            best_avg_reward = avg_reward;
            printf("Best average reward: %f\n", best_avg_reward);
            sac.save("best_model");
        }

        avg_reward = 0.;
        // Reset the position of goal to (x_new, y_new).
        double x_new = double(dist(re)); 
        double y_new = double(dist(re));
        env.SetGoal(x_new, y_new);

        // Reset the position of the agent to (0,0).
        env.Reset();

        // episode, agent_x, agent_y, goal_x, goal_y, STATUS=(PLAYING, WON, LOST, RESETTING)
        out << e << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << RESETTING << "\n";
    }

    out.close();

    return 0;
}
