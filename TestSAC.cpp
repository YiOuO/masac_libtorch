#include <fstream>
#include <Eigen/Core>
#include <torch/torch.h>
#include "SoftActorCritic.h"
#include "Models.h"
#include "TestEnvironment.h"
#include <random>

int main()
{
    // Random engine.
    std::random_device rd;
    std::mt19937 re(rd());
    std::uniform_int_distribution<> dist(-5, 5);

    // Environment.
    double x = double(dist(re)); // goal x pos
    double y = double(dist(re)); // goal y pos
    TestEnvironment env(x, y);
    
    // SAC Model.
    uint n_in = 4;
    uint n_out = 2;
    double std = 1e-2;
    float lr = 5e-4, alpha_lr = 1e-3, alpha = 0.1, gamma = 0.99, tau=0.005;
    SAC sac(n_in,n_out,alpha,alpha_lr,gamma,tau,lr);
    sac.actor->to(torch::kF64);
    sac.actor->eval();
    torch::load(sac.actor,"best_model_actor.pt");

    // Training loop.
    uint n_iter = 10000;

    // Output.
    std::ofstream out;
    out.open("../data/data_test.csv");

    // episode, agent_x, agent_y, goal_x, goal_y, STATUS=(PLAYING, WON, LOST, RESETTING)
    out << 1 << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << RESETTING << "\n";

    for (uint i = 0; i < n_iter; ++i)
    {
        // play
        auto state = env.State();
        auto [action, log_prob] = sac.actor->sample(state);
        auto action_cpu = action.squeeze().cpu();
        double x_act = action_cpu[0].item<double>();
        double y_act = action_cpu[1].item<double>();

        // Check for done state.
        auto step_data = env.Act(x_act, y_act);
        auto done = std::get<2>(step_data);

        // episode, agent_x, agent_y, goal_x, goal_y, AGENT=(PLAYING, WON, LOST, RESETTING)
        out << 1 << ", " << env.pos_(0) << ", " << env.pos_(1) << ", "
            << env.goal_(0) << ", " << env.goal_(1) << ", " << std::get<1>(step_data) << "\n";

        if (done.item<double>() == 1.0)
        {
            // Set new goal.
            double x_new = double(dist(re));
            double y_new = double(dist(re));
            env.SetGoal(x_new, y_new);

            // Reset the position of the agent.
            env.Reset();

            // episode, agent_x, agent_y, goal_x, goal_y, AGENT=(PLAYING, WON, LOST, RESETTING)
            out << 1 << ", " << env.pos_(0) << ", " << env.pos_(1) << ", "
                << env.goal_(0) << ", " << env.goal_(1) << ", " << RESETTING << "\n";
        }
    }

    out.close();

    return 0;

}