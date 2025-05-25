#pragma once

#include <torch/torch.h>
#include <math.h>
// ========== Actor Network========= //
struct ActorImpl : public torch::nn::Module
{
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, mean{nullptr}, log_std{nullptr};
    ActorImpl(int64_t input_dim, int64_t output_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, 16));
        fc2 = register_module("fc2", torch::nn::Linear(16, 32));
        mean = register_module("mean", torch::nn::Linear(32, output_dim));
        log_std = register_module("log_std", torch::nn::Linear(32, output_dim));
    }
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        auto mu = mean->forward(x);
        auto log_std_out = torch::clamp(log_std->forward(x), -20, 2); // numerical stability
        return std::make_tuple(mu, log_std_out);
    }

    std::tuple<torch::Tensor, torch::Tensor> sample(torch::Tensor x) {
        auto [mu, log_std] = forward(x);
        auto std = torch::exp(log_std);
        auto eps = torch::randn_like(std);
        auto action = mu + std * eps;
        auto log_prob = -0.5 * (eps.pow(2) + 2*log_std + log(2*M_PI)).sum(1, true);
        return {action.tanh(), log_prob};  // tanh约束动作范围
    }
};
TORCH_MODULE(Actor);

// ========== Critic Q-Network ========== //
struct CriticImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    CriticImpl(int64_t input_dim, int64_t action_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim + action_dim, 32));
        fc2 = register_module("fc2", torch::nn::Linear(32, 32));
        fc3 = register_module("fc3", torch::nn::Linear(32, 1));
    }

    torch::Tensor forward(torch::Tensor state, torch::Tensor action) {
        auto x = torch::cat({state, action}, 1);
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x);
    }
};

TORCH_MODULE(Critic);

// // Network model for Proximal Policy Optimization on Incy Wincy.
// struct ActorCriticImpl : public torch::nn::Module 
// {
//     // Actor.
//     torch::nn::Linear a_lin1_, a_lin2_, a_lin3_;
//     torch::Tensor mu_;
//     torch::Tensor log_std_;

//     // Critic.
//     torch::nn::Linear c_lin1_, c_lin2_, c_lin3_, c_val_;

//     ActorCriticImpl(int64_t n_in, int64_t n_out, double std)
//         : // Actor.
//           a_lin1_(torch::nn::Linear(n_in, 16)),
//           a_lin2_(torch::nn::Linear(16, 32)),
//           a_lin3_(torch::nn::Linear(32, n_out)),
//           mu_(torch::full(n_out, 0.)),
//           log_std_(torch::full(n_out, std)),
          
//           // Critic
//           c_lin1_(torch::nn::Linear(n_in, 16)),
//           c_lin2_(torch::nn::Linear(16, 32)),
//           c_lin3_(torch::nn::Linear(32, n_out)),
//           c_val_(torch::nn::Linear(n_out, 1)) 
//     {
//         // Register the modules.
//         register_module("a_lin1", a_lin1_);
//         register_module("a_lin2", a_lin2_);
//         register_module("a_lin3", a_lin3_);
//         register_parameter("log_std", log_std_);

//         register_module("c_lin1", c_lin1_);
//         register_module("c_lin2", c_lin2_);
//         register_module("c_lin3", c_lin3_);
//         register_module("c_val", c_val_);
//     }

//     // Forward pass.
//     auto forward(torch::Tensor x) -> std::tuple<torch::Tensor, torch::Tensor> 
//     {

//         // Actor.
//         mu_ = torch::relu(a_lin1_->forward(x));
//         mu_ = torch::relu(a_lin2_->forward(mu_));
//         mu_ = torch::tanh(a_lin3_->forward(mu_));

//         // Critic.
//         torch::Tensor val = torch::relu(c_lin1_->forward(x));
//         val = torch::relu(c_lin2_->forward(val));
//         val = torch::tanh(c_lin3_->forward(val));
//         val = c_val_->forward(val);

//         if (this->is_training()) 
//         {
//             torch::NoGradGuard no_grad;

//             torch::Tensor action = at::normal(mu_, log_std_.exp().expand_as(mu_));
//             return std::make_tuple(action, val);  
//         }
//         else 
//         {
//             return std::make_tuple(mu_, val);  
//         }
//     }

//     // Initialize network.
//     void normal(double mu, double std) 
//     {
//         torch::NoGradGuard no_grad;

//         for (auto& p: this->parameters()) 
//         {
//             p.normal_(mu,std);
//         }         
//     }

//     auto entropy() -> torch::Tensor
//     {
//         // Differential entropy of normal distribution. For reference https://pytorch.org/docs/stable/_modules/torch/distributions/normal.html#Normal
//         return 0.5 + 0.5*log(2*M_PI) + log_std_;
//     }

//     auto log_prob(torch::Tensor action) -> torch::Tensor
//     {
//         // Logarithmic probability of taken action, given the current distribution.
//         torch::Tensor var = (log_std_+log_std_).exp();

//         return -((action - mu_)*(action - mu_))/(2*var) - log_std_ - log(sqrt(2*M_PI));
//     }
// };

// TORCH_MODULE(ActorCritic);
