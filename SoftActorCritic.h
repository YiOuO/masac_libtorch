#pragma once
#include <torch/torch.h>
#include "Models.h"       // Requires SACActor / SACCritic definitions
#include "ReplayBuffer.h" // Requires ReplayBuffer definition

class SAC {
public:
    SAC(int state_dim,
        int action_dim,
        double alpha,
        double gamma,
        double tau,
        double lr)
        : actor(state_dim, action_dim),
          critic1(state_dim , action_dim),
          critic2(state_dim , action_dim),
          target1(state_dim , action_dim),
          target2(state_dim , action_dim),
          actor_opt(actor->parameters(), lr),
          critic1_opt(critic1->parameters(), lr),
          critic2_opt(critic2->parameters(), lr),
          alpha_(alpha), gamma_(gamma), tau_(tau)
    {
        // // Initialize target networks with source network parameters
        hard_update(critic1, target1);
        hard_update(critic2, target2);

        // Use double precision
        actor  ->to(torch::kF64);
        critic1->to(torch::kF64);
        critic2->to(torch::kF64);
        target1->to(torch::kF64);
        target2->to(torch::kF64);
    }

    /**
     * Performs one training step using a mini-batch from the replay buffer.
     * If there are not enough samples, the function returns without doing anything.
     */
    void train_step(ReplayBuffer& buffer, size_t batch_size)
    {
        if (buffer.size() < batch_size) return;

        auto [s_batch, a_batch, r_batch, ns_batch, d_batch] =
            buffer.sample(batch_size);

        // ---------- Critic update ----------
        auto [a_next, logp_next] = actor->sample(ns_batch);
        std::cout << "ns_batch shape: " << ns_batch.sizes() << std::endl;
        std::cout << "a_next shape: " << a_next.sizes() << std::endl;
        auto q1_next = target1->forward(ns_batch, a_next);
        std::cout<<"gogogo "<<std::endl;
        auto q2_next = target2->forward(ns_batch, a_next);
        std::cout<<"gogogo "<<std::endl;
        auto q_target = r_batch +
                        gamma_ * (1 - d_batch) *
                        (torch::min(q1_next, q2_next) - alpha_ * logp_next);

        auto q1 = critic1->forward(s_batch, a_batch);
        auto q2 = critic2->forward(s_batch, a_batch);

        auto loss1 = torch::mse_loss(q1, q_target.detach());
        auto loss2 = torch::mse_loss(q2, q_target.detach());

        critic1_opt.zero_grad();
        loss1.backward();
        critic1_opt.step();

        critic2_opt.zero_grad();
        loss2.backward();
        critic2_opt.step();

        // ---------- Actor update ----------
        auto [a_pred, logp_pred] = actor->sample(s_batch);
        auto q_pred = torch::min(critic1->forward(s_batch, a_pred),
                                 critic2->forward(s_batch, a_pred));
        auto actor_loss = (alpha_ * logp_pred - q_pred).mean();

        actor_opt.zero_grad();
        actor_loss.backward();
        actor_opt.step();

        // ---------- Soft update target networks ----------
        torch::NoGradGuard no_grad;
        soft_update(critic1, target1);
        soft_update(critic2, target2);
    }

    // Optionally: save/load model weights
    void save(const std::string& path_prefix) const {
        torch::save(actor,   path_prefix + "_actor.pt");
        torch::save(critic1, path_prefix + "_critic1.pt");
        torch::save(critic2, path_prefix + "_critic2.pt");
    }
    void load(const std::string& path_prefix) {
        torch::load(actor,   path_prefix + "_actor.pt");
        torch::load(critic1, path_prefix + "_critic1.pt");
        torch::load(critic2, path_prefix + "_critic2.pt");
        hard_update(critic1, target1);
        hard_update(critic2, target2);
    }

    // Public members for inference or external access
    Actor   actor;
    Critic  critic1, critic2;   // Main Q-networks
    Critic  target1, target2;   // Target Q-networks

private:
    torch::optim::Adam actor_opt;
    torch::optim::Adam critic1_opt;
    torch::optim::Adam critic2_opt;

    double alpha_, gamma_, tau_;

    /**
     * Soft update: target ← (1−τ)·target + τ·source
     */
    void soft_update(const Critic& source, Critic& target)
    {
        auto source_params  = source->named_parameters();
        auto target_params  = target->named_parameters();

        for (const auto& item : source->named_parameters()) {
            const auto& name = item.key();
            if (target_params.contains(name)) {
                auto& t_param = target_params[name];
                t_param.mul_(1.0 - tau_);
                t_param.add_(tau_ * item.value());
            }
        }
    }

    /**
     * Hard update: target ← source
     */
    void hard_update(const Critic& source, Critic& target) {
        auto target_params = target->parameters();
        auto source_params = source->parameters();
        for (size_t i = 0; i < target_params.size(); ++i) {
            target_params[i].data().copy_(source_params[i].data());
        }
    }

};
