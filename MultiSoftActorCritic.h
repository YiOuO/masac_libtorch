// #pragma once
// #include <torch/torch.h>
// #include <vector>
// #include <memory>
// #include "Models.h"        // Reuse your Actor / Critic implementations
// #include "ReplayBuffer.h"  // Reuse your buffer (store reward as [1, N])

// /**
//  * Multi-Agent SAC with INDEPENDENT centralized Twin-Q per agent (CTDE):
//  * - Each agent i has its own: Actor, temperature alpha_i, and Twin-Q_i (with targets).
//  * - Critic inputs are centralized: concat(all local observations), concat(all actions).
//  * - Actor i is updated against its OWN critic Q_i.
//  * - Reward is per-agent: R[:, i].
//  * - Done is shared here (shape [B,1]) to keep the example simple.
//  */
// class MultiAgentSAC{
// public:
//     MultiAgentSAC(int numberOfAgents,
//                                int localObservationDimension,
//                                int actionDimension,
//                                double initialAlpha,
//                                double alphaLearningRate,
//                                double discountFactor,
//                                double targetUpdateTau,
//                                double learningRate)
//     : numberOfAgents_(numberOfAgents),
//       localObservationDimension_(localObservationDimension),
//       actionDimension_(actionDimension),
//       globalStateDimension_(numberOfAgents * localObservationDimension),
//       jointActionDimension_(numberOfAgents * actionDimension),
//       gamma_(discountFactor),
//       tau_(targetUpdateTau),
//       targetEntropy_(-static_cast<double>(actionDimension)) // standard SAC heuristic
//     {
//         // Per-agent actor and temperature
//         actors_.reserve(numberOfAgents_);
//         actorOptimizers_.reserve(numberOfAgents_);
//         logAlphas_.reserve(numberOfAgents_);
//         alphaOptimizers_.reserve(numberOfAgents_);
//         temperatures_.resize(numberOfAgents_);

//         for (int i = 0; i < numberOfAgents_; ++i) {
//             actors_.emplace_back(Actor(localObservationDimension_, actionDimension_));
//             actors_.back()->to(torch::kF64);
//             actorOptimizers_.emplace_back(actors_.back()->parameters(), learningRate);

//             auto logAlpha = torch::log(torch::tensor({initialAlpha}, torch::kF64)).set_requires_grad(true);
//             logAlphas_.push_back(logAlpha);
//             alphaOptimizers_.emplace_back(std::vector<torch::Tensor>{logAlphas_.back()}, alphaLearningRate);
//             temperatures_[i] = initialAlpha;
//         }

//         // Per-agent twin critics (centralized inputs)
//         critics1_.resize(numberOfAgents_);
//         critics2_.resize(numberOfAgents_);
//         targetCritics1_.resize(numberOfAgents_);
//         targetCritics2_.resize(numberOfAgents_);
//         critic1Optimizers_.resize(numberOfAgents_);
//         critic2Optimizers_.resize(numberOfAgents_);

//         for (int i = 0; i < numberOfAgents_; ++i) {
//             critics1_[i] = Critic(globalStateDimension_, jointActionDimension_);
//             critics2_[i] = Critic(globalStateDimension_, jointActionDimension_);
//             targetCritics1_[i] = Critic(globalStateDimension_, jointActionDimension_);
//             targetCritics2_[i] = Critic(globalStateDimension_, jointActionDimension_);

//             critics1_[i]->to(torch::kF64);
//             critics2_[i]->to(torch::kF64);
//             targetCritics1_[i]->to(torch::kF64);
//             targetCritics2_[i]->to(torch::kF64);

//             hardCopy(critics1_[i], targetCritics1_[i]);
//             hardCopy(critics2_[i], targetCritics2_[i]);

//             critic1Optimizers_[i] = std::make_unique<torch::optim::Adam>(critics1_[i]->parameters(), learningRate);
//             critic2Optimizers_[i] = std::make_unique<torch::optim::Adam>(critics2_[i]->parameters(), learningRate);
//         }
//     }

//     // Forward a batch global state to extract the local slice for agent i
//     torch::Tensor sliceLocalObservation(const torch::Tensor& batchGlobalState, int agentIndex) const {
//         const int start = agentIndex * localObservationDimension_;
//         return batchGlobalState.index({torch::indexing::Slice(),
//                                        torch::indexing::Slice(start, start + localObservationDimension_)});
//     }

//     // Concatenate along feature dimension
//     static torch::Tensor concatAlongFeature(const std::vector<torch::Tensor>& parts) {
//         return torch::cat(parts, /*dim=*/1);
//     }

//     // Stochastic action sampling for a single-step (batch size = 1 per agent input)
//     torch::Tensor act(const std::vector<torch::Tensor>& localObservationsBatch1) {
//         std::vector<torch::Tensor> actionParts;
//         actionParts.reserve(numberOfAgents_);
//         for (int i = 0; i < numberOfAgents_; ++i) {
//             auto [action_i, logProb_i] = actors_[i]->sample(localObservationsBatch1[i]);
//             (void)logProb_i; // not needed in inference
//             actionParts.push_back(action_i);
//         }
//         return concatAlongFeature(actionParts); // shape [1, jointActionDimension_]
//     }

//     // One training step using a minibatch sampled from the buffer
//     // Buffer sample shapes:
//     //   S:  [B, globalStateDimension_]
//     //   A:  [B, jointActionDimension_]
//     //   R:  [B, numberOfAgents_]
//     //   S2: [B, globalStateDimension_]
//     //   D:  [B, 1]
//     void trainStep(ReplayBuffer& buffer, size_t batchSize) {
//         if (buffer.size() < batchSize) return;

//         auto [batchState, batchAction, batchReward, batchNextState, batchDone] = buffer.sample(batchSize);

//         // Next joint action sampled from current actors (for all agents)
//         std::vector<torch::Tensor> nextActionParts, nextLogProbParts;
//         nextActionParts.reserve(numberOfAgents_);
//         nextLogProbParts.reserve(numberOfAgents_);
//         for (int i = 0; i < numberOfAgents_; ++i) {
//             auto nextLocalObs_i = sliceLocalObservation(batchNextState, i);
//             auto [nextAction_i, nextLogProb_i] = actors_[i]->sample(nextLocalObs_i);
//             nextActionParts.push_back(nextAction_i);
//             nextLogProbParts.push_back(nextLogProb_i);
//         }
//         auto batchNextAction = concatAlongFeature(nextActionParts); // [B, jointActionDimension_]

//         // Sum of next log-probabilities across agents (appears in target)
//         auto sumNextLogProb = torch::zeros_like(nextLogProbParts[0]);
//         for (auto& lp : nextLogProbParts) sumNextLogProb = sumNextLogProb + lp;

//         // ---- Per-agent critic updates (independent centralized Q for each agent) ----
//         for (int i = 0; i < numberOfAgents_; ++i) {
//             torch::Tensor targetQ_i;
//             {
//                 torch::NoGradGuard noGrad;

//                 auto q1Next = targetCritics1_[i]->forward(batchNextState, batchNextAction); // [B,1]
//                 auto q2Next = targetCritics2_[i]->forward(batchNextState, batchNextAction); // [B,1]
//                 auto minQNext = torch::min(q1Next, q2Next);

//                 // R[:, i] -> shape [B, 1]
//                 auto rewardColumn_i = batchReward.index({torch::indexing::Slice(), i}).view({-1, 1});

//                 targetQ_i = rewardColumn_i
//                     + gamma_ * (1.0 - batchDone) * (minQNext - temperatures_[i] * sumNextLogProb);
//             }

//             auto q1 = critics1_[i]->forward(batchState, batchAction);
//             auto q2 = critics2_[i]->forward(batchState, batchAction);
//             auto criticLoss = torch::mse_loss(q1, targetQ_i) + torch::mse_loss(q2, targetQ_i);

//             critic1Optimizers_[i]->zero_grad();
//             critic2Optimizers_[i]->zero_grad();
//             criticLoss.backward();
//             critic1Optimizers_[i]->step();
//             critic2Optimizers_[i]->step();
//         }

//         // ---- Per-agent actor and temperature updates ----
//         // Re-sample current joint actions A_pi for policy evaluation
//         std::vector<torch::Tensor> currentActionParts(numberOfAgents_), currentLogProbParts(numberOfAgents_);
//         for (int i = 0; i < numberOfAgents_; ++i) {
//             auto localObs_i = sliceLocalObservation(batchState, i);
//             auto [action_i, logProb_i] = actors_[i]->sample(localObs_i);
//             currentActionParts[i] = action_i;
//             currentLogProbParts[i] = logProb_i;
//         }
//         auto jointActionFromPolicy = concatAlongFeature(currentActionParts);

//         for (int i = 0; i < numberOfAgents_; ++i) {
//             // Use agent i's own centralized critic for policy improvement
//             auto qMin_i = torch::min(
//                 critics1_[i]->forward(batchState, jointActionFromPolicy),
//                 critics2_[i]->forward(batchState, jointActionFromPolicy)
//             );
//             auto actorLoss_i = (temperatures_[i] * currentLogProbParts[i] - qMin_i).mean();

//             actorOptimizers_[i].zero_grad();
//             actorLoss_i.backward();
//             actorOptimizers_[i].step();

//             // Temperature (alpha) update towards target entropy
//             auto alphaLoss_i = -(logAlphas_[i] * (currentLogProbParts[i].detach() + targetEntropy_)).mean();
//             alphaOptimizers_[i].zero_grad();
//             alphaLoss_i.backward();
//             alphaOptimizers_[i].step();
//             temperatures_[i] = logAlphas_[i].exp().item<double>();
//         }

//         // ---- Soft update target critics ----
//         torch::NoGradGuard noGrad;
//         for (int i = 0; i < numberOfAgents_; ++i) {
//             softUpdate(critics1_[i], targetCritics1_[i]);
//             softUpdate(critics2_[i], targetCritics2_[i]);
//         }
//     }

// private:
//     int numberOfAgents_;
//     int localObservationDimension_;
//     int actionDimension_;
//     int globalStateDimension_;
//     int jointActionDimension_;

//     double gamma_;
//     double tau_;
//     double targetEntropy_;

//     // Per-agent policies and temperatures
//     std::vector<Actor> actors_;
//     std::vector<torch::optim::Adam> actorOptimizers_;

//     std::vector<torch::Tensor> logAlphas_;
//     std::vector<double> temperatures_;
//     std::vector<torch::optim::Adam> alphaOptimizers_;

//     // Per-agent centralized twin critics + targets
//     std::vector<Critic> critics1_, critics2_, targetCritics1_, targetCritics2_;
//     std::vector<std::unique_ptr<torch::optim::Adam>> critic1Optimizers_, critic2Optimizers_;

//     // Utilities
//     void softUpdate(const Critic& source, Critic& target) {
//         auto targetParams = target->parameters();
//         auto sourceParams = source->parameters();
//         for (size_t p = 0; p < targetParams.size(); ++p) {
//             targetParams[p].data().mul_(1.0 - tau_);
//             targetParams[p].data().add_(tau_ * sourceParams[p].data());
//         }
//     }
//     void hardCopy(const Critic& source, Critic& target) {
//         auto targetParams = target->parameters();
//         auto sourceParams = source->parameters();
//         for (size_t p = 0; p < targetParams.size(); ++p) {
//             targetParams[p].data().copy_(sourceParams[p].data());
//         }
//     }
// };
