import Policy
from EGraph import *
from EGraphGen import *
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

EXIT = 0
SELECT = 1
NONE = 2

class Runner:
    def __init__(self, 
                 model: Policy.Policy, 
                 valuation_fun: Callable[[ExprNode], float],
                 step_penalty: float,
                 growth_penalty: float,
                 exit_factor: float):
        self.model = model
        self.valuation = valuation_fun
        self.step_penalty = step_penalty
        self.growth_penalty = growth_penalty
        self.exit_factor = exit_factor

    def init_episode(self, expr: ExprNode):
        self.step_count = 0
        self.initial = self.valuation(expr)
        self.current_best = self.valuation(expr)
        self.eg = EGraph()
        self.eclass_id = insert_expr(expr, self.eg)

    def get_best(self):
        return self.eg.extract(self.eclass_id, self.valuation)

    def step(self, actions: dict[ENode, int], max_steps: int) -> tuple[float, bool]:
        if self.step_count >= max_steps:
            return -100000, True

        if EXIT in actions.values():
            return (self.initial - self.get_best()[1]) * self.exit_factor, True

        if SELECT in actions.values():
            expr, _ = self.eg.extract(self.eclass_id, self.valuation)
            
            self.eg = EGraph()
            self.eclass_id = insert_expr(expr, self.eg)

            new_best = self.get_best()[1]

            reward = (self.initial - new_best) - self.step_penalty
            
            self.initial = new_best
            self.current_best = new_best

            self.step_count += 1

            return reward, False
        
        for enode, action in actions.items():
            if action == 2:
                continue

            self.eg.apply_rule_at(self.model.rules[action - 3], enode)

        self.eclass_id = self.eg.find(self.eclass_id)

        new_best = self.get_best()[1]

        reward = (self.current_best - new_best) - self.step_penalty
        self.current_best = new_best

        self.step_count += 1

        return reward, False

    
    def run_episode(self, expr: ExprNode, log: bool = True, max_steps: int = 100_000) -> ExprNode:
        self.init_episode(expr)

        exit_sgn = False

        while not exit_sgn:
            print(f"\rrewrite step number: {self.step_count}", end="", flush=True)

            nodes, logits = self.model(self.eg, self.eclass_id)

            actions: dict[ENode, int] = {}

            for i in range(len(nodes)):
                actions[nodes[i]] = logits[i].argmax().item()

            _, exit_sgn = self.step(actions, max_steps)

        print("\nexited")

        return self.get_best()[0]
    
    def run_episode_train(
        self,
        expr:      ExprNode,
        max_steps: int   = 100_000,
        gamma:     float = 0.99,
        entropy_coeff: float = 0.01,
        device:    torch.device = torch.device("cpu")
    ) -> tuple[torch.Tensor, float]:
        self.init_episode(expr)

        log_probs_per_step: list[torch.Tensor] = []   # one scalar per step
        entropies_per_step: list[torch.Tensor] = []
        rewards:            list[float]        = []
        exit_sgn = False

        while not exit_sgn:
            nodes, logits = self.model(self.eg, self.eclass_id)
            # logits: list[Tensor[action_count]], one per node

            step_log_probs = []
            step_entropies = []
            actions: dict[ENode, int] = {}

            for i, node in enumerate(nodes):
                dist   = torch.distributions.Categorical(logits=logits[i].to(device))
                action = dist.sample()                     # explore
                step_log_probs.append(dist.log_prob(action))
                step_entropies.append(dist.entropy())
                actions[node] = action.item()

            # sum over nodes: one scalar per step
            log_probs_per_step.append(torch.stack(step_log_probs).sum())
            entropies_per_step.append(torch.stack(step_entropies).sum())

            reward, exit_sgn = self.step(actions, max_steps)
            rewards.append(reward)

        # --------------------------------------------------------------
        # Discounted returns  G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …
        # --------------------------------------------------------------
        returns, G = [], 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

        # Normalise to reduce variance (skip if only 1 step)
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # --------------------------------------------------------------
        # REINFORCE loss  =  -Σ log π(a|s) · G_t
        # Entropy bonus   = -Σ H(π)   (encourages exploration)
        # --------------------------------------------------------------
        log_probs_t  = torch.stack(log_probs_per_step)
        entropies_t  = torch.stack(entropies_per_step)

        policy_loss  = -(log_probs_t * returns_t).sum()
        entropy_loss = -entropies_t.sum()               # negative because we want to maximise entropy
        loss         = policy_loss + entropy_coeff * entropy_loss

        return loss, float(sum(rewards))