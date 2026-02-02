import torch
import torch.optim as optim
from collections import defaultdict
from library.config import Config


class Lookahead(optim.Optimizer):
    """
    Lookahead Optimizer Wrapper.

    Maintains a set of 'slow' weights and 'fast' weights. The fast weights are updated
    by an inner optimizer (e.g., AdamW). Every `k` steps, the slow weights are updated
    by interpolating towards the fast weights, and the fast weights are reset to the
    new slow weights. This stabilizes training.

    Reference: "Lookahead Optimizer: k steps forward, 1 step back" (Zhang et al., 2019)
    """

    def __init__(self, optimizer: optim.Optimizer, k: int = 5, alpha: float = 0.5):
        """
        Args:
            optimizer (torch.optim.Optimizer): The base optimizer (inner loop).
            k (int): The number of steps to look ahead before synchronizing.
            alpha (float): The interpolation coefficient (step size for slow weights).
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not k >= 1:
            raise ValueError(f"Invalid k value: {k}")

        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha

        # Point to the base optimizer's param_groups to ensure updates propagate
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)

        # Initialize slow weights for all parameters
        for group in self.param_groups:
            group["lookahead_step"] = 0
            for p in group["params"]:
                if p.requires_grad:
                    # Store the slow weights in the state dictionary
                    self.state[p]["slow_buffer"] = torch.clone(p.data).detach()

    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        # 1. Update fast weights using the base optimizer
        loss = self.optimizer.step(closure)

        # 2. Update slow weights and synchronize if k steps reached
        for group in self.param_groups:
            if "lookahead_step" not in group:
                group["lookahead_step"] = 0

            group["lookahead_step"] += 1

            if group["lookahead_step"] % self.k == 0:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    param_state = self.state[p]
                    if "slow_buffer" not in param_state:
                        param_state["slow_buffer"] = torch.clone(p.data).detach()

                    slow = param_state["slow_buffer"]
                    fast = p.data

                    # slow = slow + alpha * (fast - slow)
                    # We use add_ for in-place update: slow.add_(other, alpha=val) -> slow + val * other
                    slow.add_(fast - slow, alpha=self.alpha)

                    # Reset fast weights to the new slow weights
                    fast.copy_(slow)

        return loss

    def state_dict(self):
        """
        Returns the state of the optimizer as a dict.
        Includes both the base optimizer's state and the Lookahead slow weights.
        """
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            "state": self.state,
            "param_groups": self.param_groups,
        }
        return {"fast_state": fast_state_dict, "slow_state": slow_state}

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.
        """
        self.optimizer.load_state_dict(state_dict["fast_state"])
        slow_state = state_dict["slow_state"]
        self.state.update(slow_state["state"])
        self.param_groups = slow_state["param_groups"]
        # Ensure the base optimizer points to the updated param_groups
        self.optimizer.param_groups = self.param_groups

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group):
        param_group["lookahead_step"] = 0
        self.optimizer.add_param_group(param_group)
        for p in param_group["params"]:
            if p.requires_grad:
                self.state[p]["slow_buffer"] = torch.clone(p.data).detach()


def get_optimizer(model: torch.nn.Module, config: Config) -> optim.Optimizer:
    """
    Factory function to create the Lookahead optimizer wrapping AdamW.

    Args:
        model (torch.nn.Module): The model to optimize.
        config (Config): Configuration object containing hyperparameters.

    Returns:
        optim.Optimizer: The configured Lookahead optimizer.
    """
    # Initialize Base Optimizer (AdamW)
    base_optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Wrap with Lookahead
    optimizer = Lookahead(
        base_optimizer, k=config.LOOKAHEAD_K, alpha=config.LOOKAHEAD_ALPHA
    )

    return optimizer
