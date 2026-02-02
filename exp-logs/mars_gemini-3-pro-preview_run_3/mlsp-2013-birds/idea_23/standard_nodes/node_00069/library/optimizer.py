import torch
import torch.optim as optim
from collections import defaultdict


class Lookahead(optim.Optimizer):
    """
    Lookahead Optimizer Wrapper.

    Paper: "Lookahead Optimizer: k steps forward, 1 step back" (Zhang et al., 2019)
    Maintains a set of 'slow' weights that are updated every k steps by interpolating
    towards the 'fast' weights (which are updated by the base optimizer).
    """

    def __init__(self, optimizer, k=5, alpha=0.5):
        """
        Args:
            optimizer (torch.optim.Optimizer): The base optimizer (e.g., AdamW).
            k (int): Number of steps before synchronizing slow and fast weights.
            alpha (float): Interpolation factor (0.0 to 1.0). Higher alpha means
                           slow weights move more towards fast weights.
        """
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha

        # Initialize base Optimizer to setup hooks and internal state
        super(Lookahead, self).__init__(optimizer.param_groups, dict(k=k, alpha=alpha))

        # Restore reference to base optimizer's param_groups so updates are shared
        self.param_groups = self.optimizer.param_groups

        self.state = defaultdict(dict)
        self.defaults = dict(k=k, alpha=alpha)

        # Initialize slow parameters for all groups to match current parameters
        for group in self.param_groups:
            group["counter"] = 0
            for p in group["params"]:
                if p.requires_grad:
                    self.state[p]["slow_param"] = torch.clone(p.data).detach()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        loss = self.optimizer.step(closure)

        for group in self.param_groups:
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    # Retrieve slow param
                    param_state = self.state[p]
                    if "slow_param" not in param_state:
                        # Fallback if not initialized (should happen in init)
                        param_state["slow_param"] = torch.clone(p.data).detach()

                    slow = param_state["slow_param"]
                    fast = p.data

                    # Update slow weights: slow = slow + alpha * (fast - slow)
                    # This moves slow weights towards the current fast weights
                    slow.add_(fast - slow, alpha=self.alpha)

                    # Reset fast weights to the new slow weights
                    fast.copy_(slow)

        return loss

    def state_dict(self):
        """
        Returns the state of the optimizer as a dict.
        Includes base optimizer state and Lookahead-specific state (slow params).
        """
        base_state = self.optimizer.state_dict()

        # Serialize Lookahead state
        # We map params to their index in the group to avoid pickling Parameter objects directly
        slow_params_data = []
        counters = []

        for group in self.param_groups:
            counters.append(group["counter"])
            group_slow = []
            for p in group["params"]:
                if p in self.state and "slow_param" in self.state[p]:
                    group_slow.append(self.state[p]["slow_param"])
                else:
                    group_slow.append(None)
            slow_params_data.append(group_slow)

        return {
            "base_state": base_state,
            "slow_params": slow_params_data,
            "counters": counters,
            "alpha": self.alpha,
            "k": self.k,
        }

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.
        """
        # Load base optimizer state
        self.optimizer.load_state_dict(state_dict["base_state"])
        self.param_groups = self.optimizer.param_groups

        # Load Lookahead state
        self.alpha = state_dict["alpha"]
        self.k = state_dict["k"]
        counters = state_dict["counters"]
        slow_params_data = state_dict["slow_params"]

        for group_idx, group in enumerate(self.param_groups):
            group["counter"] = counters[group_idx]
            saved_slow = slow_params_data[group_idx]
            for p_idx, p in enumerate(group["params"]):
                if saved_slow[p_idx] is not None:
                    self.state[p]["slow_param"] = saved_slow[p_idx]


def get_optimizer(model, config):
    """
    Factory function to create the optimizer based on configuration.

    Args:
        model (torch.nn.Module): The model to optimize.
        config (Config): Configuration object.

    Returns:
        torch.optim.Optimizer: The configured optimizer.
    """
    # Base Optimizer: AdamW
    base_optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    if config.OPTIMIZER_NAME == "Lookahead_AdamW":
        optimizer = Lookahead(
            base_optimizer, k=config.LOOKAHEAD_K, alpha=config.LOOKAHEAD_ALPHA
        )
    elif config.OPTIMIZER_NAME == "AdamW":
        optimizer = base_optimizer
    else:
        # Default fallback
        print(
            f"Warning: Optimizer {config.OPTIMIZER_NAME} not explicitly supported. Using AdamW."
        )
        optimizer = base_optimizer

    return optimizer


def get_scheduler(optimizer, config):
    """
    Factory function to create the learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer.
        config (Config): Configuration object.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: The configured scheduler.
    """
    if config.SCHEDULER_NAME == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.EPOCHS, eta_min=config.MIN_LR
        )

    # Fallback or None if no scheduler requested
    return None
