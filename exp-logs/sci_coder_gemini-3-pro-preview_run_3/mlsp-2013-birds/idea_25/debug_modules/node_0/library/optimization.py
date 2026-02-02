import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import defaultdict
from library.config import Config
from library.models import get_llrd_params


class Lookahead(optim.Optimizer):
    """
    Implements Lookahead Optimizer.
    It wraps an inner optimizer (e.g., AdamW) and updates the "slow weights"
    in the direction of the "fast weights" every k steps.

    Reference: "Lookahead Optimizer: k steps forward, 1 step back"
    """

    def __init__(self, optimizer, k=5, alpha=0.5):
        """
        Args:
            optimizer (torch.optim.Optimizer): The inner optimizer.
            k (int): The number of steps to look ahead before synchronizing.
            alpha (float): The interpolation coefficient (step size) for the slow weights.
        """
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.defaults = dict(k=k, alpha=alpha, **optimizer.defaults)

        # Initialize slow weights
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.clone(fast.data).detach()

            slow = param_state["slow_param"]
            fast.data.mul_(self.alpha).add_(slow, alpha=1.0 - self.alpha)
            slow.data.copy_(fast.data)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)

        for group in self.param_groups:
            if "counter" not in group:
                group["counter"] = 0

            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
                self.update(group)

        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = state_dict["slow_state"]
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        self.optimizer.load_state_dict(fast_state_dict)

        # Restore slow state
        # Mapping needs to be handled carefully if params are recreated,
        # but standard PyTorch usage usually preserves object identity in memory during a session.
        # For strict checkpointing, we rely on the inner optimizer's restoration logic
        # and re-initialize slow params on the first step if missing, or manually map them here.
        # A simple re-init approach is often sufficient for resumption unless exact state is critical.

        # Here we manually restore the slow params to self.state
        self.state = defaultdict(dict)
        # Re-map based on current model parameters is complex without param_id mapping.
        # However, since Lookahead state is just the slow weights, we can often
        # just let them re-initialize from the loaded weights on the first step
        # if we accept a minor discontinuity, or implement complex id mapping.
        # For this competition context, we will rely on re-initialization from current weights
        # if the mapping isn't trivial, but let's try to restore if keys match.
        for k, v in slow_state_dict.items():
            self.state[k] = v


def get_optimizer(model):
    """
    Constructs the optimizer for the given model.
    Applies Layer-Wise Learning Rate Decay (LLRD) and wraps with Lookahead if configured.

    Args:
        model (BirdModel): The model to optimize.

    Returns:
        torch.optim.Optimizer: The configured optimizer.
    """
    # 1. Get Parameter Groups with LLRD
    # This assigns lower learning rates to earlier layers
    optimizer_grouped_parameters = get_llrd_params(
        model,
        base_lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        decay_rate=Config.LLRD_DECAY_RATE,
    )

    # 2. Create Inner Optimizer (AdamW)
    if Config.OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
    else:
        # Fallback to standard Adam
        optimizer = optim.Adam(
            optimizer_grouped_parameters,
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

    # 3. Wrap with Lookahead if enabled
    if Config.USE_LOOKAHEAD:
        optimizer = Lookahead(optimizer, k=5, alpha=0.5)

    return optimizer


def get_scheduler(optimizer, epochs):
    """
    Constructs the learning rate scheduler.
    Uses Cosine Annealing as per strategy.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer instance.
        epochs (int): Total number of training epochs.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: The configured scheduler.
    """
    if Config.SCHEDULER == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=Config.MIN_LR)
    else:
        # Fallback or default
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=Config.MIN_LR)

    return scheduler
