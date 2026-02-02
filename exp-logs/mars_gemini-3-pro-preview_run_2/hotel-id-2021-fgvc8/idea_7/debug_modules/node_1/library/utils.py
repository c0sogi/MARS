import torch
import numpy as np
import random
import os
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_optimizer(model):
    """
    Constructs the AdamW optimizer with parameter groups for backbone and head.
    Applies weight decay to weights but excludes bias and normalization layers.
    Uses Config.LR for backbone and Config.HEAD_LR for the head.

    Args:
        model (torch.nn.Module): The model to optimize.

    Returns:
        torch.optim.Optimizer: The configured AdamW optimizer.
    """
    # Define parameter groups
    backbone_params_decay = []
    backbone_params_no_decay = []
    head_params_decay = []
    head_params_no_decay = []

    # Layers to exclude from weight decay
    no_decay = ["bias", "LayerNorm.weight", "BatchNorm2d.weight", "BatchNorm1d.weight"]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Heuristic to identify head parameters based on naming convention
        # Assumes head layers contain 'head', 'arcface', or 'sub_center' in their name
        is_head = (
            "head" in name or "arcface" in name or "sub_center" in name or "fc" in name
        )

        # Check if parameter should have weight decay
        has_decay = not any(nd in name for nd in no_decay)

        if is_head:
            if has_decay:
                head_params_decay.append(param)
            else:
                head_params_no_decay.append(param)
        else:
            if has_decay:
                backbone_params_decay.append(param)
            else:
                backbone_params_no_decay.append(param)

    optimizer_grouped_parameters = [
        {
            "params": backbone_params_decay,
            "weight_decay": Config.WEIGHT_DECAY,
            "lr": Config.LR,
        },
        {"params": backbone_params_no_decay, "weight_decay": 0.0, "lr": Config.LR},
        {
            "params": head_params_decay,
            "weight_decay": Config.WEIGHT_DECAY,
            "lr": Config.HEAD_LR,
        },
        {"params": head_params_no_decay, "weight_decay": 0.0, "lr": Config.HEAD_LR},
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    return optimizer


def get_scheduler(optimizer):
    """
    Constructs the CosineAnnealingLR scheduler based on Config.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule.

    Returns:
        torch.optim.lr_scheduler.CosineAnnealingLR: The configured scheduler.
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )
    return scheduler
