import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import defaultdict
from library.config import Config


def get_optimizer(model):
    """
    Constructs the optimizer for the given model.
    Uses uniform learning rate (no LLRD) and standard AdamW. Cite {solution_lesson_node_00073}
    """
    if Config.OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

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
