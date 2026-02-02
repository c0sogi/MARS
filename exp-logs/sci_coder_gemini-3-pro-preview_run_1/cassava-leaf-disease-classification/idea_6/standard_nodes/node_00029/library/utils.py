import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
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


def worker_init_fn(worker_id):
    """
    Worker initialization function to ensure reproducible data augmentation.
    Cite solution_lesson_node_00007.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_score(y_true, y_pred):
    """
    Calculates the categorization accuracy.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        float: Accuracy score.
    """
    return accuracy_score(y_true, y_pred)


def get_llrd_params(model, base_lr, weight_decay, decay_factor):
    """
    Generates parameter groups for Layer-wise Learning Rate Decay (LLRD) specifically
    tailored for ConvNeXt-like architectures.

    The decay is applied such that the head has the base_lr, and the learning rate
    decays geometrically as we move towards the input (stem).

    Args:
        model (nn.Module): The model to optimize.
        base_lr (float): The learning rate for the head (topmost layers).
        weight_decay (float): Weight decay coefficient.
        decay_factor (float): Multiplicative decay factor for lower layers.

    Returns:
        list: A list of dictionaries suitable for torch.optim.Optimizer.
    """
    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine the layer scale (0 = Head, increasing means deeper/earlier in network)
        # ConvNeXt structure: stem -> stages.0 -> stages.1 -> stages.2 -> stages.3 -> head
        if "stages.3" in name:
            scale = 1
        elif "stages.2" in name:
            scale = 2
        elif "stages.1" in name:
            scale = 3
        elif "stages.0" in name or "stem" in name:
            scale = 4
        else:
            # Head, GeM, Norms after backbone, etc.
            scale = 0

        # Calculate target learning rate for this parameter
        target_lr = base_lr * (decay_factor**scale)

        # Determine if weight decay should be applied
        # Standard practice: no weight decay on biases, layer norms, or 1D tensors
        if param.ndim <= 1 or "bias" in name or "norm" in name or "bn" in name:
            wd = 0.0
        else:
            wd = weight_decay

        # Group parameters by (lr, weight_decay) to avoid creating too many groups
        key = (target_lr, wd)
        if key not in param_groups:
            param_groups[key] = []
        param_groups[key].append(param)

    # Convert to list of dicts
    optimizer_params = []
    for (lr, wd), params in param_groups.items():
        optimizer_params.append({"params": params, "lr": lr, "weight_decay": wd})

    return optimizer_params
