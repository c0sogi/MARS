import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    """

    def __init__(self, name: str = "Metric", fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def kl_divergence_score(
    y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-7
) -> float:
    """
    Computes the Kullback-Leibler divergence between predicted probabilities and target probabilities.

    Args:
        y_pred: Predicted probabilities (batch_size, num_classes).
        y_true: Target probabilities (batch_size, num_classes).
        eps: Small epsilon to prevent log(0).

    Returns:
        The scalar KL divergence score (batchmean).
    """
    # Ensure inputs are tensors
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)

    # Clip predictions to avoid log(0)
    y_pred = torch.clamp(y_pred, min=eps, max=1.0)

    # PyTorch KLDivLoss expects input to be log-probabilities
    # and target to be probabilities.
    # Reduction 'batchmean' mathematically aligns with the KL definition averaged over batch.
    loss_func = torch.nn.KLDivLoss(reduction="batchmean")
    loss = loss_func(torch.log(y_pred), y_true)

    return loss.item()


def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: str):
    """
    Saves the model checkpoint.

    Args:
        state: Dictionary containing model state, optimizer state, etc.
        is_best: Boolean indicating if this is the best model so far.
        checkpoint_dir: Directory to save the checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save the latest checkpoint
    filename = os.path.join(checkpoint_dir, "checkpoint_last.pth")
    torch.save(state, filename)

    # If it's the best model, save a copy as best_model.pth
    if is_best:
        best_filename = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_filename)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    device: str = "cpu",
):
    """
    Loads a model checkpoint.

    Args:
        checkpoint_path: Path to the .pth file.
        model: The model instance to load weights into.
        optimizer: (Optional) Optimizer to load state into.
        scheduler: (Optional) Scheduler to load state into.
        device: Device to map the location to.

    Returns:
        start_epoch (int), best_score (float)
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    # Handle DataParallel/DistributedDataParallel keys if necessary (remove 'module.' prefix)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    # Load optimizer state
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_score = checkpoint.get("best_score", float("inf"))

    return start_epoch, best_score


def get_lr(optimizer):
    """
    Helper to retrieve the current learning rate from the optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]
