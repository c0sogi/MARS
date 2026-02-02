import os
import random
import shutil
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and accuracy during training.
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


def save_checkpoint(
    state, is_best, filename="checkpoint.pth", output_dir=Config.OUTPUT_DIR
):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model_state_dict, optimizer, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Name of the checkpoint file.
        output_dir (str): Directory to save the checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(output_dir, "best_model.pth")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device=Config.DEVICE):
    """
    Loads a model checkpoint.

    Args:
        path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (optional): The scheduler to load state into.
        device (str): Device to map the location to.

    Returns:
        dict: The loaded checkpoint dictionary, or None if path doesn't exist.
    """
    if not os.path.exists(path):
        print(f"Checkpoint not found at {path}")
        return None

    checkpoint = torch.load(path, map_location=device)

    # Load model weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if provided
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def calculate_log_loss(y_true, y_pred):
    """
    Calculates Multi Class Log Loss.

    Args:
        y_true (array-like): Ground truth labels (indices).
        y_pred (array-like): Predicted probabilities (N_samples, N_classes).

    Returns:
        float: The log loss value.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # sklearn log_loss requires y_true to be labels or one-hot
    # If y_true is indices, log_loss handles it correctly if labels parameter is handled
    # or if it infers classes. Explicitly providing labels is safer if a batch misses a class.

    # However, for validation sets, we usually have all classes or we trust sklearn's inference
    # on the passed batch. To be robust for batches:
    # We assume y_pred shape determines the classes.

    # Clip predictions to avoid log(0) error, though sklearn does this internally.
    # We rely on sklearn implementation.

    # Note: If y_pred contains probabilities for all classes (0 to 119),
    # sklearn expects y_true to contain values in 0..119.

    return log_loss(y_true, y_pred, labels=list(range(Config.NUM_CLASSES)))
