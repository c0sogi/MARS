import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, scheduler, epoch, score, filepath):
    """
    Saves the model checkpoint including optimizer and scheduler states.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        epoch (int): The current epoch.
        score (float): The validation score (metric) at this epoch.
        filepath (str): Path to save the checkpoint file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "score": float(score),
    }
    torch.save(state, filepath)


def load_checkpoint(
    filepath, model, optimizer=None, scheduler=None, device=Config.DEVICE
):
    """
    Loads a model checkpoint.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        device (str): Device to map the location to.

    Returns:
        tuple: (epoch, score) from the checkpoint. Returns (0, inf) if file not found.
    """
    if not os.path.exists(filepath):
        # Return default values if checkpoint doesn't exist
        return 0, float("inf")

    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("score", float("inf"))


class GroupLogMAE:
    """
    Calculates the Log of the Mean Absolute Error for each scalar coupling type,
    and then averages across types.

    Metric = Mean( Log( MAE(type_i) ) ) for all types i.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # Dictionary to store sum of absolute errors and counts for each type index
        # We initialize for all known types to ensure we track them even if a batch misses one
        self.sum_abs_errors = {i: 0.0 for i in range(Config.NUM_COUPLING_TYPES)}
        self.counts = {i: 0 for i in range(Config.NUM_COUPLING_TYPES)}

    def update(self, preds, targets, types):
        """
        Updates the metric with a batch of predictions.

        Args:
            preds (torch.Tensor or np.ndarray): Predicted values.
            targets (torch.Tensor or np.ndarray): Ground truth values.
            types (torch.Tensor or np.ndarray): Coupling type indices (integers).
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if isinstance(types, torch.Tensor):
            types = types.detach().cpu().numpy()

        # Ensure flat arrays
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)
        types = types.reshape(-1)

        abs_diffs = np.abs(preds - targets)

        # Iterate and accumulate
        # Note: For very large batches, vectorized operations with pandas/numpy grouping
        # might be faster, but this loop is sufficient for standard batch sizes (e.g., 128).
        for i in range(len(preds)):
            t = int(types[i])
            if t in self.sum_abs_errors:
                self.sum_abs_errors[t] += abs_diffs[i]
                self.counts[t] += 1

    def compute(self):
        """
        Computes the final metric.

        Returns:
            avg_log_mae (float): The average of the Log MAE across all types present.
            metrics_per_type (dict): Dictionary mapping type name to its Log MAE.
        """
        log_maes = []
        metrics_per_type = {}

        for t_idx in range(Config.NUM_COUPLING_TYPES):
            count = self.counts[t_idx]
            if count > 0:
                mae = self.sum_abs_errors[t_idx] / count
                # Calculate Log MAE. Clip to avoid log(0)
                mae = max(mae, 1e-9)
                log_mae = np.log(mae)

                log_maes.append(log_mae)

                # Get string representation of type
                type_name = Config.INVERSE_COUPLING_TYPE_MAP.get(t_idx, str(t_idx))
                metrics_per_type[type_name] = log_mae

        if not log_maes:
            return 0.0, {}

        avg_log_mae = np.mean(log_maes)
        return avg_log_mae, metrics_per_type


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss during training.
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
