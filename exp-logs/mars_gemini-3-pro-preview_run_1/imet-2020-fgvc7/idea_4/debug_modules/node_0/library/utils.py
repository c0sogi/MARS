import os
import random
import shutil
import numpy as np
import torch
from copy import deepcopy
from sklearn.metrics import f1_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
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


class ModelEMA:
    """
    Model Exponential Moving Average.
    Maintains a moving average of model parameters for better generalization.
    """

    def __init__(self, model, decay=0.9999):
        """
        Args:
            model (nn.Module): The model to track.
            decay (float): The decay factor for EMA.
        """
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        # Ensure parameters in EMA model do not require gradients
        for param in self.module.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the EMA model parameters based on the current model.

        Args:
            model (nn.Module): The current training model.
        """
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.module.state_dict().items():
                if k in msd:
                    model_v = msd[k].detach()
                    # Update floating point parameters/buffers with decay
                    if v.dtype.is_floating_point:
                        v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)
                    # Copy integer buffers (e.g., num_batches_tracked) directly
                    else:
                        v.copy_(model_v)


def calculate_micro_f1(logits, targets, threshold=0.5):
    """
    Calculates the Micro-averaged F1 score.

    Args:
        logits (torch.Tensor or np.ndarray): Model outputs (logits).
        targets (torch.Tensor or np.ndarray): Ground truth labels (binary).
        threshold (float): Probability threshold for binary classification.

    Returns:
        float: Micro F1 score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Apply sigmoid to convert logits to probabilities
    # Assumes logits are provided (common with BCEWithLogitsLoss)
    if isinstance(logits, torch.Tensor):
        probs = torch.sigmoid(logits).numpy()
    else:
        probs = 1 / (1 + np.exp(-logits))

    # Binarize predictions
    preds = (probs > threshold).astype(int)

    return f1_score(targets, preds, average="micro")


def save_checkpoint(state, is_best, filepath=None):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filepath (str, optional): Path to save the checkpoint. Defaults to working dir.
    """
    if filepath is None:
        filepath = os.path.join(Config.WORKING_DIR, "checkpoint.pth")

    torch.save(state, filepath)

    if is_best:
        best_path = Config.MODEL_SAVE_PATH
        shutil.copyfile(filepath, best_path)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Loads a model checkpoint.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (optional): Scheduler to load state into.
        device (str): Device to map location.

    Returns:
        tuple: (start_epoch, best_score)
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint not found at {filepath}")
        return 0, 0.0

    checkpoint = torch.load(filepath, map_location=device)

    # Load model weights
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Resume from the next epoch
    start_epoch = checkpoint.get("epoch", -1) + 1
    best_score = checkpoint.get("best_score", 0.0)

    return start_epoch, best_score
