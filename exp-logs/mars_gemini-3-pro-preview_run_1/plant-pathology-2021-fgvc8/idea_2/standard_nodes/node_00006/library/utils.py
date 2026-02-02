import os
import torch
import numpy as np
from sklearn.metrics import f1_score
from library.config import Config, seed_everything


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and other metrics over an epoch.
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


def calculate_f1_score(preds, targets, threshold=0.5):
    """
    Computes the Mean F1-Score (Macro-averaged) for multi-label classification.

    Args:
        preds (torch.Tensor or np.ndarray): Model predictions (logits).
        targets (torch.Tensor or np.ndarray): Ground truth labels (binary).
        threshold (float): Threshold to convert probabilities to binary labels.

    Returns:
        float: The macro-averaged F1 score.
    """
    # Convert PyTorch tensors to NumPy arrays if needed
    if isinstance(preds, torch.Tensor):
        # Apply sigmoid to logits
        preds = torch.sigmoid(preds).detach().cpu().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Binarize predictions based on threshold
    preds_binary = (preds > threshold).astype(int)
    targets_binary = targets.astype(int)

    # Compute Macro F1 Score
    return f1_score(targets_binary, preds_binary, average="macro")


def save_checkpoint(model, optimizer, scheduler, epoch, score, filename):
    """
    Saves the training checkpoint including model, optimizer, and scheduler states.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler (optional).
        epoch (int): Current epoch number.
        score (float): Validation metric score (e.g., F1).
        filename (str): Name of the checkpoint file.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    save_path = os.path.join(Config.WORKING_DIR, filename)

    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "score": score,
    }

    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    torch.save(state, save_path)


class Logger:
    """
    Logs messages to both the console and a text file.
    """

    def __init__(self, filename="train_log.txt"):
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        self.log_path = os.path.join(Config.WORKING_DIR, filename)

        # Initialize log file
        with open(self.log_path, "w") as f:
            f.write(f"Log initialized at {self.log_path}\n")

    def log(self, message):
        """
        Prints the message to stdout and appends it to the log file.
        """
        print(message)
        with open(self.log_path, "a") as f:
            f.write(str(message) + "\n")
