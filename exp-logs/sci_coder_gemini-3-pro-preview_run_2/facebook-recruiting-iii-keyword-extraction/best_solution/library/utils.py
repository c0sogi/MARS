import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
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


def clean_text(text):
    """
    Cleans raw text by removing HTML tags, converting to lowercase,
    and normalizing whitespace.
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Normalize whitespace (replace multiple spaces/tabs/newlines with single space)
    text = re.sub(r"\s+", " ", text).strip()

    return text


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance by down-weighting easy examples.
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from the model (before sigmoid). Shape: (batch_size, num_classes)
            targets: Binary targets (0 or 1). Shape: (batch_size, num_classes)
        """
        # binary_cross_entropy_with_logits is numerically stable
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # p_t is the probability of the true class
        # bce_loss = -log(p_t) -> p_t = exp(-bce_loss)
        pt = torch.exp(-bce_loss)

        # Focal Loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def calculate_f1(logits, targets, threshold=0.5):
    """
    Calculates the Mean F1-Score (samples average) for multi-label classification.

    Args:
        logits: Model outputs (logits). Can be torch.Tensor or numpy array.
        targets: Ground truth labels. Can be torch.Tensor or numpy array.
        threshold: Threshold for converting probabilities to binary predictions.

    Returns:
        float: Mean F1 Score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu()
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits).numpy()
    else:
        # Assume logits are already numpy, apply sigmoid
        probs = 1 / (1 + np.exp(-logits))

    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Binarize predictions
    preds = (probs > threshold).astype(int)

    # Calculate F1 score with 'samples' average (standard for this task)
    return f1_score(targets, preds, average="samples", zero_division=0)


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model state to a file.

    Args:
        state (dict): State dictionary containing model parameters, optimizer, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Path to save the checkpoint.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the checkpoint
    torch.save(state, filename)

    # If it's the best model, create a copy named 'best_model.pth' in the same directory
    if is_best:
        dirname = os.path.dirname(filename)
        best_path = os.path.join(dirname, "best_model.pth")
        torch.save(state, best_path)


def load_checkpoint(model, optimizer=None, filename="checkpoint.pth"):
    """
    Loads a model checkpoint.

    Args:
        model: The model instance to load weights into.
        optimizer: The optimizer instance to load state into (optional).
        filename: Path to the checkpoint file.

    Returns:
        tuple: (start_epoch, best_f1)
    """
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location=Config.DEVICE)

        model.load_state_dict(checkpoint["state_dict"])

        if optimizer and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        start_epoch = checkpoint.get("epoch", 0)
        best_f1 = checkpoint.get("best_f1", 0.0)

        print(
            f"Loaded checkpoint '{filename}' (epoch {start_epoch}, best_f1 {best_f1})"
        )
        return start_epoch, best_f1
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, 0.0
