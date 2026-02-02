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
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    # Enforce deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def f05_score(preds, labels, threshold=0.5, epsilon=1e-7):
    """
    Calculates the F0.5 score for binary segmentation, weighting precision higher than recall.

    Args:
        preds (torch.Tensor or np.ndarray): Predicted probabilities (0.0 to 1.0).
        labels (torch.Tensor or np.ndarray): Ground truth binary labels (0 or 1).
        threshold (float): Threshold to convert probabilities to binary mask.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The calculated F0.5 score.
    """
    # Convert inputs to torch tensors if they are numpy arrays
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    # Ensure labels are on the same device as preds
    if preds.device != labels.device:
        labels = labels.to(preds.device)

    # Binarize predictions based on threshold
    pred_mask = (preds > threshold).float()
    labels = labels.float()

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    tp = (pred_mask * labels).sum()
    fp = (pred_mask * (1 - labels)).sum()
    fn = ((1 - pred_mask) * labels).sum()

    # Calculate Precision and Recall
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    # Calculate F0.5 Score
    # Formula: (1 + beta^2) * p * r / (beta^2 * p + r)
    beta = 0.5
    beta_sq = beta**2

    score = ((1 + beta_sq) * precision * recall) / (
        beta_sq * precision + recall + epsilon
    )

    return score.item()


def rle_encoding(mask):
    """
    Converts a binary mask to Run-Length Encoding (RLE) format for submission.
    The pixels are numbered from left to right, then top to bottom.

    Args:
        mask (np.ndarray): Binary mask where 1 indicates ink and 0 indicates background.

    Returns:
        str: Space-delimited list of pairs (start_position, run_length).
    """
    # Flatten the mask in row-major order (C-style)
    pixels = mask.flatten()

    # Pad with zeros at the beginning and end to efficiently detect state changes
    # This handles runs starting at index 0 or ending at the last index
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    # np.where returns indices in the padded array.
    # +1 adjusts to 1-based indexing required by the competition format (mostly)
    # but primarily it aligns the diff index to the start of the change.
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # 'runs' currently holds [start_1, end_1, start_2, end_2, ...]
    # We need [start_1, length_1, start_2, length_2, ...]
    # Length = end - start
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def save_checkpoint(
    model, optimizer, scheduler, epoch, score, path=Config.CHECKPOINT_PATH
):
    """
    Saves the model, optimizer, and scheduler states to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler state.
        epoch (int): The current training epoch.
        score (float): The validation score at this checkpoint.
        path (str): The file path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "epoch": epoch,
        "score": score,
    }

    torch.save(state, path)


def load_checkpoint(
    model,
    optimizer=None,
    scheduler=None,
    path=Config.CHECKPOINT_PATH,
    device=Config.DEVICE,
):
    """
    Loads the model, optimizer, and scheduler states from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        path (str): The file path of the checkpoint.
        device (torch.device): The device to map the checkpoint to.

    Returns:
        tuple: (score, epoch) from the checkpoint. Returns (0.0, 0) if file not found.
    """
    if not os.path.exists(path):
        return 0.0, 0

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        # Check if scheduler state exists and is not None
        if checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    score = checkpoint.get("score", 0.0)
    epoch = checkpoint.get("epoch", 0)

    return score, epoch
