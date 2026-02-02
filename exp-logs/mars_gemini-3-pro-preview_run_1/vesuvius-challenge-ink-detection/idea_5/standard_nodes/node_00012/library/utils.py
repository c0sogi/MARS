import os
import random
import numpy as np
import torch
from library import config


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).
    The metric checks that the pairs are sorted, positive, and the decoded pixel
    values are not duplicated. The pixels are numbered from left to right,
    then top to bottom: 1 is pixel (1,1), 2 is pixel (1,2), etc.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited run-length encoded string (start length start length ...).
    """
    # Flatten the mask (row-major: left-to-right, then top-to-bottom)
    pixels = mask.flatten()

    # Pad with zeros at start and end to detect transitions at the boundaries
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    # np.where returns indices in the padded array
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[::2] are the start indices (inclusive)
    # runs[1::2] are the end indices (exclusive)
    # Calculate lengths: length = end - start
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def fbeta_score(preds, targets, beta=0.5, threshold=0.5, epsilon=1e-7):
    """
    Calculates the F-beta score for binary classification.

    Args:
        preds (torch.Tensor): Predicted probabilities or logits.
        targets (torch.Tensor): Ground truth binary labels.
        beta (float): Beta value for F-score (default 0.5 weights precision higher).
        threshold (float): Threshold to binarize predictions.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The F-beta score.
    """
    # Ensure inputs are tensors
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)

    # Binarize predictions based on threshold
    y_pred = (preds > threshold).float()
    y_true = targets.float()

    # Flatten tensors to calculate global metrics
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    tp = (y_pred * y_true).sum()
    fp = (y_pred * (1 - y_true)).sum()
    fn = ((1 - y_pred) * y_true).sum()

    # Calculate F-beta score
    beta_sq = beta**2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = numerator / (denominator + epsilon)

    return score.item()


def optimize_threshold(preds, targets, beta=0.5, num_steps=100):
    """
    Performs a post-hoc search to find the optimal probability threshold
    that maximizes the F-beta score on the provided data.

    Args:
        preds (torch.Tensor or np.ndarray): Predicted probabilities.
        targets (torch.Tensor or np.ndarray): Ground truth labels.
        beta (float): Beta value for F-score.
        num_steps (int): Number of threshold steps to evaluate between 0 and 1.

    Returns:
        tuple: (best_threshold, best_score)
    """
    # Convert numpy arrays to tensors if necessary
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    # Move targets to the same device as predictions
    device = preds.device
    targets = targets.to(device)

    best_threshold = 0.5
    best_score = -1.0

    # Generate range of thresholds to test
    thresholds = torch.linspace(0.01, 0.99, steps=num_steps).to(device)

    # Pre-flatten tensors for efficiency
    p_flat = preds.view(-1)
    t_flat = targets.float().view(-1)
    beta_sq = beta**2

    # Iterate through thresholds
    for thresh in thresholds:
        th_val = thresh.item()

        # Binarize
        y_pred = (p_flat > th_val).float()

        # Calculate metrics
        tp = (y_pred * t_flat).sum()
        fp = (y_pred * (1 - t_flat)).sum()
        fn = ((1 - y_pred) * t_flat).sum()

        numerator = (1 + beta_sq) * tp
        denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

        if denominator == 0:
            score_val = 0.0
        else:
            score_val = (numerator / denominator).item()

        if score_val > best_score:
            best_score = score_val
            best_threshold = th_val

    return best_threshold, best_score


def save_checkpoint(model, optimizer, epoch, score, path):
    """
    Saves the model state, optimizer state, and training metadata.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): Current training epoch.
        score (float): Validation score at this epoch.
        path (str): Full file path to save the checkpoint.
    """
    # Ensure the directory exists
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "score": score,
    }

    torch.save(state, path)


def load_checkpoint(model, path, optimizer=None):
    """
    Loads a checkpoint into the model and optionally the optimizer.

    Args:
        model (torch.nn.Module): The model to load weights into.
        path (str): Path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.

    Returns:
        dict: The full checkpoint dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    # Load checkpoint to the configured device
    checkpoint = torch.load(path, map_location=config.DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
