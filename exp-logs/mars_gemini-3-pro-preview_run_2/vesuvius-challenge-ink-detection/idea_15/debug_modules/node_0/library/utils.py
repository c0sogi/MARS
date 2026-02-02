import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config, set_seed


def get_device():
    """
    Returns the PyTorch device specified in the configuration.
    """
    return torch.device(Config.DEVICE)


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility using the library's set_seed function.
    """
    set_seed(seed)


def rle_encoding(mask):
    """
    Converts a binary mask to Run-Length Encoding (RLE) format.

    The metric checks that pairs are sorted, positive, and decoded pixel values are not duplicated.
    Pixels are numbered from left to right, then top to bottom.

    Args:
        mask (np.ndarray or torch.Tensor): Binary mask (0 or 1).

    Returns:
        str: Space-delimited list of pairs (start_position, run_length).
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Ensure binary
    mask = mask > 0.5

    pixels = mask.flatten()
    # We prepend and append 0 to detect changes at the start and end of the array
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The 'runs' array contains indices where the value changes.
    # Even indices (0, 2, ...) are starts of 1s (since we padded with 0).
    # Odd indices (1, 3, ...) are ends of 1s (starts of 0s).
    # Length of a run is end - start.
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def fbeta_score(preds, targets, beta=0.5, threshold=0.5, epsilon=1e-7):
    """
    Calculates the F-Beta score.

    Args:
        preds (torch.Tensor): Predictions (probabilities or logits).
        targets (torch.Tensor): Ground truth labels.
        beta (float): Beta value for F-score (default 0.5 weights precision higher).
        threshold (float): Threshold to convert predictions to binary.
        epsilon (float): Small value to prevent division by zero.

    Returns:
        float: The F-Beta score.
    """
    # Ensure inputs are tensors
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)

    # Apply threshold to get binary predictions
    # Assuming preds are probabilities. If they are logits, sigmoid should be applied before this function
    # or the threshold adjusted. Here we assume probabilities [0, 1].
    preds_bin = (preds > threshold).float()
    targets_bin = (targets > threshold).float()

    tp = (preds_bin * targets_bin).sum()
    fp = (preds_bin * (1 - targets_bin)).sum()
    fn = ((1 - preds_bin) * targets_bin).sum()

    beta_sq = beta**2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = numerator / (denominator + epsilon)

    return score.item()


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss.
    """

    def __init__(self, bce_weight=0.5, smooth=1e-6):
        """
        Args:
            bce_weight (float): Weight for the BCE component (0.0 to 1.0).
                                Dice weight will be (1 - bce_weight).
            smooth (float): Smoothing factor for Dice calculation.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid).
            targets (torch.Tensor): Ground truth labels (0 or 1).

        Returns:
            torch.Tensor: Calculated loss.
        """
        # Binary Cross Entropy with Logits
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

        # Dice Loss
        probs = torch.sigmoid(logits)

        # Flatten for dice calculation
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        # Combined Loss
        loss = self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss

        return loss
