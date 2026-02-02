import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from library.config import seed_everything


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.

    The metric checks that the pairs are sorted, positive, and the decoded pixel
    values are not duplicated. The pixels are numbered from left to right,
    then top to bottom: 1 is pixel (1,1), 2 is pixel (1,2), etc.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W) where 1 indicates ink.

    Returns:
        str: Space-delimited string of RLE pairs (start, length).
    """
    # Ensure mask is flattened (row-major order: left-to-right, top-to-bottom)
    pixels = mask.flatten()

    # We prepend and append 0 to detect runs starting at index 0 or ending at the last index
    # This simplifies the logic for finding transitions
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The runs array currently contains [start1, end1, start2, end2, ...]
    # The competition format requires [start1, length1, start2, length2, ...]
    # Length = end - start
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def fbeta_score(preds, targets, beta=0.5, threshold=0.5, smooth=1e-6):
    """
    Calculates the F-Beta score for binary segmentation.

    The F0.5 score weights precision higher than recall, which improves the ability
    to form coherent characters out of detected ink areas.

    Args:
        preds (torch.Tensor or np.ndarray): Predicted probabilities or binary mask.
        targets (torch.Tensor or np.ndarray): Ground truth binary mask.
        beta (float): Beta value for F-score (default 0.5 weights precision higher).
        threshold (float): Threshold to binarize predictions.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: The F-Beta score.
    """
    # Convert tensors to numpy arrays if necessary
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()

    # Binarize predictions based on the threshold
    preds_bin = (preds > threshold).astype(float)
    targets_bin = targets.astype(float)

    # Calculate True Positives (tp), False Positives (fp), and False Negatives (fn)
    tp = (preds_bin * targets_bin).sum()
    fp = (preds_bin * (1 - targets_bin)).sum()
    fn = ((1 - preds_bin) * targets_bin).sum()

    # Calculate Precision and Recall
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)

    # Calculate F-Beta Score
    # Formula: (1 + beta^2) * p * r / (beta^2 * p + r)
    beta_sq = beta**2
    fbeta = (
        (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall + smooth)
    )

    return fbeta


class ModelEMA:
    """
    Maintains an Exponential Moving Average (EMA) of the model weights.

    EMA averages the parameters of the model over the course of training, which
    often leads to better generalization and more stable results than using
    the weights from the final iteration directly.
    """

    def __init__(self, model, decay=0.99):
        """
        Args:
            model (nn.Module): The model to track.
            decay (float): The decay factor for EMA (default 0.99).
        """
        self.decay = decay
        self.ema_model = deepcopy(model)
        self.ema_model.eval()

        # Disable gradients for the EMA model to save memory and computation
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the EMA model weights based on the current model weights.

        Args:
            model (nn.Module): The current training model.
        """
        with torch.no_grad():
            # Update parameters: shadow = decay * shadow + (1 - decay) * new_param
            for ema_param, param in zip(
                self.ema_model.parameters(), model.parameters()
            ):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

            # Update buffers (e.g., BatchNorm running mean/var)
            # We strictly copy buffers to keep statistics synced with the latest training state
            for ema_buffer, buffer in zip(self.ema_model.buffers(), model.buffers()):
                ema_buffer.copy_(buffer)

    def get_model(self):
        """
        Returns the EMA model.
        """
        return self.ema_model
