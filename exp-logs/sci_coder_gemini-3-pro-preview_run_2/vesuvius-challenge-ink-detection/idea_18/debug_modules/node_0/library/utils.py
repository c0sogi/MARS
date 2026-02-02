import numpy as np
import torch
import os


def sigmoid(x):
    """
    Applies the sigmoid function to the input array.

    Args:
        x (np.ndarray or float): Input data.

    Returns:
        np.ndarray or float: Sigmoid transformed data.
    """
    return 1 / (1 + np.exp(-x))


def rle_encoding(mask):
    """
    Converts a binary mask to Run-Length Encoding (RLE) format.

    Args:
        mask (np.ndarray): Binary mask (0s and 1s).

    Returns:
        str: Space-delimited RLE string.
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def dice_loss(pred, target, smooth=1e-5):
    """
    Computes the Dice Loss.

    Args:
        pred (torch.Tensor): Predicted probabilities.
        target (torch.Tensor): Ground truth binary masks.
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        torch.Tensor: Dice loss value.
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return 1.0 - dice


def fbeta_score(pred, target, beta=0.5, threshold=0.5, smooth=1e-5):
    """
    Computes the F-beta score.

    Args:
        pred (torch.Tensor): Predicted probabilities.
        target (torch.Tensor): Ground truth binary masks.
        beta (float): Weight of precision in harmonic mean.
        threshold (float): Threshold to binarize predictions.
        smooth (float): Smoothing factor.

    Returns:
        torch.Tensor: F-beta score.
    """
    pred_bin = (pred > threshold).float()

    pred_bin = pred_bin.contiguous().view(-1)
    target = target.contiguous().view(-1)

    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()

    beta_sq = beta**2
    numerator = (1 + beta_sq) * tp + smooth
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp + smooth

    return numerator / denominator


class EarlyStopping:
    """
    Early stops the training if validation score doesn't improve after a given patience.
    Maximizes the score (e.g., F0.5).
    """

    def __init__(
        self,
        patience=5,
        verbose=False,
        delta=0,
        path="checkpoint.pth",
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation score improved.
            verbose (bool): If True, prints a message for each validation score improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_max = -np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when validation score increases."""
        if self.verbose:
            self.trace_func(
                f"Validation score improved ({self.val_score_max:.6f} --> {score:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_score_max = score
