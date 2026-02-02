import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=Config.MIXUP_ALPHA, device=Config.DEVICE):
    """
    Applies Mixup augmentation to the input batch.
    Returns:
        mixed_x: The mixed input tensor.
        y_a: Targets for the first image.
        y_b: Targets for the second image.
        lam: The mixing coefficient (lambda).
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the Mixup loss as a weighted average of the criterion
    applied to both sets of targets.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_score(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (AUC).
    """
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        # Handle edge cases where only one class is present in the batch
        return 0.5


def apply_tta(model, inputs, device=Config.DEVICE):
    """
    Applies Test Time Augmentation (TTA) by averaging predictions across
    4 views: Original, Horizontal Flip, Vertical Flip, and Horizontal+Vertical Flip.

    Args:
        model: The trained PyTorch model.
        inputs: Input tensor of shape (B, C, H, W).
        device: Computation device.

    Returns:
        avg_probs: Tensor of averaged probabilities.
    """
    model.eval()
    inputs = inputs.to(device)
    probs_list = []

    # 1. Original
    with torch.no_grad():
        logits = model(inputs)
        probs_list.append(torch.sigmoid(logits))

    # 2. Horizontal Flip (Time Reversal)
    # Flips the last dimension (Width/Time)
    with torch.no_grad():
        inputs_h = torch.flip(inputs, dims=[-1])
        logits_h = model(inputs_h)
        probs_list.append(torch.sigmoid(logits_h))

    # 3. Vertical Flip (Frequency Inversion)
    # Flips the second to last dimension (Height/Frequency)
    with torch.no_grad():
        inputs_v = torch.flip(inputs, dims=[-2])
        logits_v = model(inputs_v)
        probs_list.append(torch.sigmoid(logits_v))

    # 4. Horizontal + Vertical Flip
    with torch.no_grad():
        inputs_hv = torch.flip(inputs, dims=[-1, -2])
        logits_hv = model(inputs_hv)
        probs_list.append(torch.sigmoid(logits_hv))

    # Stack and average probabilities
    # Shape becomes (4, B, ...) -> Mean over dim 0 -> (B, ...)
    probs_stack = torch.stack(probs_list)
    avg_probs = torch.mean(probs_stack, dim=0)

    return avg_probs
