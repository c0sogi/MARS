import os
import random
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from sklearn.metrics import f1_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_f1_score(y_true, y_pred, threshold=0.5):
    """
    Calculates the Mean F1-Score (Macro F1) for multi-label classification.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth binary labels.
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities.
        threshold (float): Threshold to convert probabilities to binary predictions.

    Returns:
        float: The macro F1 score.
    """
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    # Apply threshold
    y_pred_binary = (y_pred > threshold).astype(int)

    # Calculate Macro F1
    return f1_score(y_true, y_pred_binary, average="macro")


def rand_bbox(size, lam):
    """
    Generates a random bounding box for CutMix.

    Args:
        size (tuple): Dimensions of the image tensor (N, C, H, W).
        lam (float): Lambda value derived from Beta distribution.

    Returns:
        tuple: (bbx1, bby1, bbx2, bby2) coordinates.
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class MixupCutmix:
    """
    Implements Mixup and Cutmix data augmentation.
    """

    def __init__(self, mixup_alpha=1.0, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob

    def __call__(self, x, target):
        """
        Applies Mixup or Cutmix to the batch.

        Args:
            x (torch.Tensor): Input images (Batch, Channels, Height, Width).
            target (torch.Tensor): Target labels (Batch, Num_Classes).

        Returns:
            tuple: (mixed_x, mixed_target)
        """
        if np.random.rand() > self.prob:
            return x, target

        # Decide between Mixup and Cutmix
        use_cutmix = np.random.rand() < self.switch_prob

        alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        if use_cutmix:
            # CutMix
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
            # Adjust lambda to match exact pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))

            # Mix targets
            target = lam * target + (1 - lam) * target[index]
        else:
            # MixUp
            x = lam * x + (1 - lam) * x[index]
            target = lam * target + (1 - lam) * target[index]

        return x, target


class ModelEMA:
    """
    Model Exponential Moving Average.
    Maintains a moving average of model parameters for better generalization.
    """

    def __init__(self, model, decay=0.9999, device=None):
        self.decay = decay
        # Create a deep copy of the model
        self.ema = deepcopy(model)
        self.ema.eval()

        # Move to device if specified, otherwise use model's device
        if device:
            self.ema.to(device)
        else:
            # Try to infer device from model parameters
            try:
                device = next(model.parameters()).device
                self.ema.to(device)
            except StopIteration:
                pass  # Model might be empty

        # Disable gradients for EMA model
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the EMA model parameters using the current model parameters.
        """
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()

            for k in msd.keys():
                if k in esd:
                    # Update: ema_param = decay * ema_param + (1 - decay) * current_param
                    # We use in-place operations for efficiency
                    esd[k].mul_(self.decay).add_(msd[k], alpha=1 - self.decay)
