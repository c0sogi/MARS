import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import TARGET_COLS


def seed_everything(seed: int):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WeightedMultiLabelLoss(nn.Module):
    """
    Weighted Multi-Label Logarithmic Loss.

    This loss function computes the binary cross-entropy for each label and applies
    specific weights to each class. It is designed for the RSNA Cervical Spine
    Fracture Detection task where the 'patient_overall' label may be weighted
    differently than specific vertebrae labels.

    The loss is calculated as:
    L_ij = -w_j * [y_ij * log(p_ij) + (1 - y_ij) * log(1 - p_ij)]

    The final loss is the mean over all rows (samples * labels).
    """

    def __init__(self, weights=None, pos_weight=None):
        """
        Args:
            weights (list or torch.Tensor, optional): A list or tensor of weights
                for each class. Must match the number of target columns (8).
                If None, defaults to equal weights (1.0).
            pos_weight (float or torch.Tensor, optional): Weight for the positive class
                to handle imbalance.
        """
        super().__init__()
        num_classes = len(TARGET_COLS)

        if weights is None:
            # Default to 1.0 for all classes if not specified
            w = torch.ones(num_classes, dtype=torch.float32)
        else:
            w = torch.tensor(weights, dtype=torch.float32)

        if w.shape[0] != num_classes:
            raise ValueError(f"Expected {num_classes} weights, got {w.shape[0]}.")

        # Register weights as a buffer so they move to the correct device with the model
        self.register_buffer("weights", w)

        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor(pos_weight)
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        """
        Computes the weighted binary cross-entropy loss.

        Args:
            logits (torch.Tensor): Predicted logits of shape (Batch, Num_Classes).
            targets (torch.Tensor): Ground truth labels of shape (Batch, Num_Classes).

        Returns:
            torch.Tensor: The scalar mean loss.
        """
        # Ensure targets are float for BCE
        targets = targets.float()

        # Compute binary cross-entropy loss per element (no reduction)
        # logits are used for numerical stability
        if self.pos_weight is not None:
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none", pos_weight=self.pos_weight
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )

        # Apply class-specific weights
        # self.weights shape (Num_Classes,) broadcasts to (Batch, Num_Classes)
        weighted_loss = bce_loss * self.weights

        # Return the mean loss across all elements (rows in the submission context)
        return weighted_loss.mean()
