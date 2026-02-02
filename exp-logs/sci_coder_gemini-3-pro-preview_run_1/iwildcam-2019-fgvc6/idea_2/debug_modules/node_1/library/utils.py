import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from library.config import seed_everything, NUM_CLASSES


def get_class_weights(df):
    """
    Calculates class weights based on the inverse frequency of classes.
    Weight = Total Samples / (Number of Classes * Class Count)

    Args:
        df (pd.DataFrame): DataFrame containing the 'Category' column.

    Returns:
        torch.Tensor: Tensor of shape (NUM_CLASSES,) containing the weights.
    """
    # Count samples per category
    class_counts = df["Category"].value_counts().sort_index()

    n_samples = len(df)
    n_classes = NUM_CLASSES

    # Initialize weights with 1.0 (default for missing classes to avoid errors)
    weights = np.ones(n_classes, dtype=np.float32)

    for cat_id, count in class_counts.items():
        if count > 0:
            weights[cat_id] = n_samples / (n_classes * count)

    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """
    Implements the Focal Loss for addressing class imbalance.
    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        """
        Args:
            alpha (torch.Tensor, optional): Pre-computed class weights.
            gamma (float): Focusing parameter.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits of shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels of shape (batch_size).

        Returns:
            torch.Tensor: The computed loss.
        """
        # Compute cross entropy loss (log(pt))
        # reduction='none' is required to apply alpha and focal term element-wise
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Compute pt
        pt = torch.exp(-ce_loss)

        # Compute focal term
        focal_term = (1 - pt) ** self.gamma

        # Combine
        loss = focal_term * ce_loss

        # Apply class weights (alpha)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)

            # Gather alpha for the specific targets
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def calculate_macro_f1(y_true, y_pred):
    """
    Calculates the Macro F1 score.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        float: The Macro F1 score.
    """
    return f1_score(y_true, y_pred, average="macro")
