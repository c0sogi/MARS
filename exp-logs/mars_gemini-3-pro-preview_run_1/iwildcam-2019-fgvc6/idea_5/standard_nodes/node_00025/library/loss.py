import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from library.config import Config


def get_dampened_class_weights(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """
    Calculates class weights using Square-Root Inverse Frequency.
    Formula: W_c = sqrt(N_total / N_c)
    This dampens the aggressive penalty of standard inverse frequency for long-tail datasets.

    Args:
        df (pd.DataFrame): DataFrame containing a 'Category' column.
        device (torch.device): The device to store the weights on.

    Returns:
        torch.Tensor: Normalized class weights of shape [NUM_CLASSES].
    """
    # Get value counts for the classes present in the dataframe
    class_counts = df["Category"].value_counts().sort_index()

    # Initialize counts array for all defined classes with 1 to avoid division by zero
    # (though all classes should ideally be present in the full training set)
    counts = np.ones(Config.NUM_CLASSES, dtype=np.float32)

    for cat, count in class_counts.items():
        if 0 <= cat < Config.NUM_CLASSES:
            counts[cat] = count

    total_samples = np.sum(counts)

    # Calculate weights: Square Root of Inverse Frequency
    weights = np.sqrt(total_samples / counts)

    # Normalize weights so that the mean weight is 1.0
    # This ensures the scale of the loss doesn't drift significantly from standard CrossEntropy
    weights = weights / np.mean(weights)

    return torch.FloatTensor(weights).to(device)


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss implementation.
    Loss(x, class) = -alpha[class] * (1 - p[class])^gamma * log(p[class])
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        """
        Args:
            alpha (torch.Tensor, optional): Pre-computed class weights.
            gamma (float): Focusing parameter to down-weight easy examples.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predictions [Batch, Num_Classes] (Logits)
            targets (torch.Tensor): Ground truth labels [Batch]
        """
        # Calculate standard Cross Entropy Loss (log(pt))
        # reduction='none' allows us to apply the modulating factor per sample
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Get the probability of the true class (pt)
        pt = torch.exp(-ce_loss)

        # Calculate the modulating factor (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma

        # Combine terms
        loss = focal_term * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            # Ensure alpha is on the correct device
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)

            # Gather weights corresponding to the targets
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CompositeLoss(nn.Module):
    """
    Composite Loss Function for Multi-Task Learning.
    Combines Species Classification (Focal Loss) and Animal Detection (BCE Loss).

    L_total = L_species + lambda * L_detection
    """

    def __init__(self, class_weights=None, lambda_detection=Config.LAMBDA_DETECTION):
        """
        Args:
            class_weights (torch.Tensor, optional): Weights for the species Focal Loss.
            lambda_detection (float): Weighting factor for the auxiliary detection loss.
        """
        super(CompositeLoss, self).__init__()
        self.species_loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
        self.detection_loss_fn = nn.BCEWithLogitsLoss()
        self.lambda_detection = lambda_detection

    def forward(self, outputs, targets):
        """
        Calculates the weighted sum of losses.

        Args:
            outputs (dict): Model outputs containing 'species_logits' and 'detection_logits'.
            targets (dict): Batch targets containing 'species_label' and 'detection_label'.

        Returns:
            torch.Tensor: The scalar total loss.
        """
        # Unpack model outputs
        species_logits = outputs["species_logits"]
        detection_logits = outputs["detection_logits"]

        # Unpack targets
        species_labels = targets["species_label"]
        detection_labels = targets["detection_label"]

        # Ensure detection labels are [Batch, 1] for BCEWithLogitsLoss
        if detection_labels.dim() == 1:
            detection_labels = detection_labels.unsqueeze(1)

        # 1. Species Loss (Focal Loss)
        l_species = self.species_loss_fn(species_logits, species_labels)

        # 2. Detection Loss (Binary Cross Entropy)
        # This auxiliary task helps the backbone learn generic animal features
        l_detection = self.detection_loss_fn(detection_logits, detection_labels)

        # Total Loss
        total_loss = l_species + (self.lambda_detection * l_detection)

        return total_loss
