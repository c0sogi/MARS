import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_logger

logger = get_logger("loss_module")


def get_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Calculates class weights based on the inverse frequency of classes in the training data.
    Formula: Weight_c = N_total / N_c

    Args:
        df (pd.DataFrame): The training metadata containing a 'Category' column.

    Returns:
        torch.Tensor: A tensor of shape (NUM_CLASSES,) containing the weight for each class.
                      The tensor is moved to Config.DEVICE.
    """
    # Total number of samples
    n_total = len(df)

    # Get value counts for the classes present in the dataframe
    class_counts = df["Category"].value_counts().to_dict()

    # Initialize counts for all defined classes (0 to 22) to handle potential missing classes
    # We use a small epsilon or 1 to avoid division by zero if a class is completely absent
    # However, since these weights multiply the loss, if a class is absent from targets,
    # its weight doesn't impact the gradient. We set a default of 1 for safety.
    counts_vec = np.ones(Config.NUM_CLASSES, dtype=np.float32)

    for cls_id in range(Config.NUM_CLASSES):
        if cls_id in class_counts:
            counts_vec[cls_id] = class_counts[cls_id]
        else:
            # If a class is not in the training set, we treat it as having 1 sample
            # to avoid div/0, or we could handle it differently.
            # Given the task, we'll assume at least 1 to keep math stable.
            counts_vec[cls_id] = 1.0

    # Calculate weights: N_total / N_c
    # We apply square root dampening to prevent weights from becoming too large for rare classes
    # Cite solution_lesson_node_00020
    weights = np.sqrt(n_total / counts_vec)

    # Convert to tensor
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(Config.DEVICE)

    logger.info(
        f"Class weights calculated. Min: {weights_tensor.min().item():.4f}, Max: {weights_tensor.max().item():.4f}"
    )

    return weights_tensor


class FocalLoss(nn.Module):
    """
    Implements the Focal Loss function for addressing class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where:
        - p_t is the model's estimated probability for the target class.
        - gamma is the focusing parameter.
        - alpha_t is the class weight.
    """

    def __init__(
        self, weight: torch.Tensor = None, gamma: float = 2.0, reduction: str = "mean"
    ):
        """
        Args:
            weight (torch.Tensor, optional): A manual rescaling weight given to each class.
                                             If provided, it must be a Tensor of size C.
            gamma (float): The focusing parameter. Higher values focus more on hard examples.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Logits from the model of shape (N, C).
            targets (torch.Tensor): Ground truth labels of shape (N,).

        Returns:
            torch.Tensor: The calculated loss.
        """
        # Calculate Cross Entropy Loss without reduction to get log(p_t)
        # We pass self.weight=None here because we apply it manually with the focal term
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=None)

        # p_t = exp(-ce_loss)
        pt = torch.exp(-ce_loss)

        # Focal term: (1 - p_t)^gamma
        focal_term = (1 - pt) ** self.gamma

        # Calculate final loss: focal_term * ce_loss
        loss = focal_term * ce_loss

        # Apply class weights if provided
        if self.weight is not None:
            # Gather the weights corresponding to the target classes
            # self.weight is shape (C,), targets is shape (N,)
            # alpha_t shape will be (N,)
            alpha_t = self.weight[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
