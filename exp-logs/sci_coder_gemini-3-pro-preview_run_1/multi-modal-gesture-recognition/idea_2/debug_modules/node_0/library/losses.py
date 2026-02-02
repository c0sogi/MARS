import torch
import torch.nn as nn
from library.config import Config


class HierarchicalLoss(nn.Module):
    """
    Implements the Hierarchical Loss for the Multi-Stage Refinement Network.
    It computes the sum of Cross-Entropy losses for both the Generation Stage (Stage 1)
    and the Refinement Stage (Stage 2).

    Features:
    - Class Weighting: Applies a specific weight to the background class to balance recall/precision.
    - Label Smoothing: Regularizes the model by smoothing target labels.
    - Ignore Index: Handles padded sequences by ignoring a specific target value (default -100).
    """

    def __init__(self, ignore_index=-100):
        """
        Args:
            ignore_index (int): The target value to ignore during loss computation
                                (usually used for padding). Default: -100.
        """
        super(HierarchicalLoss, self).__init__()

        # 1. Define Class Weights
        # Initialize weights to 1.0 for all classes
        weights = torch.ones(Config.NUM_CLASSES)

        # Apply specific weight to the Background class (Index 0)
        # This helps in balancing precision and recall for the null class
        if Config.NUM_CLASSES > 0:
            weights[0] = Config.BACKGROUND_WEIGHT

        # 2. Initialize CrossEntropyLoss
        # We pass the weights and label smoothing configuration here.
        # The loss module handles moving weights to the appropriate device
        # when .to(device) is called on this module.
        self.criterion = nn.CrossEntropyLoss(
            weight=weights,
            ignore_index=ignore_index,
            label_smoothing=Config.LABEL_SMOOTHING,
            reduction="mean",
        )

    def forward(self, stage1_logits, stage2_logits, targets):
        """
        Computes the hierarchical loss.

        Args:
            stage1_logits (torch.Tensor): Logits from the Generation Stage.
                                          Shape: (Batch, Time, NumClasses)
            stage2_logits (torch.Tensor): Logits from the Refinement Stage.
                                          Shape: (Batch, Time, NumClasses)
            targets (torch.Tensor): Ground truth labels.
                                    Shape: (Batch, Time)

        Returns:
            torch.Tensor: The scalar loss value (sum of stage 1 and stage 2 losses).
        """
        # Reshape inputs for CrossEntropyLoss
        # The criterion expects (N, C) for logits and (N) for targets,
        # where N is the total number of time steps across the batch (Batch * Time).

        # Flatten Stage 1 Logits: (B, T, C) -> (B*T, C)
        s1_flat = stage1_logits.view(-1, Config.NUM_CLASSES)

        # Flatten Stage 2 Logits: (B, T, C) -> (B*T, C)
        s2_flat = stage2_logits.view(-1, Config.NUM_CLASSES)

        # Flatten Targets: (B, T) -> (B*T)
        targets_flat = targets.view(-1)

        # Compute losses for each stage
        loss_stage1 = self.criterion(s1_flat, targets_flat)
        loss_stage2 = self.criterion(s2_flat, targets_flat)

        # Sum the losses
        total_loss = loss_stage1 + loss_stage2

        return total_loss
