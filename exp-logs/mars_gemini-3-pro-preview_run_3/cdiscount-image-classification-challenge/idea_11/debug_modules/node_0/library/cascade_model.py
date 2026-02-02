import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    INPUT_DIM,
    HIDDEN_DIM,
    DROPOUT_RATE,
    NUM_CLASSES_L1,
    NUM_CLASSES_L2,
    NUM_CLASSES_L3,
)


class ConditionalCascadeMLP(nn.Module):
    """
    A conditional hierarchical neural network that predicts product categories
    in a cascading fashion (Level 1 -> Level 2 -> Level 3).

    Each subsequent stage takes the original features concatenated with the
    logits from the previous stage as input, allowing coarse-grained predictions
    to guide fine-grained decisions.
    """

    def __init__(
        self,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
        num_classes_l1=NUM_CLASSES_L1,
        num_classes_l2=NUM_CLASSES_L2,
        num_classes_l3=NUM_CLASSES_L3,
    ):
        super(ConditionalCascadeMLP, self).__init__()

        # Stage 1: Predict Level 1 (Coarse)
        # Input: Product Embeddings (ResNet + EffNet)
        self.stage1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes_l1),
        )

        # Stage 2: Predict Level 2 (Intermediate)
        # Input: Product Embeddings + Level 1 Logits
        input_dim_s2 = input_dim + num_classes_l1
        self.stage2 = nn.Sequential(
            nn.Linear(input_dim_s2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes_l2),
        )

        # Stage 3: Predict Level 3 (Fine-grained Target)
        # Input: Product Embeddings + Level 2 Logits
        input_dim_s3 = input_dim + num_classes_l2
        self.stage3 = nn.Sequential(
            nn.Linear(input_dim_s3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes_l3),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Input_Dim).

        Returns:
            tuple: (logits_l1, logits_l2, logits_l3)
        """
        # Stage 1 Forward
        logits_l1 = self.stage1(x)

        # Stage 2 Forward
        # Concatenate original features with L1 logits to condition L2 prediction
        x_s2 = torch.cat([x, logits_l1], dim=1)
        logits_l2 = self.stage2(x_s2)

        # Stage 3 Forward
        # Concatenate original features with L2 logits to condition L3 prediction
        x_s3 = torch.cat([x, logits_l2], dim=1)
        logits_l3 = self.stage3(x_s3)

        return logits_l1, logits_l2, logits_l3


class HierarchicalLoss(nn.Module):
    """
    Computes the sum of Cross Entropy losses for all three hierarchical levels.
    Handles MixUp augmentation by interpolating losses between two sets of targets.
    """

    def __init__(self):
        super(HierarchicalLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, targets_a, targets_b, lam):
        """
        Args:
            outputs (tuple): (logits_l1, logits_l2, logits_l3) from the model.
            targets_a (tuple): (l1, l2, l3) labels for the first mixup component.
            targets_b (tuple): (l1, l2, l3) labels for the second mixup component.
            lam (float): MixUp interpolation coefficient (lambda).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        logits_l1, logits_l2, logits_l3 = outputs
        l1_a, l2_a, l3_a = targets_a
        l1_b, l2_b, l3_b = targets_b

        # Calculate total loss for target set A
        loss_l1_a = self.ce(logits_l1, l1_a)
        loss_l2_a = self.ce(logits_l2, l2_a)
        loss_l3_a = self.ce(logits_l3, l3_a)
        loss_a = loss_l1_a + loss_l2_a + loss_l3_a

        # Calculate total loss for target set B
        loss_l1_b = self.ce(logits_l1, l1_b)
        loss_l2_b = self.ce(logits_l2, l2_b)
        loss_l3_b = self.ce(logits_l3, l3_b)
        loss_b = loss_l1_b + loss_l2_b + loss_l3_b

        # Interpolate
        return lam * loss_a + (1 - lam) * loss_b
