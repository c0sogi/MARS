import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.config import Config


class ClassBalancedFocalLoss(nn.Module):
    """
    Implementation of Class-Balanced Loss combined with Focal Loss.

    This loss function addresses two issues:
    1. Class Imbalance: Using the 'Effective Number of Samples' weighting.
    2. Easy/Hard Sample Imbalance: Using Focal Loss to focus on hard examples.

    It supports both standard targets (indices) and soft targets (Mixup/CutMix).

    Reference: https://arxiv.org/abs/1901.05555
    """

    def __init__(self, beta=Config.CLASS_BETA, gamma=Config.FOCAL_GAMMA):
        """
        Args:
            beta (float): Hyperparameter for Class Balanced Loss.
                          0 < beta < 1. Closer to 1 means more reweighting.
            gamma (float): Focusing parameter for Focal Loss.
        """
        super(ClassBalancedFocalLoss, self).__init__()

        self.beta = beta
        self.gamma = gamma

        # 1. Load Metadata to calculate class counts
        if not os.path.exists(Config.TRAIN_META_PATH):
            raise FileNotFoundError(
                f"Metadata file not found at {Config.TRAIN_META_PATH}"
            )

        df = pd.read_csv(Config.TRAIN_META_PATH)

        # Get counts for all classes 0 to NUM_CLASSES-1
        # reindex ensures we have an entry for every class ID even if count is 0
        class_counts = (
            df["Category"]
            .value_counts()
            .reindex(range(Config.NUM_CLASSES), fill_value=0)
            .sort_index()
            .values
        )

        # 2. Calculate Effective Number of Samples
        # Formula: En = (1 - beta^n) / (1 - beta)
        # We handle the case where count is 0 by treating it as 1 to avoid division by zero errors,
        # though in this dataset all classes should have samples.
        class_counts = np.maximum(class_counts, 1)

        effective_num = 1.0 - np.power(self.beta, class_counts)
        per_class_weights = (1.0 - self.beta) / effective_num

        # 3. Normalize weights
        # Normalizing so that the sum equals the number of classes keeps the loss scale consistent
        # roughly around the scale of standard CrossEntropy
        per_class_weights = (
            per_class_weights / np.sum(per_class_weights) * Config.NUM_CLASSES
        )

        # Register as buffer so it moves to device automatically with the module
        # but is not updated by the optimizer.
        self.register_buffer("weight", torch.tensor(per_class_weights).float())

    def forward(self, logits, targets):
        """
        Compute the Class Balanced Focal Loss.

        Args:
            logits: [Batch, Num_Classes] - Raw output from model (before Softmax)
            targets: [Batch] (indices) or [Batch, Num_Classes] (soft labels/mixup)

        Returns:
            torch.Tensor: Scalar loss value
        """
        # Compute probabilities
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        # Handle target format
        if targets.dim() == 1:
            # Hard labels: convert to one-hot
            targets_one_hot = F.one_hot(targets, num_classes=Config.NUM_CLASSES).float()
        else:
            # Soft labels (Mixup/CutMix)
            targets_one_hot = targets.float()

        # Focal Loss Term: (1 - p_t)^gamma
        # Since we are doing multi-class with potential soft labels, we apply the focal modulation
        # to the cross entropy term for each class.
        # Standard CE = - sum(y * log(p))
        # Focal CE    = - sum(y * (1-p)^gamma * log(p))

        focal_modulation = (1.0 - probs).pow(self.gamma)

        # Class Balanced Weight
        # self.weight is shape [C]. We broadcast to [B, C]
        cb_weight = self.weight.view(1, -1)

        # Combine terms
        # loss = - y * (1-p)^gamma * log(p) * w_class
        # This computes the loss contribution for every class for every sample
        loss = -targets_one_hot * focal_modulation * log_probs * cb_weight

        # Sum over classes to get per-sample loss, then mean over batch
        return loss.sum(dim=1).mean()
