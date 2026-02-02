import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RSNALoss(nn.Module):
    """
    Custom loss function for the RSNA Cervical Spine Fracture Detection task.
    Implements the Implicitly Weighted Multi-Task Loss strategy.

    The loss is calculated as the sum of the binary cross-entropy for the
    'patient_overall' label and the mean binary cross-entropy of the 7
    vertebrae labels (C1-C7). This structure implicitly weights the patient
    outcome more heavily than individual vertebrae.
    """

    def __init__(self):
        super(RSNALoss, self).__init__()
        # Use BCEWithLogitsLoss as the model outputs raw logits.
        # reduction='none' allows us to separate and weight specific columns.
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        """
        Calculates the weighted multi-label logarithmic loss.

        Args:
            logits (torch.Tensor): Predicted logits of shape (Batch, 8).
                                   Expected order: [C1, C2, C3, C4, C5, C6, C7, Patient_Overall]
            targets (torch.Tensor): Ground truth labels of shape (Batch, 8).
                                    Expected order: [C1, C2, C3, C4, C5, C6, C7, Patient_Overall]

        Returns:
            torch.Tensor: The scalar loss value averaged over the batch.
        """
        # Compute element-wise BCE loss
        loss = self.bce(logits, targets)

        # Separate vertebrae losses (indices 0-6) and patient loss (index 7)
        vertebrae_loss = loss[:, :7]
        patient_loss = loss[:, 7]

        # Calculate the mean loss across the 7 vertebrae for each sample
        # This reduces the contribution of individual vertebrae relative to the patient label
        mean_vertebrae_loss = vertebrae_loss.mean(dim=1)

        # Combine: Patient Loss + Average Vertebrae Loss
        # This effectively creates a 1:7 weighting ratio for individual vertebrae vs patient
        total_loss = mean_vertebrae_loss + patient_loss

        # Return the mean loss over the batch
        return total_loss.mean()
