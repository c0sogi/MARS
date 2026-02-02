import torch
import torch.nn as nn
from library.config import Config


class WhaleLoss(nn.Module):
    """
    Implements the loss function for the Whale Species Prediction task.

    This module works in conjunction with the WhaleDenseNet model defined in library/model.py.
    The model's ArcMarginProduct head applies the additive angular margin penalty and scaling
    to the logits during the forward pass when labels are provided. Therefore, this module
    implements Cross Entropy Loss to optimize those logits.

    Attributes:
        criterion (nn.CrossEntropyLoss): The underlying loss function with label smoothing.
    """

    def __init__(self, label_smoothing=Config.LABEL_SMOOTHING):
        """
        Initializes the WhaleLoss.

        Args:
            label_smoothing (float): The amount of label smoothing to apply.
                                     Defaults to Config.LABEL_SMOOTHING (0.1).
        """
        super(WhaleLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, targets):
        """
        Computes the loss between the model output and the ground truth labels.

        Args:
            logits (torch.Tensor): The output from the model's forward pass.
                                   Shape: (Batch Size, Num Classes).
                                   In the context of ArcFace, these are the scaled cosine
                                   similarities with the margin penalty applied to the target class.
            targets (torch.Tensor): The ground truth class indices.
                                    Shape: (Batch Size).

        Returns:
            torch.Tensor: The scalar loss value.
        """
        return self.criterion(logits, targets)
