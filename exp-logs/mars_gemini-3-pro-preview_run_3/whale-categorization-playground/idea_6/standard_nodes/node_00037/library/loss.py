import torch
import torch.nn as nn
from library.config import Config


class WhaleLoss(nn.Module):
    """
    Loss function for the Whale Identification task.

    This module wraps the standard CrossEntropyLoss. In the context of ArcFace,
    the model's head is responsible for manipulating the logits (applying the
    angular margin and scaling). This loss function then computes the log-softmax
    and negative log-likelihood on those modified logits to maximize inter-class
    separability and minimize intra-class variance.
    """

    def __init__(self):
        """
        Initializes the WhaleLoss module.
        """
        super(WhaleLoss, self).__init__()
        # CrossEntropyLoss combines LogSoftmax and NLLLoss in one single class.
        # It expects raw logits (which are provided by the ArcFace head).
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        """
        Computes the loss.

        Args:
            logits (torch.Tensor): The scaled and margin-modified logits from the model.
                                   Shape: (batch_size, num_classes)
            targets (torch.Tensor): The ground truth class indices.
                                    Shape: (batch_size)

        Returns:
            torch.Tensor: The computed scalar loss.
        """
        return self.criterion(logits, targets)


def get_loss():
    """
    Factory function to instantiate and return the loss function.

    Returns:
        WhaleLoss: An instance of the loss module.
    """
    return WhaleLoss()
