import torch
import torch.nn as nn
from library.config import Config


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Implements Cross Entropy Loss with Label Smoothing.

    This loss function applies a smoothing factor to the target labels, preventing
    the model from predicting training examples with 100% confidence. This is
    critical for the Whale Identification task where many classes are singletons
    (only one example), helping to reduce overfitting to background artifacts
    and improving generalization.
    """

    def __init__(self, smoothing=Config.LABEL_SMOOTHING):
        """
        Initialize the LabelSmoothingCrossEntropy loss.

        Args:
            smoothing (float): The smoothing factor epsilon.
                               Defaults to Config.LABEL_SMOOTHING (0.1).
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        # Use PyTorch's built-in CrossEntropyLoss with label_smoothing
        # This is available and optimized in the provided PyTorch environment.
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.smoothing)

    def forward(self, preds, targets):
        """
        Compute the loss during the forward pass.

        Args:
            preds (torch.Tensor): Model logits of shape (Batch Size, Num Classes).
                                  For ArcFace, these are the scaled cosine similarities.
            targets (torch.Tensor): Ground truth class indices of shape (Batch Size,).

        Returns:
            torch.Tensor: The computed scalar loss.
        """
        return self.criterion(preds, targets)
