import torch
import torch.nn as nn


class AdaFaceLoss(nn.Module):
    """
    Implements the loss component for the AdaFace training pipeline.

    In this architecture, the 'AdaFaceHead' (located in library/model.py) handles
    the calculation of the adaptive margin and the scaling of cosine similarities.
    It outputs modified logits: s * cos(theta + m_adaptive).

    This class wraps standard CrossEntropyLoss to optimize these modified logits.
    Minimizing CrossEntropy on these logits effectively maximizes the margin
    between the true class and others, weighted by the image quality (norm).
    """

    def __init__(self):
        """
        Initializes the AdaFaceLoss.
        """
        super(AdaFaceLoss, self).__init__()
        # Standard CrossEntropyLoss combines LogSoftmax and NLLLoss.
        # It expects raw logits (which our model provides, albeit scaled/shifted).
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        """
        Computes the loss.

        Args:
            logits (torch.Tensor): The margin-adjusted and scaled logits output
                                   by the model's AdaFaceHead.
                                   Shape: (batch_size, num_classes)
            targets (torch.Tensor): The ground truth class indices.
                                    Shape: (batch_size,)

        Returns:
            torch.Tensor: The computed scalar loss.
        """
        loss = self.criterion(logits, targets)
        return loss
