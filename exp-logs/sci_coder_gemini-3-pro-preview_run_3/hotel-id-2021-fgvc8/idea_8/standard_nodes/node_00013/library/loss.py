import torch
import torch.nn as nn


class ArcFaceLoss(nn.Module):
    """
    Implements the ArcFace Loss wrapper.

    In this architecture, the SubCenterArcFaceHead (defined in model.py) is responsible
    for the metric learning specifics:
    1. Computing cosine similarities between embeddings and sub-centers.
    2. Applying the angular margin penalty (m) to the ground truth class.
    3. Scaling the logits by the scale factor (s).

    Thus, this Loss module receives the fully processed logits and computes
    the standard Cross Entropy Loss.
    """

    def __init__(self, label_smoothing=0.0):
        """
        Args:
            label_smoothing (float): Amount of label smoothing to apply (0.0 to 1.0).
        """
        super(ArcFaceLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, labels):
        """
        Computes the loss.

        Args:
            logits (torch.Tensor): The scaled and margin-penalized logits from the model.
                                   Shape: (batch_size, num_classes).
            labels (torch.Tensor): The ground truth class indices.
                                   Shape: (batch_size,).

        Returns:
            torch.Tensor: The scalar loss value.
        """
        return self.criterion(logits, labels)
