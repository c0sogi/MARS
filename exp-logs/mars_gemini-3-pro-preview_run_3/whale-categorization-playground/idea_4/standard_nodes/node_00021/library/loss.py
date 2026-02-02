import torch
import torch.nn as nn


class ArcFaceLoss(nn.Module):
    """
    Implements the loss component for the ArcFace training pipeline.
    Wraps CrossEntropyLoss for the scaled logits from ArcFaceHead.
    """

    def __init__(self):
        super(ArcFaceLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        loss = self.criterion(logits, targets)
        return loss
