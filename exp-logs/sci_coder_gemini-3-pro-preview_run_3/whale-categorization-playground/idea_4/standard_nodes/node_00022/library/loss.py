import torch
import torch.nn as nn


class AdaFaceLoss(nn.Module):
    """
    Implements the loss component for the AdaFace training pipeline.
    Wraps CrossEntropyLoss for the scaled logits from AdaFaceHead.
    """

    def __init__(self):
        super(AdaFaceLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        loss = self.criterion(logits, targets)
        return loss
