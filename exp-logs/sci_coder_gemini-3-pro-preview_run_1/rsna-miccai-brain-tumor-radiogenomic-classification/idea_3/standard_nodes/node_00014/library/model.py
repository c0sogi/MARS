import torch
import torch.nn as nn
import timm
import math
from library.config import Config


class BraTS2DCNN(nn.Module):
    """
    Simple 2.5D CNN (Cite solution_lesson_node_00008).
    Takes a 3-channel input (FLAIR, T1wCE, T2w) and predicts the target.
    """

    def __init__(self):
        super(BraTS2DCNN, self).__init__()
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED_BACKBONE,
            num_classes=1,
            in_chans=Config.NUM_CHANNELS,
        )

    def forward(self, x):
        # x shape: (Batch, 3, H, W)
        return self.backbone(x)
