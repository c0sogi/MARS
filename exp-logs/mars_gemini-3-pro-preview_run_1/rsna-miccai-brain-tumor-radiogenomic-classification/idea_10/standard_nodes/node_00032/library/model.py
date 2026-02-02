import torch
import torch.nn as nn
import timm
from library.config import MODEL_BACKBONE, PRETRAINED, DROP_RATE, NUM_CLASSES
from library.utils import set_seed

# Ensure reproducibility upon module import
set_seed()


class EarlyFusionNet(nn.Module):
    """
    Early Fusion Network.

    Uses a single EfficientNet-B0 backbone to process a 3-channel input
    composed of FLAIR, T1wCE, and T2w modalities stacked channel-wise.
    This reduces parameter count and overfitting compared to multi-stream architectures
    (Cite solution_lesson_node_00031).
    """

    def __init__(
        self,
        backbone_name=MODEL_BACKBONE,
        pretrained=PRETRAINED,
        drop_rate=DROP_RATE,
        num_classes=NUM_CLASSES,
    ):
        super(EarlyFusionNet, self).__init__()

        # Single backbone taking 3 input channels (FLAIR, T1wCE, T2w)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=3,
            drop_rate=drop_rate,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Batch of stacked images (B, 3, H, W).
        """
        return self.backbone(x)
