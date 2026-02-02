import torch
import torch.nn as nn
import timm


class EffNet25D(nn.Module):
    """
    2.5D Stacked EfficientNet.

    Processes 3D volumes by stacking slices into the channel dimension (Early Fusion).
    Cite solution_lesson_node_00018: Early Fusion vs Late Fusion.
    """

    def __init__(self, model_name="efficientnet_b0", pretrained=True):
        super(EffNet25D, self).__init__()

        # Cite solution_lesson_node_00038: Use timm library for robust adaptation
        # in_chans=64: 16 slices * 4 modalities
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=64,
            num_classes=1,
            drop_path_rate=0.2,
            drop_rate=0.2,  # Explicit dropout
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor. Shape (B, 64, H, W).
        """
        return self.backbone(x)
