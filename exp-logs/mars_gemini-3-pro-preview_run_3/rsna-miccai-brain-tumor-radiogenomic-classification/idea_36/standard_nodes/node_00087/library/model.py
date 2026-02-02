import torch
import torch.nn as nn
import timm


class Stacked25DNet(nn.Module):
    """
    2.5D Stacked Network.

    Processes MRI volumes by stacking slices and modalities into the input channel dimension.
    Uses Early Fusion (Cite solution_lesson_node_00018).
    """

    def __init__(self, model_name="efficientnet_b0", pretrained=True):
        super(Stacked25DNet, self).__init__()

        # Cite solution_lesson_node_00038: Use timm for robust adaptation.
        # in_chans=64: 16 slices * 4 modalities.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=64,
            num_classes=1,
            drop_path_rate=0.2,
            drop_rate=0.2,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor. Shape (B, 64, H, W).
        Returns:
            torch.Tensor: Logits. Shape (B, 1).
        """
        return self.backbone(x)
