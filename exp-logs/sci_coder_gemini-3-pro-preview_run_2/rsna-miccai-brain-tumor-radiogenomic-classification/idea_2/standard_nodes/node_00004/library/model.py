import torch
import torch.nn as nn
import timm
from library.config import Config


class ModalityGroupedEfficientNet(nn.Module):
    """
    A 2.5D EfficientNet-B0 model modified for multi-modal MRI input.

    It accepts a 12-channel input tensor (4 modalities * 3 slices) and uses
    a grouped convolution in the first layer to process each modality
    independently before feature fusion.
    """

    def __init__(self):
        super().__init__()

        # 1. Load the pretrained backbone
        # We use num_classes=Config.NUM_CLASSES (1) to automatically generate
        # the correct linear classification head.
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=Config.NUM_CLASSES,
            drop_rate=Config.DROPOUT_RATE,
        )

        # 2. Modify the Input Stem (First Convolutional Layer)
        # Original Stem: Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        original_stem = self.backbone.conv_stem

        # Configuration for the new stem
        new_in_channels = Config.TOTAL_CHANNELS  # 12
        out_channels = original_stem.out_channels
        kernel_size = original_stem.kernel_size
        stride = original_stem.stride
        padding = original_stem.padding
        groups = Config.FIRST_CONV_GROUPS  # 4 (One group per modality)

        # Create the new layer with grouped convolutions
        new_stem = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False if original_stem.bias is None else True,
        )

        # 3. Weight Initialization via Transfer Learning
        # The shape of the weights for a Conv2d layer is (Out, In/Groups, k, k).
        # Original (RGB): (32, 3/1, 3, 3) -> (32, 3, 3, 3)
        # New (Grouped):  (32, 12/4, 3, 3) -> (32, 3, 3, 3)
        # Since the shapes align perfectly, we can copy the pretrained ImageNet weights.
        # This initializes each modality's feature extractor with proven edge/texture detectors.

        with torch.no_grad():
            if new_stem.weight.shape == original_stem.weight.shape:
                new_stem.weight.copy_(original_stem.weight)
                if original_stem.bias is not None and new_stem.bias is not None:
                    new_stem.bias.copy_(original_stem.bias)
            else:
                # Fallback initialization if shapes somehow don't match (unlikely for B0)
                nn.init.kaiming_normal_(
                    new_stem.weight, mode="fan_out", nonlinearity="relu"
                )

        # Replace the stem in the backbone
        self.backbone.conv_stem = new_stem

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 12, H, W).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        return self.backbone(x)
