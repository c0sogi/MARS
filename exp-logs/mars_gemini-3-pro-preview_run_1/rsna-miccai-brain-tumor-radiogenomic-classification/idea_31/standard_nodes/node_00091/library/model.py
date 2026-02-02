import torch
import torch.nn as nn
import timm
from library.config import (
    MODEL_NAME,
    NUM_CLASSES,
    INPUT_DROPOUT_PROB,
    DROPOUT_RATE,
    WEIGHT_INFLATION_RATIOS,
    NUM_CHANNELS,
)


class StructuredInputDropout(nn.Module):
    """
    Applies structured dropout to the 9-channel volumetric input.

    Logic:
    With probability p, it masks out a specific anatomical group of channels.
    - Group Center: Channels 3, 4, 5 (Relative Depth 50%)
    - Group Periphery: Channels 0, 1, 2 (Depth 40%) AND 6, 7, 8 (Depth 60%)

    This prevents the model from relying solely on the center slice or the periphery.
    """

    def __init__(self, p=INPUT_DROPOUT_PROB):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        # x shape: (Batch, Channels, Height, Width)
        B, C, H, W = x.shape

        # Create a mask of ones
        mask = torch.ones_like(x)

        # Iterate through batch to apply dropout independently per sample
        # (Vectorized implementation is possible but loop is clearer for this specific logic)
        for i in range(B):
            if torch.rand(1).item() < self.p:
                # Decide which group to drop
                if torch.rand(1).item() < 0.5:
                    # Drop Center (Channels 3, 4, 5)
                    mask[i, 3:6, :, :] = 0.0
                else:
                    # Drop Periphery (Channels 0, 1, 2 AND 6, 7, 8)
                    mask[i, 0:3, :, :] = 0.0
                    mask[i, 6:9, :, :] = 0.0

        return x * mask


class RNWIVEfficientNet(nn.Module):
    """
    Relative-Norm Weight-Inflated Volumetric (RN-WIV) Network.

    Uses EfficientNet-B0 backbone with a modified first layer to accept 9 channels.
    Weights are initialized via Gaussian Weight Inflation from the pre-trained RGB weights.
    """

    def __init__(self, model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()

        # 1. Structured Input Dropout
        self.input_dropout = StructuredInputDropout(p=INPUT_DROPOUT_PROB)

        # 2. Backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove default classifier
            in_chans=3,  # Load with standard 3 channels first to get pretrained weights
        )

        # 3. Modify First Layer (Gaussian Weight Inflation)
        if hasattr(self.backbone, "conv_stem"):
            old_conv = self.backbone.conv_stem

            # Create new conv layer with 9 input channels
            new_conv = nn.Conv2d(
                in_channels=NUM_CHANNELS,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

            # Weight Inflation Logic
            # w_orig shape: (Out, 3, K, K)
            w_orig = old_conv.weight.data

            # w_new shape: (Out, 9, K, K)
            w_new = torch.zeros_like(new_conv.weight.data)

            # Center Channels (3, 4, 5) -> 50% energy
            # Corresponds to indices 3:6
            w_new[:, 3:6, :, :] = w_orig * WEIGHT_INFLATION_RATIOS["center"]

            # Peripheral Channels (0, 1, 2) -> 25% energy
            # Corresponds to indices 0:3
            w_new[:, 0:3, :, :] = w_orig * WEIGHT_INFLATION_RATIOS["periphery"]

            # Peripheral Channels (6, 7, 8) -> 25% energy
            # Corresponds to indices 6:9
            w_new[:, 6:9, :, :] = w_orig * WEIGHT_INFLATION_RATIOS["periphery"]

            # Assign new weights
            new_conv.weight.data = w_new

            # Copy bias if it exists
            if old_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data

            # Replace the layer in backbone
            self.backbone.conv_stem = new_conv
        else:
            # Fallback or error if architecture changes (e.g. not EfficientNet)
            raise AttributeError(f"Backbone {model_name} does not have 'conv_stem'.")

        # 4. Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE), nn.Linear(self.backbone.num_features, num_classes)
        )

    def forward(self, x):
        # x: (B, 9, H, W)
        x = self.input_dropout(x)
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
