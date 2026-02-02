import torch
import torch.nn as nn
import timm
from library.config import Config


class WIVSNet(nn.Module):
    """
    Weight-Inflated 9-Channel Volumetric Stack (WIVS-Net).

    This model adapts an EfficientNet-B0 backbone to process a 9-channel volumetric slab
    (3 modalities x 3 depths). It employs a specific 'Weight Inflation' initialization
    strategy to preserve ImageNet priors while enabling immediate volumetric feature extraction.
    """

    def __init__(self, pretrained=True):
        super(WIVSNet, self).__init__()

        # 1. Initialize standard EfficientNet-B0 with 3 channels
        # We set num_classes=1 for binary classification (MGMT promoter methylation)
        # drop_rate controls the dropout before the classifier
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=1,
            drop_rate=Config.DROPOUT,
            in_chans=3,  # Start with 3 to get standard ImageNet weights
        )

        # 2. Apply Weight Inflation to the first convolutional layer
        self._inflate_first_layer()

    def _inflate_first_layer(self):
        """
        Replaces the first convolutional layer (3 channels) with a 9-channel layer.
        Initializes weights by distributing RGB weights to corresponding modalities across depths.
        """
        # Access the first layer (conv_stem in timm EfficientNet)
        old_conv = self.backbone.conv_stem

        # Create new Conv2d layer
        # in_channels = 9 (FLAIR, T1wCE, T2w repeated for 3 depths)
        new_conv = nn.Conv2d(
            in_channels=Config.NUM_CHANNELS,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # --- Weight Inflation Logic ---
        with torch.no_grad():
            # Original weights shape: (Out, 3, K, K)
            old_weights = old_conv.weight
            # New weights shape: (Out, 9, K, K)
            new_weights = torch.zeros_like(new_conv.weight)

            # Scaling factor to preserve activation magnitude (averaging over 3 depths)
            scale = 1.0 / 3.0

            # Mapping Strategy:
            # Input Channel Order:
            #   0: FLAIR (D-s), 1: T1wCE (D-s), 2: T2w (D-s)
            #   3: FLAIR (D),   4: T1wCE (D),   5: T2w (D)
            #   6: FLAIR (D+s), 7: T1wCE (D+s), 8: T2w (D+s)

            # Source Weights:
            #   Red (idx 0) -> FLAIR
            #   Green (idx 1) -> T1wCE
            #   Blue (idx 2) -> T2w

            # Distribute Red weights to FLAIR channels (0, 3, 6)
            new_weights[:, 0, :, :] = old_weights[:, 0, :, :] * scale
            new_weights[:, 3, :, :] = old_weights[:, 0, :, :] * scale
            new_weights[:, 6, :, :] = old_weights[:, 0, :, :] * scale

            # Distribute Green weights to T1wCE channels (1, 4, 7)
            new_weights[:, 1, :, :] = old_weights[:, 1, :, :] * scale
            new_weights[:, 4, :, :] = old_weights[:, 1, :, :] * scale
            new_weights[:, 7, :, :] = old_weights[:, 1, :, :] * scale

            # Distribute Blue weights to T2w channels (2, 5, 8)
            new_weights[:, 2, :, :] = old_weights[:, 2, :, :] * scale
            new_weights[:, 5, :, :] = old_weights[:, 2, :, :] * scale
            new_weights[:, 8, :, :] = old_weights[:, 2, :, :] * scale

            # Assign weights
            new_conv.weight.copy_(new_weights)

            # Copy bias if present
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        # Replace the layer in the backbone
        self.backbone.conv_stem = new_conv

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 9, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        return self.backbone(x)
