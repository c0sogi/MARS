import torch
import torch.nn as nn
import timm
from library.config import BACKBONE, NUM_CLASSES, DROPOUT_RATE, INPUT_CHANNELS, SEED


class StructuredInputDropout(nn.Module):
    """
    Custom Structured Input Dropout layer.
    Randomly zeros out either the center triplet (channels 3-5) or
    the peripheral triplets (channels 0-2, 6-8) with a given probability.
    """

    def __init__(self, p=0.0):
        super(StructuredInputDropout, self).__init__()
        self.p = p

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 9, H, W)
        Returns:
            torch.Tensor: Masked input tensor.
        """
        if not self.training or self.p == 0.0:
            return x

        B = x.shape[0]
        device = x.device

        # 1. Decide which samples to drop: (B, 1, 1, 1)
        # 1.0 means drop, 0.0 means keep
        should_drop = (torch.rand(B, 1, 1, 1, device=device) < self.p).float()

        # 2. Decide which part to drop: (B, 1, 1, 1)
        # 1.0 means drop Center, 0.0 means drop Periphery
        drop_center_decision = (torch.rand(B, 1, 1, 1, device=device) < 0.5).float()

        # 3. Construct Masks
        # Shape: (1, 9, 1, 1)
        mask_drop_center = torch.ones((1, 9, 1, 1), device=device)
        mask_drop_center[:, 3:6, :, :] = 0.0

        mask_drop_periph = torch.ones((1, 9, 1, 1), device=device)
        mask_drop_periph[:, 0:3, :, :] = 0.0
        mask_drop_periph[:, 6:9, :, :] = 0.0

        # 4. Select Mask based on decision
        # If drop_center_decision is 1, use mask_drop_center, else mask_drop_periph
        selected_mask = (
            drop_center_decision * mask_drop_center
            + (1 - drop_center_decision) * mask_drop_periph
        )

        # 5. Apply dropout decision
        # If should_drop is 1, use selected_mask. If 0, use all ones (keep original)
        final_mask = should_drop * selected_mask + (1 - should_drop) * torch.ones_like(
            selected_mask
        )

        return x * final_mask


class RNVSNetwork(nn.Module):
    """
    ROI-Normalized Volumetric Stack (RN-VS) Network.
    Uses EfficientNet-B0 with modified input layer for 9 channels (3 slices x 3 modalities).
    Implements Gaussian Weight Inflation for initialization.
    """

    def __init__(
        self,
        backbone_name=BACKBONE,
        pretrained=True,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        input_dropout_prob=0.0,
    ):
        super(RNVSNetwork, self).__init__()

        # 1. Create Backbone
        # efficientnet_b0 usually has 'conv_stem' as first layer and 'classifier' as last
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout_rate,
        )

        # 2. Structured Input Dropout
        # Default to 0.0 here because the provided Dataset class already handles this augmentation.
        # However, the layer is included as per architectural requirements.
        self.input_dropout = StructuredInputDropout(p=input_dropout_prob)

        # 3. Modify First Layer (Gaussian Weight Inflation)
        self._modify_first_layer()

    def _modify_first_layer(self):
        """
        Replaces the first convolutional layer (3 channels) with a 9-channel version.
        Initializes weights using Gaussian Weight Inflation logic:
        - Center channels (3-5): 50% of original weights
        - Peripheral channels (0-2, 6-8): 25% of original weights
        """
        # Retrieve original layer
        old_conv = self.backbone.conv_stem

        # Create new layer
        new_conv = nn.Conv2d(
            in_channels=INPUT_CHANNELS,  # 9
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Initialize weights
        with torch.no_grad():
            old_weights = old_conv.weight  # Shape: (Out, 3, K, K)
            new_weights = torch.zeros_like(new_conv.weight)  # Shape: (Out, 9, K, K)

            # Center Slices (Channels 3, 4, 5) -> 50% Energy
            new_weights[:, 3:6, :, :] = old_weights * 0.5

            # Peripheral Slices (Channels 0, 1, 2) -> 25% Energy
            new_weights[:, 0:3, :, :] = old_weights * 0.25

            # Peripheral Slices (Channels 6, 7, 8) -> 25% Energy
            new_weights[:, 6:9, :, :] = old_weights * 0.25

            new_conv.weight.copy_(new_weights)

            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        # Replace in backbone
        self.backbone.conv_stem = new_conv

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 9, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        x = self.input_dropout(x)
        x = self.backbone(x)
        return x
