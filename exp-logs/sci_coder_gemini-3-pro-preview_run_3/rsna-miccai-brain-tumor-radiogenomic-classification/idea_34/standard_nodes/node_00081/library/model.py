import torch
import torch.nn as nn
import timm


def replace_bn_with_gn(module, num_groups=32):
    """
    Recursively replaces all BatchNorm2d layers with GroupNorm layers.

    This function traverses the module tree. When it encounters a BatchNorm2d layer,
    it calculates a valid number of groups (closest divisor of channels to the target num_groups)
    and replaces the layer with GroupNorm, transferring weights, biases, and eps.

    Args:
        module (nn.Module): The PyTorch module to modify in-place.
        num_groups (int): The target number of groups for GroupNorm (default: 32).
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features

            # Determine a valid number of groups
            # GroupNorm requires: num_channels % num_groups == 0
            groups = num_groups

            # 1. Try reducing by powers of 2 if target is too large or not a divisor
            while groups > 1 and (num_channels % groups != 0 or groups > num_channels):
                groups //= 2

            # 2. Fallback: If power-of-2 reduction fails (e.g. for odd/prime channel counts),
            # find the largest divisor of num_channels that is <= target num_groups
            if num_channels % groups != 0:
                found = False
                for g in range(min(num_channels, num_groups), 0, -1):
                    if num_channels % g == 0:
                        groups = g
                        found = True
                        break
                if not found:
                    groups = 1  # Fallback to LayerNorm behavior

            # Create the GroupNorm layer
            # We copy 'eps' to maintain numerical stability characteristics
            gn = nn.GroupNorm(
                num_groups=groups, num_channels=num_channels, eps=child.eps
            )

            # Transfer learnable parameters (gamma/weight and beta/bias)
            if child.weight is not None:
                gn.weight.data = child.weight.data.clone()
            if child.bias is not None:
                gn.bias.data = child.bias.data.clone()

            # Replace the BatchNorm layer with the new GroupNorm layer
            setattr(module, name, gn)
        else:
            # Recursively apply to child modules (e.g., blocks, sub-blocks)
            replace_bn_with_gn(child, num_groups)


class GNHRNet(nn.Module):
    """
    Group-Normalized High-Resolution 2.5D Network (GNHR-Net).

    This architecture is designed to handle high-resolution MRI volumes by stacking slices
    into the channel dimension (2.5D). It replaces Batch Normalization with Group Normalization
    to maintain training stability when using small batch sizes, which are necessary due to
    the memory footprint of high-resolution (320x320) inputs.
    """

    def __init__(
        self,
        model_name="efficientnet_b0",
        pretrained=True,
        in_chans=64,
        num_classes=1,
        drop_path_rate=0.2,
    ):
        """
        Args:
            model_name (str): Name of the timm model backbone.
            pretrained (bool): Whether to load pretrained ImageNet weights.
            in_chans (int): Number of input channels (16 slices * 4 modalities = 64).
            num_classes (int): Number of output classes (1 for binary classification).
            drop_path_rate (float): Stochastic depth rate for regularization.
        """
        super(GNHRNet, self).__init__()

        # Initialize the EfficientNet backbone using timm
        # timm handles the adaptation of the first convolutional layer for in_chans=64
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

        # Structural Innovation: Replace BN with GN
        # This decouples normalization statistics from batch size
        replace_bn_with_gn(self.model, num_groups=32)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 64, 320, 320).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        return self.model(x)
