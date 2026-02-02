import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from library.config import IN_CHANNELS, NUM_CLASSES


class FixedHighPassFilter(nn.Module):
    """
    A fixed (non-trainable) high-pass filter layer using the KV kernel.
    This layer is used to extract noise residuals from the input image,
    suppressing image content to highlight steganographic signals.
    """

    def __init__(self):
        super(FixedHighPassFilter, self).__init__()
        # Initialize a 1-in, 1-out convolution with 5x5 kernel
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=5, padding=2, bias=False
        )

        # Define the standard KV kernel
        # This kernel is designed to suppress low-frequency content (smooth areas)
        kv_kernel = np.array(
            [
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1],
            ],
            dtype=np.float32,
        )

        # Normalize the kernel (optional, but helps keep scale reasonable)
        # Often divided by 12 in literature, but raw integer values are also common.
        # We use the raw values here; the subsequent BatchNorm in ResNet will handle scaling.

        # Convert to tensor and reshape to (Out_Channels, In_Channels, H, W) -> (1, 1, 5, 5)
        kv_kernel_tensor = torch.from_numpy(kv_kernel).unsqueeze(0).unsqueeze(0)

        # Set weights and disable gradient computation
        self.conv.weight.data = kv_kernel_tensor
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)


class MonoResidualResNet(nn.Module):
    """
    A ResNet-18 based model adapted for Steganalysis on the Luminance channel.

    Architecture:
    1. Fixed High-Pass Filter (KV Kernel)
    2. ResNet-18 Backbone (First Conv modified for 1-channel input)
    3. Binary Classification Head
    """

    def __init__(self, pretrained=True):
        super(MonoResidualResNet, self).__init__()

        # 1. Preprocessing: Fixed Residual Layer
        self.preprocessing = FixedHighPassFilter()

        # 2. Backbone: ResNet-18
        # Load weights if requested
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # 3. Input Adaptation
        # The original ResNet-18 Conv1 expects 3 channels (RGB).
        # We need to modify it to accept 1 channel (Luminance residuals).
        original_conv1 = self.backbone.conv1

        new_conv1 = nn.Conv2d(
            in_channels=IN_CHANNELS,  # Should be 1
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias,
        )

        if pretrained:
            # Initialize the new 1-channel weights by summing the original 3-channel weights.
            # This preserves the magnitude of activations roughly consistent with the pre-trained state.
            # Original shape: (64, 3, 7, 7) -> Sum over dim 1 -> New shape: (64, 1, 7, 7)
            new_conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)

        self.backbone.conv1 = new_conv1

        # 4. Output Adaptation
        # Replace the final fully connected layer for binary classification
        original_fc = self.backbone.fc
        self.backbone.fc = nn.Linear(original_fc.in_features, NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 1, Height, Width).
                              Expected to be the Luminance channel normalized to [0, 1].

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        # Step 1: Extract residuals using the fixed high-pass filter
        x = self.preprocessing(x)

        # Step 2: Pass through the modified ResNet backbone
        x = self.backbone(x)

        return x
