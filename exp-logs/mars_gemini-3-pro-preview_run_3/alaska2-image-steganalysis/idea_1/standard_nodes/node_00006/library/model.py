import torch
import torch.nn as nn
import numpy as np
import timm
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


class MonoResidualEfficientNet(nn.Module):
    """
    An EfficientNet-B0 based model adapted for Steganalysis on the Luminance channel.

    Architecture:
    1. Fixed High-Pass Filter (KV Kernel)
    2. EfficientNet-B0 Backbone (Input adapted for 1-channel)
    3. Binary Classification Head
    """

    def __init__(self, pretrained=True):
        super(MonoResidualEfficientNet, self).__init__()

        # 1. Preprocessing: Fixed Residual Layer
        self.preprocessing = FixedHighPassFilter()

        # 2. Backbone: EfficientNet-B0
        # timm allows specifying in_chans=1, which handles weight aggregation automatically
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            in_chans=IN_CHANNELS,
            num_classes=NUM_CLASSES,
        )

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

        # Step 2: Pass through the EfficientNet backbone
        x = self.backbone(x)

        return x
