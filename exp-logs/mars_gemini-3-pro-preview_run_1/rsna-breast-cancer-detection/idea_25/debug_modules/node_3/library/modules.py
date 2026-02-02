import torch
import torch.nn as nn
import torchvision.ops as ops
from library import config


class DeformableAlignmentBlock(nn.Module):
    """
    Aligns contralateral features to target features using Deformable Convolution.

    This module predicts dense offsets based on the concatenation of target and
    contralateral features, then applies Deformable Convolution to the
    contralateral features to spatially align them with the target.
    """

    def __init__(self, in_channels, kernel_size=3):
        super(DeformableAlignmentBlock, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Offset Predictor
        # Input: Concatenated Target + Contralateral (2 * in_channels)
        # Output: 2 * kernel_size * kernel_size (x, y offsets for each kernel point)
        # We use a 3x3 convolution to capture local context for offset prediction.
        self.offset_conv = nn.Conv2d(
            in_channels * 2,
            2 * kernel_size * kernel_size,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # Deformable Convolution Layer
        # Acts as the alignment operator.
        # We use a standard DeformConv2d which includes learnable weights.
        self.dcn = ops.DeformConv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=False,
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize offset convolution weights and bias to zero.
        # This ensures that at the start of training, the deformation is zero,
        # effectively behaving like a standard convolution. This stability is
        # critical for convergence.
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, target_features, contra_features):
        """
        Args:
            target_features: Tensor of shape (B, C, H, W)
            contra_features: Tensor of shape (B, C, H, W)
        Returns:
            aligned_contra: Tensor of shape (B, C, H, W)
        """
        # 1. Concatenate features along channel dimension
        combined = torch.cat([target_features, contra_features], dim=1)

        # 2. Predict offsets
        offsets = self.offset_conv(combined)

        # 3. Apply Deformable Convolution
        # Warps 'contra_features' using the calculated offsets to match 'target_features'
        aligned_contra = self.dcn(contra_features, offsets)

        return aligned_contra


class AsymmetryGatingBlock(nn.Module):
    """
    Implements the Asymmetry Gating Mechanism.

    Calculates the difference between the Target features and the Aligned Contralateral
    features. This difference map is processed to create a spatial attention mask,
    which is then multiplied element-wise with the Target features. This suppresses
    symmetric background information (including demographic bias) and highlights
    asymmetric regions (potential lesions).
    """

    def __init__(self, in_channels):
        super(AsymmetryGatingBlock, self).__init__()

        # Lightweight convolutional block to generate attention mask from difference
        # Structure: Conv(3x3) -> BN -> ReLU -> Conv(1x1) -> Sigmoid
        # Using a bottleneck (C -> C/4 -> 1) keeps it lightweight.
        self.attention_net = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, target_features, aligned_contra_features):
        """
        Args:
            target_features: Tensor of shape (B, C, H, W)
            aligned_contra_features: Tensor of shape (B, C, H, W)
        Returns:
            gated_features: Tensor of shape (B, C, H, W)
        """
        # 1. Compute Absolute Difference
        # D = |F_target - F'_contra|
        diff = torch.abs(target_features - aligned_contra_features)

        # 2. Generate Attention Mask
        # M = Sigmoid(Conv(D))
        # Shape: (B, 1, H, W) - Spatial Attention
        mask = self.attention_net(diff)

        # 3. Gated Fusion
        # F_out = F_target * M
        # Broadcasting the 1-channel mask across the C channels of target_features.
        # This suppresses features where the mask is close to 0 (symmetric regions).
        gated_features = target_features * mask

        return gated_features
