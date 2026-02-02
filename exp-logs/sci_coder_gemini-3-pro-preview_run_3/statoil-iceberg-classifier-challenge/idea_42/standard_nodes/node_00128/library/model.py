import torch
import torch.nn as nn
import torch.nn.functional as F
from library.layers import MaxBlurPool2d, SEModule
from library.config import Config


class AAHACNN(nn.Module):
    """
    Anti-Aliased Hybrid-Attentive Plain CNN (AAHA-CNN).

    A 4-stage CNN backbone optimized for shift-invariance and speckle noise reduction.
    Features:
    - Plain CNN structure (no residuals) to enforce aggressive filtering.
    - Anti-Aliased Downsampling (MaxBlurPool) instead of standard MaxPool.
    - Hybrid SE Modules for channel attention.
    - Selective Hierarchical Max Pooling (Stage 3 & 4) for readout.
    - Raw incidence angle fusion.
    """

    def __init__(self):
        super(AAHACNN, self).__init__()

        # Hyperparameters
        self.in_channels = Config.IN_CHANNELS
        self.blur_kernel = Config.BLUR_KERNEL_SIZE
        self.dropout_rate = Config.DROPOUT_RATE

        # --- Backbone ---

        # Block 1: 3 -> 64
        # We retain bias=True to preserve initialization dynamics (Lesson 76)
        self.block1_conv = nn.Conv2d(
            self.in_channels, 64, kernel_size=3, padding=1, bias=True
        )
        self.block1_bn = nn.BatchNorm2d(64)
        self.block1_act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.block1_se = SEModule(64)
        self.block1_pool = MaxBlurPool2d(64, kernel_size=self.blur_kernel)

        # Block 2: 64 -> 128
        self.block2_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.block2_bn = nn.BatchNorm2d(128)
        self.block2_act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.block2_se = SEModule(128)
        self.block2_pool = MaxBlurPool2d(128, kernel_size=self.blur_kernel)

        # Block 3: 128 -> 128
        self.block3_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.block3_bn = nn.BatchNorm2d(128)
        self.block3_act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.block3_se = SEModule(128)
        self.block3_pool = MaxBlurPool2d(128, kernel_size=self.blur_kernel)

        # Block 4: 128 -> 128
        self.block4_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.block4_bn = nn.BatchNorm2d(128)
        self.block4_act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.block4_se = SEModule(128)
        self.block4_pool = MaxBlurPool2d(128, kernel_size=self.blur_kernel)

        # --- Classification Head ---

        # Input Dimension Calculation:
        # Stage 3 Output (Global Max Pool): 128 features
        # Stage 4 Output (Global Max Pool): 128 features
        # Incidence Angle: 1 feature
        # Total: 128 + 128 + 1 = 257

        self.head_fc1 = nn.Linear(257, 256)
        self.head_act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.head_drop = nn.Dropout(p=self.dropout_rate)
        self.head_fc2 = nn.Linear(256, 1)

        # Initialization:
        # We rely on PyTorch default initialization (Kaiming Uniform for Conv/Linear),
        # as specified in the solution design.

    def forward(self, x, angle):
        """
        Args:
            x (torch.Tensor): Image input of shape (B, 3, 75, 75)
            angle (torch.Tensor): Incidence angle of shape (B,)
        """

        # --- Stage 1 ---
        x = self.block1_conv(x)
        x = self.block1_bn(x)
        x = self.block1_act(x)
        x = self.block1_se(x)
        x = self.block1_pool(x)

        # --- Stage 2 ---
        x = self.block2_conv(x)
        x = self.block2_bn(x)
        x = self.block2_act(x)
        x = self.block2_se(x)
        x = self.block2_pool(x)

        # --- Stage 3 ---
        x = self.block3_conv(x)
        x = self.block3_bn(x)
        x = self.block3_act(x)
        x = self.block3_se(x)
        x3_out = self.block3_pool(x)

        # --- Stage 4 ---
        x = self.block4_conv(x3_out)
        x = self.block4_bn(x)
        x = self.block4_act(x)
        x = self.block4_se(x)
        x4_out = self.block4_pool(x)

        # --- Readout (Selective Hierarchical Max Pooling) ---
        # We use Max Pooling to capture sparse high-intensity peaks (icebergs)

        # Pool Stage 3 Output: (B, 128, H3, W3) -> (B, 128)
        p3 = F.adaptive_max_pool2d(x3_out, output_size=1).view(x3_out.size(0), -1)

        # Pool Stage 4 Output: (B, 128, H4, W4) -> (B, 128)
        p4 = F.adaptive_max_pool2d(x4_out, output_size=1).view(x4_out.size(0), -1)

        # --- Feature Fusion ---
        # Reshape angle to (B, 1)
        angle = angle.view(-1, 1)

        # Concatenate: [Stage3_Pool, Stage4_Pool, Angle]
        features = torch.cat([p3, p4, angle], dim=1)

        # --- Classification ---
        out = self.head_fc1(features)
        out = self.head_act(out)
        out = self.head_drop(out)
        out = self.head_fc2(out)

        return out
