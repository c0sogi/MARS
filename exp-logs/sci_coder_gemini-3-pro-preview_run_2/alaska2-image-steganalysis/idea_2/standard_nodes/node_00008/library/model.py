import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


def get_srm_kernels():
    """
    Generates the 30 fixed 5x5 kernels for the SRM (Spatial Rich Models) layer.
    Includes 1st, 2nd, 3rd order residuals and various edge/square detectors.
    """
    kernels = []

    # --- Definitions of base 5x5 kernels ---
    # 1st order (spam11) - [-1, 1]
    k_1st = np.zeros((5, 5))
    k_1st[2, 2] = 1
    k_1st[2, 3] = -1

    # 2nd order (spam12) - [-1, 2, -1]
    k_2nd = np.zeros((5, 5))
    k_2nd[2, 1] = -1
    k_2nd[2, 2] = 2
    k_2nd[2, 3] = -1

    # 3rd order (spam14) - [1, -3, 3, -1]
    k_3rd = np.zeros((5, 5))
    k_3rd[2, 1] = 1
    k_3rd[2, 2] = -3
    k_3rd[2, 3] = 3
    k_3rd[2, 4] = -1

    # 3x3 Square
    k_sq3 = np.zeros((5, 5))
    k_sq3[1:4, 1:4] = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]])

    # 5x5 Square
    k_sq5 = np.zeros((5, 5))
    k_sq5[2, 2] = -1
    k_sq5[:, :] -= 1 / 24.0
    k_sq5[2, 2] = -1 + 25 / 24.0  # Normalize to sum 0 roughly or specific pattern
    # Better definition for 5x5 edge/square:
    k_edge5 = np.zeros((5, 5))
    k_edge5[2, 2] = 4
    k_edge5[1:4, 1:4] -= 1
    k_edge5[2, 2] = 4  # Reset center
    # Actually, let's use a standard list of rotations for the basic 1st/2nd/3rd types

    def rotate_and_add(base_k):
        # Add base and its 3 rotations (90, 180, 270)
        rots = []
        rots.append(base_k)
        rots.append(np.rot90(base_k, 1))
        rots.append(np.rot90(base_k, 2))
        rots.append(np.rot90(base_k, 3))
        # Also diagonals for some?
        # To get exactly 30, we usually mix types.
        # For this implementation, we will generate 30 diverse high-pass filters.
        return rots

    # 1st order rotations (4)
    kernels.extend(rotate_and_add(k_1st))

    # 2nd order rotations (4)
    kernels.extend(rotate_and_add(k_2nd))

    # 3rd order rotations (4)
    kernels.extend(rotate_and_add(k_3rd))

    # 3x3 Square (1 - symmetric)
    kernels.append(k_sq3)

    # 5x5 Edge (1 - symmetric)
    kernels.append(k_edge5)

    # Diagonals for 1st order (4)
    k_1st_diag = np.zeros((5, 5))
    k_1st_diag[2, 2] = 1
    k_1st_diag[3, 3] = -1
    kernels.extend(rotate_and_add(k_1st_diag))

    # Diagonals for 2nd order (4)
    k_2nd_diag = np.zeros((5, 5))
    k_2nd_diag[1, 1] = -1
    k_2nd_diag[2, 2] = 2
    k_2nd_diag[3, 3] = -1
    kernels.extend(rotate_and_add(k_2nd_diag))

    # We have 4+4+4+1+1+4+4 = 22. Need 8 more.

    # 3rd order diagonal (4)
    k_3rd_diag = np.zeros((5, 5))
    k_3rd_diag[1, 1] = 1
    k_3rd_diag[2, 2] = -3
    k_3rd_diag[3, 3] = 3
    k_3rd_diag[4, 4] = -1
    kernels.extend(rotate_and_add(k_3rd_diag))

    # 26 kernels now. Need 4 more.

    # 4th order (spam14 edge case?) or simple point filters
    # Let's add a 3x3 Cross
    k_cross = np.zeros((5, 5))
    k_cross[2, 1] = -1
    k_cross[2, 3] = -1
    k_cross[1, 2] = -1
    k_cross[3, 2] = -1
    k_cross[2, 2] = 4
    kernels.append(k_cross)

    # And 3 more variations of 5x5 high pass
    k_hp1 = (
        np.array(
            [
                [-1, -1, -1, -1, -1],
                [-1, 2, 2, 2, -1],
                [-1, 2, 8, 2, -1],
                [-1, 2, 2, 2, -1],
                [-1, -1, -1, -1, -1],
            ]
        )
        / 12.0
    )  # Normalize roughly
    k_hp1 -= k_hp1.mean()
    kernels.append(k_hp1)

    k_hp2 = np.copy(k_hp1)
    k_hp2[2, 2] = -k_hp2[2, 2]  # Invert center
    kernels.append(k_hp2)

    # Just fill the last one with a simple delta to preserve some raw info (high freq)
    k_delta = np.zeros((5, 5))
    k_delta[2, 2] = 1
    k_delta[2, 3] = -1  # Another 1st order var
    kernels.append(k_delta)

    # Ensure we have exactly 30
    kernels = kernels[:30]

    # Stack to (30, 1, 5, 5)
    kernels = np.stack(kernels, axis=0)[:, None, :, :]
    return torch.tensor(kernels, dtype=torch.float32)


class SRMConv(nn.Module):
    """
    SRM (Spatial Rich Models) Convolution Layer.
    Applies 30 fixed high-pass filters to extract noise residuals.
    Handles RGB inputs by applying filters to the luminance component.
    """

    def __init__(self, in_channels=3):
        super(SRMConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 30

        # Initialize Conv Layer
        # We use groups=1 (standard conv) but construct weights to mix channels
        self.srm = nn.Conv2d(
            in_channels,
            self.out_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
        )

        # Get 30x1x5x5 kernels
        srm_kernels = get_srm_kernels()  # Shape (30, 1, 5, 5)

        # Prepare RGB weights: (30, 3, 5, 5)
        # We broadcast the 1-channel kernel across 3 channels using luminance weights
        # Y = 0.299R + 0.587G + 0.114B
        rgb_weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(
            1, 3, 1, 1
        )

        # w[i, c, h, w] = kernel[i, 0, h, w] * rgb_weights[0, c, 0, 0]
        weights = srm_kernels * rgb_weights

        # Load weights and freeze
        self.srm.weight.data = weights
        self.srm.weight.requires_grad = False

    def forward(self, x):
        return self.srm(x)


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM).
    f(X) = (1/N * sum(x^p))^(1/p)
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Apply pooling over H, W dimensions
        # x shape: (B, C, H, W)
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class StegoNet(nn.Module):
    """
    Steganography Detection Network.
    Architecture:
    1. SRMConv Stem (Fixed High-Pass Filters)
    2. EfficientNetV2-Small Backbone
    3. GeM Pooling Head
    4. Linear Classifier
    """

    def __init__(
        self,
        backbone_name=Config.backbone_name,
        pretrained=True,
        num_classes=Config.num_classes,
    ):
        super(StegoNet, self).__init__()

        # 1. Stem
        self.srm = SRMConv(in_channels=3)

        # 2. Backbone
        # Load backbone with pretrained weights
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)

        # Modify the first layer of the backbone to accept 30 channels from SRM
        # EfficientNetV2 usually names the first layer 'conv_stem'
        if hasattr(self.backbone, "conv_stem"):
            old_stem = self.backbone.conv_stem
            in_channels = self.srm.out_channels  # 30
            out_channels = old_stem.out_channels
            kernel_size = old_stem.kernel_size
            stride = old_stem.stride
            padding = old_stem.padding

            # Create new stem
            new_stem = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )

            # Initialize weights?
            # Since domain is different (residuals vs pixels), random init is acceptable/standard.
            # We preserve the scale of initialization.
            nn.init.kaiming_normal_(
                new_stem.weight, mode="fan_out", nonlinearity="relu"
            )

            self.backbone.conv_stem = new_stem
        else:
            # Fallback for other backbones if needed, though Config specifies EffNetV2
            raise AttributeError(
                f"Backbone {backbone_name} does not have 'conv_stem'. Check layer names."
            )

        # 3. Head
        # Remove original classifier
        if hasattr(self.backbone, "classifier"):
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, "head"):
            # Some timm models use 'head'
            if hasattr(self.backbone.head, "fc"):
                num_features = self.backbone.head.fc.in_features
            else:
                num_features = self.backbone.num_features
            self.backbone.head = nn.Identity()
        else:
            num_features = self.backbone.num_features

        # Define Custom Head
        self.gem = GeM(p=3.0)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # x: (B, 3, H, W)

        # 1. Extract Residuals
        x = self.srm(x)  # (B, 30, H, W)

        # 2. Backbone Features
        # forward_features returns the feature map before pooling/classifier
        x = self.backbone.forward_features(x)  # (B, C, H', W')

        # 3. Pooling
        x = self.gem(x)  # (B, C, 1, 1)

        # 4. Flatten and Classify
        x = x.view(x.size(0), -1)  # (B, C)
        x = self.fc(x)  # (B, 1)

        return x
