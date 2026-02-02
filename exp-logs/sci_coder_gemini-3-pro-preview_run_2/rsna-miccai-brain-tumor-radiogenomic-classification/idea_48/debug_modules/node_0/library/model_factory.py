import torch
import torch.nn as nn
from torchvision import models
from library.config import INPUT_CHANNELS, GROUPS, SEED

# Ensure reproducibility
torch.manual_seed(SEED)


class StemSEBlock(nn.Module):
    """
    Squeeze-and-Excitation block designed for the Stem.
    Allows early cross-modality interaction and feature recalibration
    before spatial downsampling in the MBConv blocks.
    """

    def __init__(self, channel, reduction=4):
        super(StemSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class AsymmetricEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet with Dense-Volumetric Stem Attention.

    Features:
    1. Dense-Volumetric Input: Accepts 20 channels (4 modalities x 5 slices).
    2. Grouped Stem: Isolates modalities using groups=4.
    3. Asymmetric Initialization: Adapts RGB weights to 5-slice depth using central replication.
    4. Stem SE Block: Early global context calibration.
    """

    def __init__(self):
        super(AsymmetricEfficientNet, self).__init__()

        # Load backbone with default (ImageNet) weights
        self.base_model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        # ---------------------------------------------------------------------
        # 1. Modify Stem Convolution
        # ---------------------------------------------------------------------
        # Original Stem: Conv2dNormActivation(
        #   (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        #   (1): BatchNorm2d(...)
        #   (2): SiLU(...)
        # )

        old_conv = self.base_model.features[0][0]

        # Create new Conv2d with 20 input channels and groups=4
        # We preserve out_channels (32), kernel, stride, padding, and bias settings.
        new_conv = nn.Conv2d(
            in_channels=INPUT_CHANNELS,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            groups=GROUPS,
            bias=(old_conv.bias is not None),
        )

        # ---------------------------------------------------------------------
        # 2. Asymmetric Central-Replication Initialization
        # ---------------------------------------------------------------------
        # Old weights shape: (32, 3, 3, 3) -> (Out, In, K, K)
        # New weights shape: (32, 5, 3, 3) -> (Out, In/Groups, K, K)
        # Note: In PyTorch, weight shape for grouped conv is (Out, In/Groups, K, K)

        w_old = old_conv.weight.data
        w_new = torch.zeros_like(new_conv.weight.data)  # Shape: (32, 5, 3, 3)

        # Mapping Strategy:
        # Index 0 (Edge)    <- Old Index 0 (R)
        # Index 1 (Texture) <- Old Index 1 (G)
        # Index 2 (Texture) <- Old Index 1 (G)
        # Index 3 (Texture) <- Old Index 1 (G)
        # Index 4 (Edge)    <- Old Index 2 (B)

        w_new[:, 0, :, :] = w_old[:, 0, :, :]
        w_new[:, 1, :, :] = w_old[:, 1, :, :]
        w_new[:, 2, :, :] = w_old[:, 1, :, :]
        w_new[:, 3, :, :] = w_old[:, 1, :, :]
        w_new[:, 4, :, :] = w_old[:, 2, :, :]

        new_conv.weight.data = w_new

        # Replace the convolution in the stem sequence
        self.base_model.features[0][0] = new_conv

        # ---------------------------------------------------------------------
        # 3. Insert Stem SE Block
        # ---------------------------------------------------------------------
        # We insert the SE block after the Stem (features[0]) and before the first MBConv (features[1])
        layers = list(self.base_model.features.children())
        # layers[0] is the Stem. Insert SE at index 1.
        layers.insert(1, StemSEBlock(channel=32, reduction=4))

        # Reconstruct features
        self.base_model.features = nn.Sequential(*layers)

        # ---------------------------------------------------------------------
        # 4. Modify Classifier
        # ---------------------------------------------------------------------
        # EfficientNet B0 classifier: Dropout -> Linear(1280, 1000)
        # We replace the Linear layer for binary classification

        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x):
        # EfficientNet forward pass
        x = self.base_model.features(x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.classifier(x)
        return x
