import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for Sequence Data.
    Operates on (Batch, Channels, Sequence_Length).
    Pools across Sequence_Length to recalibrate Channels based on global patient context.
    """

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (Batch, Channel, Sequence_Length)
        b, c, _ = x.size()
        # Squeeze: Global Average Pooling over sequence
        y = self.avg_pool(x).view(b, c)
        # Excitation: MLP
        y = self.fc(y).view(b, c, 1)
        # Scale
        return x * y.expand_as(x)


class ContextBlock(nn.Module):
    """
    Context Layer combining Local Smoothing (1D Conv) and Global Recalibration (SE).
    """

    def __init__(self, in_channels, reduction=16):
        super(ContextBlock, self).__init__()
        # Local Smoothing: 1D Convolution (k=3, p=1)
        # Smooths artifacts across adjacent slices
        self.conv = nn.Conv1d(
            in_channels, in_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        # Global Recalibration: SE Block
        self.se = SEBlock(in_channels, reduction)

    def forward(self, x):
        # x shape: (Batch, Channel, Sequence_Length)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)
        return x


class CervicalFractureNet(nn.Module):
    """
    Channel-Attentive 2.5D MIL Network.

    Architecture:
    1. Backbone: ResNet18 (2.5D Input: z-1, z, z+1) features per slice.
    2. Context: 1D Conv (Smoothing) + SE Block (Global Recalibration).
    3. Classification: Instance-level (per slice) prediction for C1-C7 + Patient.
    4. Aggregation: Global Max Pooling across sequence.
    """

    def __init__(self):
        super(CervicalFractureNet, self).__init__()

        # 1. Backbone
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=True)

        # We use the first conv layer as is (accepts 3 channels).
        # Remove the final pooling and fc layer to get feature maps.
        # Output of layer4 is (Batch, 512, H/32, W/32)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Spatial pooling to get (Batch, 512) per slice
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Feature dimension from ResNet18
        feature_dim = 512

        # 2. Context Layer
        self.context = ContextBlock(feature_dim)

        # 3. Classification Head
        # Predicts 8 classes (C1-C7 + Patient) for each slice
        self.head = nn.Linear(feature_dim, Config.NUM_CLASSES)

        # Initialization
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (Batch, Sequence_Length, 3, H, W)
        Returns:
            pooled_logits: (Batch, 8) - Logits for C1-C7 + Patient aggregated over the scan.
        """
        b, seq_len, c, h, w = x.size()

        # Merge Batch and Sequence dimensions for efficient backbone processing
        # Treat every slice as an independent sample initially
        x = x.view(b * seq_len, c, h, w)

        # Backbone Forward
        features = self.backbone(x)  # (B*S, 512, h', w')
        features = self.spatial_pool(features)  # (B*S, 512, 1, 1)
        features = features.view(b, seq_len, -1)  # (B, S, 512)

        # Permute for Context Block (requires B, C, S)
        features = features.permute(0, 2, 1)  # (B, 512, S)

        # Context Block (Smoothing + SE)
        features = self.context(features)

        # Permute back to (B, S, 512) for classification
        features = features.permute(0, 2, 1)

        # Instance-Level Classification
        # Generate logits for every slice
        logits = self.head(features)  # (B, S, 8)

        # Aggregation: Global Max Pooling across sequence
        # We take the maximum logit across all slices for each class.
        pooled_logits, _ = torch.max(logits, dim=1)  # (B, 8)

        return pooled_logits
