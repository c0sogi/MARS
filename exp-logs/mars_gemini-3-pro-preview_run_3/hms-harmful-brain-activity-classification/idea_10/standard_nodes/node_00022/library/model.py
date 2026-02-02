import torch
import torch.nn as nn
import timm
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.
    Used here to attend to specific resolutions/channels in the multi-resolution input.
    """

    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiResAdapter(nn.Module):
    """
    Adapter module for Stream A.
    1. Applies SE attention to the 57 input channels (19 electrodes * 3 resolutions).
    2. Projects the 57 channels down to 3 channels using a 1x1 convolution.
       This maps the multi-scale sensor data into standard image space for the backbone.
    """

    def __init__(self, in_channels=57, out_channels=3):
        super(MultiResAdapter, self).__init__()
        # SE Block to dynamically weight resolutions/electrodes
        self.se = SEBlock(in_channels=in_channels, reduction=8)
        # 1x1 Conv to project to 3 channels (RGB-like)
        self.project = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x shape: (Batch, 57, H, W)
        x = self.se(x)
        x = self.project(x)
        x = self.bn(x)
        x = self.relu(x)
        # Output shape: (Batch, 3, H, W)
        return x


class MultiResDualStreamNet(nn.Module):
    """
    Multi-Resolution Dual-Stream Network with Gated Context.

    Stream A (Detail):
        - Input: Multi-Res Spectrograms (57 channels).
        - Adapter: SE + 1x1 Conv -> 3 channels.
        - Backbone: EfficientNet-B2.
        - Captures fine-grained morphological details (spikes, waves).

    Stream B (Context):
        - Input: Kaggle Spectrograms (4 channels).
        - Backbone: EfficientNet-B0.
        - Captures long-term trends (10 min context).

    Fusion:
        - Stream B features are projected to match Stream A dimensions.
        - Sigmoid gate applied to Stream B features.
        - Element-wise multiplication: Stream A * Gate(Stream B).
        - Ensures context modulates the interpretation of details.
    """

    def __init__(self, pretrained=True):
        super(MultiResDualStreamNet, self).__init__()

        # --- Stream A: Detail (EEG Multi-Res) ---
        self.adapter_a = MultiResAdapter(
            in_channels=Config.IN_CHANNELS_A, out_channels=3
        )

        # EfficientNet-B2
        # num_classes=0 returns the pooled feature vector
        self.backbone_a = timm.create_model(
            Config.MODEL_BACKBONE_A,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3,  # Adapter projects to 3
        )

        # Get feature dimension for B2 (usually 1408)
        self.dim_a = self.backbone_a.num_features

        # --- Stream B: Context (Spectrograms) ---
        # EfficientNet-B0
        self.backbone_b = timm.create_model(
            Config.MODEL_BACKBONE_B,
            pretrained=pretrained,
            num_classes=0,
            in_chans=Config.IN_CHANNELS_B,  # 4 regions
        )

        # Get feature dimension for B0 (usually 1280)
        self.dim_b = self.backbone_b.num_features

        # --- Fusion & Head ---
        # Project Stream B features to match Stream A dimension for gating
        self.context_project = nn.Linear(self.dim_b, self.dim_a)

        # Final Classifier
        self.classifier = nn.Linear(self.dim_a, Config.NUM_CLASSES)

        # Softmax is applied during loss calculation (KLDivLoss usually takes log-probs)
        # or we output raw logits. The submission format requires probabilities.
        # We will return logits here for numerical stability in loss,
        # and apply Softmax in inference/training loop if needed.
        # However, for KLDivLoss in PyTorch, input is usually log-probs.
        # For the provided metric function `kl_divergence_score`, it expects probabilities.
        # We will return logits and handle conversion outside or return probs.
        # Standard practice: return logits.

    def forward(self, x):
        """
        Args:
            x (tuple): (x_a, x_b)
                x_a: Stream A input (Batch, 57, 128, 500)
                x_b: Stream B input (Batch, 4, 256, 256)
        """
        x_a, x_b = x

        # --- Stream A Forward ---
        # Adapter: (B, 57, H, W) -> (B, 3, H, W)
        x_a = self.adapter_a(x_a)
        # Backbone: (B, 3, H, W) -> (B, dim_a)
        feat_a = self.backbone_a(x_a)

        # --- Stream B Forward ---
        # Backbone: (B, 4, H, W) -> (B, dim_b)
        feat_b = self.backbone_b(x_b)

        # --- Gated Fusion ---
        # Project context to match detail dimension
        # (B, dim_b) -> (B, dim_a)
        context = self.context_project(feat_b)

        # Compute Gate
        gate = torch.sigmoid(context)

        # Modulate Detail with Context
        fused = feat_a * gate

        # --- Classification ---
        logits = self.classifier(fused)

        return logits
