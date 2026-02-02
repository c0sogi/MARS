import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for Channel-Wise Attention.

    This block adaptively recalibrates channel-wise feature responses by explicitly
    modelling interdependencies between channels. In the context of EEG, this allows
    the model to dynamically weight the importance of specific electrodes (channels)
    based on the global signal content.
    """

    def __init__(self, in_channels, reduction=4):
        """
        Args:
            in_channels (int): Number of input channels (e.g., 19 EEG electrodes).
            reduction (int): Reduction ratio for the bottleneck in the SE block.
                             Lower reduction allows capturing more complex dependencies
                             but increases parameter count.
        """
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
        # Squeeze: Global Average Pooling -> (B, C, 1, 1)
        y = self.avg_pool(x).view(b, c)
        # Excitation: FC layers -> (B, C)
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: Reweight input channels
        return x * y


class AttentiveDualScaleNetwork(nn.Module):
    """
    Attentive Dual-Scale Fusion Network with Gated Context.

    Architecture:
    1. Stream A (EEG):
       - Input: (B, 19, 128, 256) Mel-Spectrograms from 50s EEG.
       - SE-Adapter: Reweights 19 electrode channels.
       - Projection: 1x1 Conv reduces 19 channels to 3.
       - Backbone: EfficientNet-B2 (Pretrained).

    2. Stream B (Context):
       - Input: (B, 4, 256, 256) Spectrograms from 10m window.
       - Backbone: EfficientNet-B0 (Pretrained).

    3. Fusion Head:
       - Gated Context: The context vector (Stream B) generates a gate
         that modulates the morphology vector (Stream A).
       - Output: 6-class probability logits.
    """

    def __init__(self):
        super(AttentiveDualScaleNetwork, self).__init__()

        # ==========================
        # Stream A: Fine-Grained EEG
        # ==========================
        # Innovation 1: Channel-Wise Attention Adapter
        self.se_block = SEBlock(in_channels=Config.IN_CHANNELS_A, reduction=4)

        # Projection layer to map 19 EEG channels to 3 channels for ImageNet backbone
        self.projector = nn.Conv2d(Config.IN_CHANNELS_A, 3, kernel_size=1, bias=False)

        # Backbone A: EfficientNet-B2
        # num_classes=0 removes the classification head and returns pooled features
        self.backbone_a = timm.create_model(
            Config.BACKBONE_A, pretrained=Config.PRETRAINED, num_classes=0, in_chans=3
        )

        # ==========================
        # Stream B: Long-Term Context
        # ==========================
        # Backbone B: EfficientNet-B0
        # in_chans=4 adapts the first conv layer to accept 4 regions (LL, RL, LP, RP)
        self.backbone_b = timm.create_model(
            Config.BACKBONE_B,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            in_chans=Config.IN_CHANNELS_B,
        )

        # ==========================
        # Fusion & Classification
        # ==========================
        # Feature dimensions
        # EfficientNet-B2 output features: 1408
        # EfficientNet-B0 output features: 1280
        self.dim_a = self.backbone_a.num_features
        self.dim_b = self.backbone_b.num_features

        # Innovation 2: Gated Context Mechanism
        # Maps context features (dim_b) to the same space as morphology features (dim_a)
        # to create a gating mask.
        self.gate_layer = nn.Linear(self.dim_b, self.dim_a)

        # Final Classifier
        self.classifier = nn.Linear(self.dim_a, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x (tuple): (x_eeg, x_spec)
                x_eeg: Tensor of shape (B, 19, 128, 256)
                x_spec: Tensor of shape (B, 4, 256, 256)
        Returns:
            logits: Tensor of shape (B, 6)
        """
        x_eeg, x_spec = x

        # ---------------------------
        # Stream A Processing
        # ---------------------------
        # 1. Apply Attention (Reweight Channels)
        x_eeg = self.se_block(x_eeg)

        # 2. Project 19 -> 3 channels
        x_eeg = self.projector(x_eeg)

        # 3. Backbone Feature Extraction
        # Output shape: (B, 1408)
        feat_a = self.backbone_a(x_eeg)

        # ---------------------------
        # Stream B Processing
        # ---------------------------
        # 1. Backbone Feature Extraction
        # Output shape: (B, 1280)
        feat_b = self.backbone_b(x_spec)

        # ---------------------------
        # Gated Fusion
        # ---------------------------
        # 1. Generate Gate from Context
        # transform feat_b to match feat_a dimension -> Sigmoid
        gate = torch.sigmoid(self.gate_layer(feat_b))

        # 2. Modulate Morphology with Context
        # Element-wise multiplication
        fused = feat_a * gate

        # ---------------------------
        # Classification
        # ---------------------------
        logits = self.classifier(fused)

        return logits
