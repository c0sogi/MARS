import torch
import torch.nn as nn
import torchvision.models as models
from library.config import ModelConfig
from library.layers import CoordinateAttention, SpecFPN, AttentionPooling


class CABasicBlock(nn.Module):
    """
    Wrapper for ResNet BasicBlock to inject Coordinate Attention.
    Applies attention after the second BN and before the residual addition.
    """

    def __init__(self, basic_block, channels, reduction=32):
        super().__init__()
        self.basic_block = basic_block
        self.ca = CoordinateAttention(channels, reduction)

    def forward(self, x):
        identity = x

        # Handle downsampling if present in the original block
        if self.basic_block.downsample is not None:
            identity = self.basic_block.downsample(x)

        # First Conv Block
        out = self.basic_block.conv1(x)
        out = self.basic_block.bn1(out)
        out = self.basic_block.relu(out)

        # Second Conv Block
        out = self.basic_block.conv2(out)
        out = self.basic_block.bn2(out)

        # Inject Coordinate Attention
        out = self.ca(out)

        # Residual Connection
        out += identity
        out = self.basic_block.relu(out)

        return out


class TimePreservingResNet18(nn.Module):
    """
    ResNet-18 Backbone modified for Audio Spectrograms.
    1. 1-Channel Input.
    2. Asymmetric Strides (2, 1) in deeper layers to preserve time resolution.
    3. Coordinate Attention in every block.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        # Load standard ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # 1. Modify first conv for 1-channel input
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained:
            # Average the weights across the RGB channels to initialize the 1-channel weight
            with torch.no_grad():
                self.backbone.conv1.weight.data = original_conv1.weight.data.mean(
                    dim=1, keepdim=True
                )

        # 2. Modify Strides for Time Preservation
        # Layer 3: Stride (2, 1) instead of (2, 2)
        self.backbone.layer3[0].conv1.stride = (2, 1)
        if self.backbone.layer3[0].downsample is not None:
            self.backbone.layer3[0].downsample[0].stride = (2, 1)

        # Layer 4: Stride (2, 1) instead of (2, 2)
        self.backbone.layer4[0].conv1.stride = (2, 1)
        if self.backbone.layer4[0].downsample is not None:
            self.backbone.layer4[0].downsample[0].stride = (2, 1)

        # 3. Inject Coordinate Attention into all blocks
        self._inject_ca(self.backbone.layer1, 64)
        self._inject_ca(self.backbone.layer2, 128)
        self._inject_ca(self.backbone.layer3, 256)
        self._inject_ca(self.backbone.layer4, 512)

    def _inject_ca(self, layer, channels):
        """Wraps every block in the layer with CABasicBlock."""
        for i in range(len(layer)):
            layer[i] = CABasicBlock(layer[i], channels)

    def forward(self, x):
        # Stem
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Layers
        c1 = self.backbone.layer1(x)  # 64 ch
        c2 = self.backbone.layer2(c1)  # 128 ch
        c3 = self.backbone.layer3(c2)  # 256 ch
        c4 = self.backbone.layer4(c3)  # 512 ch

        # Return features for FPN (L2, L3, L4)
        return [c2, c3, c4]


class SpecFPN_CRNN(nn.Module):
    """
    Ensemble of SpecFPN-Enhanced Time-Preserving ResNet-18 CRNN.

    Architecture:
    1. TimePreservingResNet18 Backbone (w/ Coordinate Attention).
    2. SpecFPN Neck (Fuses L2, L3, L4).
    3. Bottleneck (Compresses Channels * Freq).
    4. Bi-GRU.
    5. Attention Pooling.
    6. Classifier.
    """

    def __init__(self):
        super().__init__()
        self.config = ModelConfig

        # 1. Backbone
        self.backbone = TimePreservingResNet18(pretrained=self.config.PRETRAINED)

        # 2. SpecFPN Neck
        # ResNet18 channel sizes: Layer2=128, Layer3=256, Layer4=512
        self.fpn = SpecFPN(
            in_channels_list=[128, 256, 512], out_channels=self.config.FPN_OUT_CHANNELS
        )

        # 3. Bottleneck
        # Calculate input dimension for RNN.
        # Input Mel: 128 bins.
        # ResNet Stem (Conv1+Pool): /4 -> 32 bins.
        # Layer 1: /1 -> 32 bins.
        # Layer 2: /2 -> 16 bins.
        # Layer 3: /2 (Freq only) -> 8 bins.
        # Layer 4: /2 (Freq only) -> 4 bins.
        # SpecFPN output is aligned with Layer 2 resolution -> 16 bins.

        self.freq_bins = 16
        fpn_flat_dim = self.config.FPN_OUT_CHANNELS * self.freq_bins

        # Compress to RNN hidden size * 2 (for bidirectional capacity)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(fpn_flat_dim, self.config.RNN_HIDDEN_SIZE * 2, kernel_size=1),
            nn.BatchNorm1d(self.config.RNN_HIDDEN_SIZE * 2),
            nn.ReLU(),
        )

        # 4. RNN
        self.rnn = nn.GRU(
            input_size=self.config.RNN_HIDDEN_SIZE * 2,
            hidden_size=self.config.RNN_HIDDEN_SIZE,
            num_layers=self.config.RNN_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=self.config.RNN_DROPOUT if self.config.RNN_LAYERS > 1 else 0,
        )

        # 5. Head
        self.attention_pool = AttentionPooling(self.config.RNN_HIDDEN_SIZE * 2)
        self.classifier = nn.Linear(self.config.RNN_HIDDEN_SIZE * 2, 1)

    def forward(self, x):
        # x: (Batch, 1, Freq, Time)

        # Backbone Feature Extraction
        features = self.backbone(x)  # [c2, c3, c4]

        # FPN Fusion
        # Output: (B, FPN_Channels, Freq_L2, Time)
        fused_map = self.fpn(features)

        # Flatten Frequency into Channels
        # (B, C, F, T) -> (B, C*F, T)
        b, c, f, t = fused_map.shape
        x_flat = fused_map.flatten(1, 2)

        # Bottleneck
        x_compressed = self.bottleneck(x_flat)  # (B, Hidden*2, T)

        # Prepare for RNN
        # (B, Hidden*2, T) -> (B, T, Hidden*2)
        x_rnn_in = x_compressed.permute(0, 2, 1)

        # RNN
        self.rnn.flatten_parameters()
        x_rnn_out, _ = self.rnn(x_rnn_in)  # (B, T, Hidden*2)

        # Attention Pooling
        embedding = self.attention_pool(x_rnn_out)  # (B, Hidden*2)

        # Classifier
        logits = self.classifier(embedding)  # (B, 1)

        return torch.sigmoid(logits)
