import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class VisualBackbone(nn.Module):
    """
    Modified ResNet-18 backbone for 1-channel input and custom downsampling.
    Preserves horizontal resolution to ensure sufficient sequence length for CTC.
    """

    def __init__(self):
        super().__init__()
        # Load pretrained ResNet18
        base = models.resnet18(pretrained=Config.ENCODER_PRETRAINED)

        # 1. Modify input layer to accept 1 channel (grayscale)
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(
            Config.INPUT_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize with average of pretrained RGB weights
        with torch.no_grad():
            self.conv1.weight.data = base.conv1.weight.data.mean(dim=1, keepdim=True)

        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool  # Stride 2 (H/4, W/4)

        # 2. Extract layers
        self.layer1 = base.layer1  # Stride 1 (H/4, W/4)
        self.layer2 = (
            base.layer2
        )  # Original Stride 2 (H/8, W/8) -> Modify to (H/8, W/4)
        self.layer3 = (
            base.layer3
        )  # Original Stride 2 (H/16, W/16) -> Modify to (H/16, W/4)
        self.layer4 = (
            base.layer4
        )  # Original Stride 2 (H/32, W/32) -> Modify to (H/32, W/4)

        # 3. Modify strides to preserve width resolution
        # We want vertical downsampling but less horizontal downsampling
        # Layer 2
        self.layer2[0].conv1.stride = (2, 1)
        if self.layer2[0].downsample is not None:
            self.layer2[0].downsample[0].stride = (2, 1)

        # Layer 3
        self.layer3[0].conv1.stride = (2, 1)
        if self.layer3[0].downsample is not None:
            self.layer3[0].downsample[0].stride = (2, 1)

        # Layer 4
        self.layer4[0].conv1.stride = (2, 1)
        if self.layer4[0].downsample is not None:
            self.layer4[0].downsample[0].stride = (2, 1)

    def forward(self, x):
        # x: (B, 1, 256, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (B, 64, 64, W/4)

        x = self.layer1(x)  # (B, 64, 64, W/4)
        x = self.layer2(x)  # (B, 128, 32, W/4)
        x = self.layer3(x)  # (B, 256, 16, W/4)
        x = self.layer4(x)  # (B, 512, 8, W/4)

        return x


class GatedBlock(nn.Module):
    """
    1D Convolutional block with Gated Linear Unit (GLU).
    Uses dilation to expand receptive field.
    """

    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        # Conv1d outputs 2*channels for GLU (split in half)
        # Padding = dilation ensures output length == input length for kernel_size=3
        self.conv = nn.Conv1d(
            channels, channels * 2, kernel_size, padding=dilation, dilation=dilation
        )
        self.dropout = nn.Dropout(dropout)
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        residual = x
        out = self.dropout(x)
        out = self.conv(out)
        out = self.glu(out)
        return out + residual


class SequenceEncoder(nn.Module):
    """
    Fully Convolutional Sequence Encoder.
    Stacks GatedBlocks with increasing dilation.
    """

    def __init__(self, input_channels, hidden_channels, num_layers=5):
        super().__init__()
        # Project backbone features to hidden dimension
        self.proj = nn.Conv1d(input_channels, hidden_channels, 1)

        layers = []
        # Exponential dilation schedule: 1, 2, 4, 8, 16...
        for i in range(num_layers):
            dilation = 2**i
            layers.append(
                GatedBlock(hidden_channels, dilation=dilation, dropout=Config.DROPOUT)
            )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.proj(x)
        x = self.layers(x)
        return x


class GFCN(nn.Module):
    """
    Gated Fully Convolutional Network for InChI recognition.
    Structure: Visual Backbone -> Vertical Collapse -> Sequence Encoder -> Head
    """

    def __init__(self, num_classes):
        super().__init__()
        self.backbone = VisualBackbone()

        # ResNet18 Layer 4 outputs 512 channels
        backbone_out_channels = 512

        self.encoder = SequenceEncoder(
            input_channels=backbone_out_channels,
            hidden_channels=Config.DECODER_CHANNELS,
        )

        # Final projection to vocabulary size
        self.head = nn.Conv1d(Config.DECODER_CHANNELS, num_classes, 1)

    def forward(self, x):
        # 1. Visual Feature Extraction
        # Input: (B, 1, H, W) -> Output: (B, 512, H/32, W/4)
        features = self.backbone(x)

        # 2. Vertical Collapse
        # Max pool along height dimension to create 1D sequence
        # (B, 512, H', W') -> (B, 512, W')
        features = features.max(dim=2)[0]

        # 3. Sequence Encoding
        # (B, 512, W') -> (B, Hidden, W')
        features = self.encoder(features)

        # 4. Prediction Head
        # (B, Hidden, W') -> (B, NumClasses, W')
        logits = self.head(features)

        # Permute to (B, T, C) for compatibility with standard loss/decoding
        logits = logits.permute(0, 2, 1)

        return logits
