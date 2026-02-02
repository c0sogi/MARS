import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class BasicBlock1D(nn.Module):
    """
    Basic Residual Block for 1D ResNet.
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet1D(nn.Module):
    """
    ResNet-34 adapted for 1D EEG signal processing.
    Input: (Batch, Channels, Time)
    """

    def __init__(self, block, num_blocks, num_classes=6, in_channels=19):
        super(ResNet1D, self).__init__()
        self.in_planes = 64

        # Initial Stem
        self.conv1 = nn.Conv1d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual Layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Output dimension for fusion
        self.output_dim = 512 * block.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Global Average Pooling
        out = F.adaptive_avg_pool1d(out, 1)
        out = out.view(out.size(0), -1)
        return out


def resnet34_1d(in_channels=19):
    """Constructs a ResNet-34 1D model."""
    return ResNet1D(BasicBlock1D, [3, 4, 6, 3], in_channels=in_channels)


class DualStreamNetwork(nn.Module):
    """
    Orthogonal Dual-Stream Network.
    Stream A: 1D ResNet for Raw EEG (Morphology).
    Stream B: EfficientNet-B0 for Spectrograms (Context/Frequency).
    """

    def __init__(self, num_classes=Config.N_CLASSES, pretrained=True):
        super(DualStreamNetwork, self).__init__()

        # ==========================
        # Stream A: Raw EEG (1D)
        # ==========================
        # Input: (Batch, 2500, 19) -> Transposed to (Batch, 19, 2500) in forward
        self.stream_a = resnet34_1d(in_channels=Config.N_EEG_CHANNELS)
        self.dim_a = self.stream_a.output_dim  # 512

        # ==========================
        # Stream B: Spectrogram (2D)
        # ==========================
        # Input: (Batch, 4, 256, 256)
        # We use timm to load EfficientNet-B0 and adapt input channels to 4
        self.stream_b = timm.create_model(
            Config.BACKBONE_STREAM_B,
            pretrained=pretrained,
            num_classes=0,  # 0 means return features (pooling included usually)
            in_chans=Config.N_SPEC_CHANNELS,  # 4 Channels
        )
        self.dim_b = self.stream_b.num_features  # 1280 for EfficientNet-B0

        # ==========================
        # Fusion Head
        # ==========================
        self.fusion_dim = self.dim_a + self.dim_b

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(self.fusion_dim, num_classes)
        )

    def forward(self, x_eeg, x_spec):
        """
        Args:
            x_eeg: (Batch, Time=2500, Channels=19)
            x_spec: (Batch, Channels=4, Height=256, Width=256)
        """
        # --- Stream A Processing ---
        # Permute EEG to (Batch, Channels, Time) for Conv1d
        x_eeg = x_eeg.permute(0, 2, 1)
        feat_a = self.stream_a(x_eeg)

        # --- Stream B Processing ---
        # x_spec is already (Batch, C, H, W)
        feat_b = self.stream_b(x_spec)

        # --- Fusion ---
        # Concatenate features
        combined = torch.cat((feat_a, feat_b), dim=1)

        # Classification
        logits = self.classifier(combined)

        return logits
