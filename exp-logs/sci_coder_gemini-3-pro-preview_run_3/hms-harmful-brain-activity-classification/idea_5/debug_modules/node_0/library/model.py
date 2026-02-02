import torch
import torch.nn as nn
import timm
import library.config as config


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation Block adapted for 1D signals.
    Enhances channel interdependencies.
    """

    def __init__(self, channel, reduction=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class BasicBlock1D(nn.Module):
    """
    Standard ResNet Basic Block with 1D Convolutions and SE Block.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SEBlock1D(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply SE Block
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    """
    ResNet-18 adapted for 1D EEG data.
    Input: (Batch, 19, 2500)
    """

    def __init__(self):
        super(ResNet1D, self).__init__()
        self.inplanes = 64

        # Initial Convolution: Adapt 19 channels to 64
        self.conv1 = nn.Conv1d(
            config.EEG_CHANNELS_COUNT,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet Layers
        layers = config.RESNET_BLOCKS
        filters = config.RESNET_FILTERS

        self.layer1 = self._make_layer(BasicBlock1D, filters[0], layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock1D, filters[1], layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock1D, filters[2], layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock1D, filters[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(config.EEG_DROPOUT)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return x


class EfficientNetSpec(nn.Module):
    """
    EfficientNet Encoder for Spectrograms.
    Input: (Batch, 4, 256, 256)
    """

    def __init__(self):
        super(EfficientNetSpec, self).__init__()
        # Load pretrained EfficientNet, modify input channels to 4
        self.net = timm.create_model(
            config.EFFICIENTNET_VERSION,
            pretrained=True,
            in_chans=config.SPEC_CHANNELS,
            num_classes=0,  # Return features, not logits
        )
        self.dropout = nn.Dropout(config.SPEC_DROPOUT)

    def forward(self, x):
        x = self.net(x)
        x = self.dropout(x)
        return x


class HybridModel(nn.Module):
    """
    Hybrid Model fusing 1D EEG ResNet and 2D Spectrogram EfficientNet.
    """

    def __init__(self):
        super(HybridModel, self).__init__()
        self.eeg_encoder = ResNet1D()
        self.spec_encoder = EfficientNetSpec()

        # Determine feature dimensions
        eeg_dim = config.RESNET_FILTERS[-1]  # 512
        spec_dim = self.spec_encoder.net.num_features  # Typically 1280 for B0

        fusion_input_dim = eeg_dim + spec_dim

        # Fusion Head
        self.head = nn.Sequential(
            nn.Linear(fusion_input_dim, config.FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.FUSION_DROPOUT),
            nn.Linear(config.FUSION_HIDDEN_DIM, config.NUM_CLASSES),
        )

    def forward(self, x_eeg, x_spec):
        # Stream A: EEG
        feat_eeg = self.eeg_encoder(x_eeg)

        # Stream B: Spectrogram
        feat_spec = self.spec_encoder(x_spec)

        # Fusion
        fused = torch.cat([feat_eeg, feat_spec], dim=1)

        # Classification
        logits = self.head(fused)
        return logits
