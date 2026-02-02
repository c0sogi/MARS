import torch
import torch.nn as nn
import timm


class InceptionBlock1D(nn.Module):
    """
    A 1D Inception-style block that applies parallel convolutions with different
    kernel sizes to capture multi-scale temporal features.
    """

    def __init__(self, in_channels, out_channels, kernel_sizes):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            # Padding is k // 2 to maintain temporal dimension
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=k,
                        padding=k // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x):
        # Concatenate branch outputs along the channel dimension
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class EEGNet1D(nn.Module):
    """
    Stream A: Local Temporal Stream for Raw EEG.
    Uses a stack of InceptionBlock1D layers and MaxPooling.
    """

    def __init__(self, in_channels, kernel_sizes, filters):
        super().__init__()

        # Initial stem to project input channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, filters[0], kernel_size=1, bias=False),
            nn.BatchNorm1d(filters[0]),
            nn.ReLU(inplace=True),
        )

        layers = []
        curr_channels = filters[0]

        for i, f in enumerate(filters):
            # Inception Block
            # Input channels = curr_channels
            # Output channels = f * len(kernel_sizes) (concatenated)
            block = InceptionBlock1D(curr_channels, f, kernel_sizes)
            layers.append(block)

            curr_channels = f * len(kernel_sizes)

            # Pooling (reduce temporal dim by 4 at each stage)
            # We skip pooling after the last block to do GlobalAvgPool instead
            if i < len(filters) - 1:
                layers.append(nn.MaxPool1d(kernel_size=4, stride=4))

        self.encoder = nn.Sequential(*layers)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = curr_channels

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        x = self.stem(x)
        x = self.encoder(x)
        x = self.global_pool(x)
        return x.flatten(1)


class SpecNet2D(nn.Module):
    """
    Stream B: Global Context Stream for Spectrograms.
    Uses an EfficientNet-B0 backbone adapted for 1-channel input.
    """

    def __init__(self, backbone_name, pretrained):
        super().__init__()
        # Create model using timm
        # in_chans=1 adapts the first conv layer for grayscale input
        # num_classes=0 returns the pooled feature vector
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=1,
            global_pool="avg",
        )
        self.out_dim = self.backbone.num_features

    def forward(self, x):
        # x shape: (Batch, 1, Height, Width)
        return self.backbone(x)


class HybridModel(nn.Module):
    """
    Context-Aware Dual-Stream Hybrid Network.
    Fuses features from EEGNet1D and SpecNet2D.
    """

    def __init__(self, config):
        super().__init__()

        # 1. EEG Stream
        self.eeg_net = EEGNet1D(
            in_channels=config.eeg_channels,
            kernel_sizes=config.kernel_sizes_1d,
            filters=config.filters_1d,
        )

        # 2. Spectrogram Stream
        self.spec_net = SpecNet2D(
            backbone_name=config.backbone_2d, pretrained=config.pretrained_2d
        )

        # 3. Fusion Head
        fusion_in_dim = self.eeg_net.out_dim + self.spec_net.out_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, config.fusion_hidden_dim),
            nn.BatchNorm1d(config.fusion_hidden_dim),
            nn.SiLU(),  # Swish activation
            nn.Dropout(config.drop_rate),
            nn.Linear(config.fusion_hidden_dim, config.num_classes),
        )

    def forward(self, x_eeg, x_spec):
        """
        Args:
            x_eeg: (Batch, Channels, Time)
            x_spec: (Batch, 1, Height, Width)
        Returns:
            logits: (Batch, Num_Classes)
        """
        # Extract features
        feat_1d = self.eeg_net(x_eeg)  # (Batch, eeg_dim)
        feat_2d = self.spec_net(x_spec)  # (Batch, spec_dim)

        # Concatenate
        combined = torch.cat([feat_1d, feat_2d], dim=1)

        # Classify
        logits = self.fusion_head(combined)

        return logits
