import torch
import torch.nn as nn
import timm
from library.config import Config


class RawEncoder(nn.Module):
    """
    Stream A: Processes raw EEG waveforms using 1D-CNN and Bi-directional GRU.
    Input: (Batch, Channels, Time) -> (B, 19, 2500)
    """

    def __init__(self, input_channels=19, hidden_dim=128):
        super(RawEncoder, self).__init__()

        # 1D CNN for local feature extraction and downsampling
        # Reduces time dimension from 2500 to approx 39 to make GRU efficient
        self.features = nn.Sequential(
            # Block 1: 2500 -> 1250 -> 625
            nn.Conv1d(
                input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Block 2: 625 -> 313 -> 156
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Block 3: 156 -> 78 -> 39
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Block 4: Feature refinement (no pooling)
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(),
        )

        # Bi-directional GRU for temporal dependencies
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        # Output dimension: hidden_size * 2 (bidirectional)
        self.output_dim = hidden_dim * 2

    def forward(self, x):
        # x: (Batch, 19, 2500)

        # CNN Feature Extraction
        x = self.features(x)  # -> (Batch, 256, 39)

        # Permute for GRU: (Batch, Time, Features)
        x = x.permute(0, 2, 1)  # -> (Batch, 39, 256)

        # GRU Processing
        x, _ = self.gru(x)  # -> (Batch, 39, 256)

        # Global Average Pooling over time
        x = torch.mean(x, dim=1)  # -> (Batch, 256)

        return x


class SpecEncoder(nn.Module):
    """
    Stream B: Processes Multi-Channel Spectrograms using EfficientNet-B0.
    Input: (Batch, Channels, Freq, Time) -> (B, 19, 64, 256)
    """

    def __init__(self, input_channels=19, pretrained=True):
        super(SpecEncoder, self).__init__()

        # EfficientNet-B0 Backbone
        # We modify in_chans to accept 19 stacked spectrograms directly
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            in_chans=input_channels,
            num_classes=0,  # Return feature vector, not logits
            global_pool="avg",
        )

        self.output_dim = self.backbone.num_features  # 1280 for EfficientNet-B0

    def forward(self, x):
        # x: (Batch, 19, 64, 256)
        # EfficientNet expects (B, C, H, W), which matches our input structure
        x = self.backbone(x)  # -> (Batch, 1280)
        return x


class HybridEEGModel(nn.Module):
    """
    Dual-Stream Hybrid Model fusing Raw Waveform and Spectrogram features.
    """

    def __init__(self, num_classes=Config.N_CLASSES, pretrained_spec=True):
        super(HybridEEGModel, self).__init__()

        # Initialize Encoders
        self.raw_encoder = RawEncoder(input_channels=Config.N_CHANNELS, hidden_dim=128)
        self.spec_encoder = SpecEncoder(
            input_channels=Config.N_CHANNELS, pretrained=pretrained_spec
        )

        # Fusion Dimension
        fusion_dim = self.raw_encoder.output_dim + self.spec_encoder.output_dim

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fusion_dim, num_classes),
            nn.Softmax(dim=1),  # Output probabilities for KL Divergence
        )

    def forward(self, raw_x, spec_x):
        # raw_x: (Batch, 19, 2500)
        # spec_x: (Batch, 19, 64, 256)

        # Stream A
        raw_feat = self.raw_encoder(raw_x)

        # Stream B
        spec_feat = self.spec_encoder(spec_x)

        # Fusion
        fused = torch.cat([raw_feat, spec_feat], dim=1)

        # Prediction
        probs = self.classifier(fused)

        return probs
