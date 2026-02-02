import torch
import torch.nn as nn
from library.config import Config
from library.backbone import TimePreservingResNet18
from library.layers import AdaptiveSpectralFusion, AttentionPooling


class HierarchicalCRNN(nn.Module):
    """
    Spectrally-Adaptive Hierarchical CA-ResNet-18 CRNN.

    Architecture:
    1. Backbone: Time-Preserving ResNet-18 with Coordinate Attention.
       - Extracts features from Layer 2, 3, and 4.
    2. Fusion: Adaptive Spectral Fusion.
       - Pools features to specific frequency bins and fuses them via bottleneck.
    3. Temporal Modeling: Bi-Directional GRU.
    4. Aggregation: Attention Pooling.
    5. Classification: Linear Layer.
    """

    def __init__(self):
        super(HierarchicalCRNN, self).__init__()

        # 1. Backbone
        self.backbone = TimePreservingResNet18(pretrained=Config.PRETRAINED)

        # 2. Adaptive Spectral Fusion
        # ResNet18 channel sizes for layers 2, 3, 4 are [128, 256, 512]
        backbone_channels = [128, 256, 512]

        self.fusion = AdaptiveSpectralFusion(
            in_channels_list=backbone_channels,
            pool_bins_list=Config.SPECTRAL_POOL_BINS,
            fusion_channels=Config.FUSION_CHANNELS,
        )

        # 3. Temporal Modeling (Bi-GRU)
        self.rnn = nn.GRU(
            input_size=Config.FUSION_CHANNELS,
            hidden_size=Config.RNN_HIDDEN_SIZE,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT if Config.RNN_LAYERS > 1 else 0,
        )

        self.dropout = nn.Dropout(Config.DROPOUT)

        # 4. Attention Pooling
        # Input dim is hidden_size * 2 (bidirectional)
        rnn_out_dim = Config.RNN_HIDDEN_SIZE * 2
        self.pooling = AttentionPooling(input_dim=rnn_out_dim)

        # 5. Classifier
        self.classifier = nn.Linear(rnn_out_dim, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrogram of shape (N, 1, F, T).

        Returns:
            torch.Tensor: Logits of shape (N, 1).
        """
        # 1. Backbone Extraction
        # Returns list of features from [Layer2, Layer3, Layer4]
        # Shapes: [(N, 128, F2, T'), (N, 256, F3, T'), (N, 512, F4, T')]
        features = self.backbone(x)

        # 2. Adaptive Spectral Fusion
        # Fuses hierarchical features into a single sequence
        # Output: (N, Fusion_Channels, T')
        fused = self.fusion(features)

        # 3. Temporal Modeling
        # Permute to (N, T', C) for RNN
        rnn_input = fused.permute(0, 2, 1)

        # RNN Output: (N, T', Hidden*2)
        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(rnn_input)

        rnn_out = self.dropout(rnn_out)

        # 4. Aggregation
        # Output: (N, Hidden*2)
        pooled = self.pooling(rnn_out)

        # 5. Classification
        # Output: (N, 1)
        logits = self.classifier(pooled)

        return logits
