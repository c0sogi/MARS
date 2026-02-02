import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling Layer.
    Computes a weighted average of the input sequence using a learned attention mechanism.
    """

    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x):
        # x shape: (Batch, Time, Features)

        # Compute attention scores
        # scores shape: (Batch, Time, 1)
        scores = self.attention(x)

        # Normalize scores across the time dimension
        weights = torch.softmax(scores, dim=1)

        # Weighted sum: (Batch, Features)
        # Broadcasting weights: (B, T, 1) * (B, T, F) -> (B, T, F) -> sum(dim=1)
        output = torch.sum(x * weights, dim=1)

        return output


class MultiResConvNeXtCRNN(nn.Module):
    """
    Multi-Resolution ConvNeXt-CRNN.

    Architecture:
    1. Input: 3-Channel Multi-Resolution Log-Mel Spectrogram.
    2. Backbone: ConvNeXt Tiny (Pretrained).
       - Strides modified in later stages to preserve temporal resolution.
    3. Neck: Frequency Averaging + BiGRU.
    4. Head: Attention Pooling + Linear Classifier.
    """

    def __init__(self):
        super(MultiResConvNeXtCRNN, self).__init__()

        # 1. Backbone: ConvNeXt Tiny
        # Using DEFAULT weights (ImageNet1K V1)
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        self.backbone = convnext_tiny(weights=weights)

        # Modify strides to preserve temporal resolution
        # The 'features' sequential container has the following structure for ConvNeXt:
        # 0: Conv2dNormActivation (Stem, stride=4)
        # 1: Sequential (Stage 0 Blocks)
        # 2: Sequential (Downsample Layer 0->1, stride=2)
        # 3: Sequential (Stage 1 Blocks)
        # 4: Sequential (Downsample Layer 1->2, stride=2) -> Target for modification
        # 5: Sequential (Stage 2 Blocks)
        # 6: Sequential (Downsample Layer 2->3, stride=2) -> Target for modification
        # 7: Sequential (Stage 3 Blocks)

        # We modify the Conv2d layer within the downsampling blocks at indices 4 and 6.
        # The downsample block is usually: Sequential(LayerNorm, Conv2d)
        # So we access [1] for the Conv2d layer.

        # Modify Downsample Layer before Stage 2 (features[4])
        # Changing stride from 2 to 1 prevents halving the feature map size
        self.backbone.features[4][1].stride = (1, 1)

        # Modify Downsample Layer before Stage 3 (features[6])
        self.backbone.features[6][1].stride = (1, 1)

        # Determine backbone output channels
        # ConvNeXt Tiny final stage output channels = 768
        self.backbone_out_channels = 768

        # 2. Recurrent Layer (BiGRU)
        self.gru = nn.GRU(
            input_size=self.backbone_out_channels,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT,
        )

        # 3. Attention Pooling
        # BiGRU output dimension is hidden_size * 2
        gru_out_dim = Config.HIDDEN_SIZE * 2
        self.attn_pooling = AttentionPooling(gru_out_dim)

        # 4. Classifier
        self.classifier = nn.Linear(gru_out_dim, Config.NUM_CLASSES)

    def forward(self, x):
        # Input x: (Batch, 3, F, T)

        # Pass through Backbone
        # Output shape depends on input size.
        # With 1 sec audio (16k samples), hop=160 -> ~100 frames.
        # Stem (s=4) -> 25. Downsample 0 (s=2) -> 12.
        # Modified Downsamples (s=1) keep it at ~12 frames.
        x = self.backbone.features(x)  # (Batch, 768, F', T')

        # Average over Frequency dimension
        # We assume the relevant features are distributed across frequency but we want to aggregate them
        # to form a sequence of temporal feature vectors.
        x = x.mean(dim=2)  # (Batch, 768, T')

        # Permute for RNN: (Batch, T', 768)
        x = x.permute(0, 2, 1)

        # Pass through BiGRU
        self.gru.flatten_parameters()
        x, _ = self.gru(x)  # (Batch, T', 2*Hidden)

        # Attention Pooling
        # Aggregates the sequence into a single vector
        x = self.attn_pooling(x)  # (Batch, 2*Hidden)

        # Classification
        x = self.classifier(x)  # (Batch, NumClasses)

        return x
