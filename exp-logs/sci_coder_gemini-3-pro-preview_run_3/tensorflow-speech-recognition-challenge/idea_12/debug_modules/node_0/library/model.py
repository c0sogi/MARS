import torch
import torch.nn as nn
import timm
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Multi-Head Attention Pooling layer.
    Uses a learnable Query vector to aggregate the temporal sequence.
    """

    def __init__(self, input_dim, num_heads=4):
        super(AttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads

        # MultiheadAttention: Batch first ensures (Batch, Seq, Dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True
        )

        # Learnable Query Vector: (1, 1, Dim)
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))

        # LayerNorm for stability
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input sequence of shape (Batch, Time, Dim)
        Returns:
            torch.Tensor: Pooled representation of shape (Batch, Dim)
        """
        batch_size = x.size(0)

        # Expand query to match batch size: (Batch, 1, Dim)
        query = self.query.repeat(batch_size, 1, 1)

        # Attention: Query attends to Key/Value (x)
        # attn_output shape: (Batch, 1, Dim)
        attn_output, _ = self.mha(query, x, x)

        # Remove time dimension
        out = attn_output.squeeze(1)

        return self.layer_norm(out)


class FrequencyPreservingSKResNetCRNN(nn.Module):
    """
    Frequency-Preserving Multi-Resolution SK-ResNet-CRNN.

    Architecture:
    1. Input: 3-Channel Multi-Res Log-Mel Spectrogram.
    2. Backbone: SK-ResNet34 with modified strides (1,1) in Layer 3 & 4.
    3. Neck: Frequency-Preserving Flattening -> Linear Projection -> BiGRU.
    4. Head: Multi-Head Attention Pooling -> Classifier.
    """

    def __init__(self):
        super(FrequencyPreservingSKResNetCRNN, self).__init__()

        # 1. Backbone: SK-ResNet34
        # We use features_only=False but access forward_features to get the map before pooling
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            in_chans=Config.IN_CHANNELS,
        )

        # 2. Modify Strides to preserve resolution
        # Standard ResNet downsamples by factor 32 (Stem /4, L1 /1, L2 /2, L3 /2, L4 /2)
        # We modify L3 and L4 to stride 1, resulting in total downsample /8.
        self._modify_strides(self.backbone.layer3)
        self._modify_strides(self.backbone.layer4)

        # 3. Calculate Feature Dimensions
        # Input Freq: 80 Mels
        # Total Downsampling: 8
        # Final Freq Dim: 80 / 8 = 10
        self.freq_dim = 10

        # ResNet34 Layer 4 output channels: 512
        self.backbone_channels = 512

        # Flattened Dimension: Channels * Freq
        self.flattened_dim = self.backbone_channels * self.freq_dim  # 5120

        # 4. Projection Layer
        self.projection = nn.Sequential(
            nn.Linear(self.flattened_dim, Config.PROJECTION_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # 5. Bidirectional GRU
        self.rnn = nn.GRU(
            input_size=Config.PROJECTION_DIM,
            hidden_size=Config.RNN_HIDDEN_SIZE,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.RNN_DROPOUT if Config.RNN_LAYERS > 1 else 0,
        )

        # BiGRU output dimension
        self.rnn_out_dim = Config.RNN_HIDDEN_SIZE * 2

        # 6. Attention Pooling
        self.attn_pooling = AttentionPooling(
            self.rnn_out_dim, num_heads=Config.ATTENTION_HEADS
        )

        # 7. Classifier
        self.classifier = nn.Linear(self.rnn_out_dim, Config.NUM_CLASSES)

    def _modify_strides(self, layer):
        """
        Recursively sets stride to (1, 1) for all Conv2d layers that have stride (2, 2).
        This effectively removes downsampling from the specified layer.
        """
        for module in layer.modules():
            if isinstance(module, nn.Conv2d):
                if module.stride == (2, 2):
                    module.stride = (1, 1)
            # Handle Downsample blocks which might use AvgPool or MaxPool in some variants
            elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
                if module.stride == 2 or module.stride == (2, 2):
                    module.stride = 1

    def forward(self, x):
        # x shape: (Batch, 3, 80, Time)

        # 1. Backbone Feature Extraction
        # forward_features returns (Batch, 512, F_out, T_out)
        x = self.backbone.forward_features(x)

        # 2. Frequency-Preserving Flattening
        # Permute to (Batch, Time, Channels, Freq)
        x = x.permute(0, 3, 1, 2)

        B, T, C, F = x.shape
        # Flatten Channels and Freq: (Batch, Time, C*F)
        x = x.reshape(B, T, C * F)

        # 3. Projection
        x = self.projection(x)  # (Batch, Time, 1024)

        # 4. RNN
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)  # (Batch, Time, 512)

        # 5. Attention Pooling
        x = self.attn_pooling(x)  # (Batch, 512)

        # 6. Classifier
        x = self.classifier(x)  # (Batch, 12)

        return x
