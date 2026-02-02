import torch
import torch.nn as nn
import timm
from library.config import (
    NUM_SENSORS,
    RNN_HIDDEN_DIM,
    RNN_NUM_LAYERS,
    RNN_BIDIRECTIONAL,
    RNN_DROPOUT,
    MLP_HIDDEN_DIM,
    DROPOUT_RATE,
    BACKBONE_NAME,
    PRETRAINED,
)


class AttentionPool(nn.Module):
    """
    Attention Pooling Layer.
    Computes a weighted sum of the input sequence based on learned attention scores.
    """

    def __init__(self, input_dim):
        super(AttentionPool, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # x shape: (Batch, Time, Features)
        # weights shape: (Batch, Time, 1)
        weights = self.attention(x)
        # context shape: (Batch, Features)
        context = torch.sum(weights * x, dim=1)
        return context


class HybridCRNN(nn.Module):
    """
    Hybrid CRNN with Attention for Volcanic Eruption Prediction.

    Branch 1: EfficientNet-B0 backbone -> Frequency Pooling -> Bi-GRU -> Attention
    Branch 2: MLP for Statistical Features
    Fusion: Concatenation -> Regression Head
    """

    def __init__(self, num_stats_features=150):
        super(HybridCRNN, self).__init__()

        # ------------------------------------------------------------------
        # Branch 1: Spectrogram CRNN
        # ------------------------------------------------------------------
        # Load EfficientNet-B0 backbone
        # features_only=False, num_classes=0, global_pool='' gives us the feature maps (N, C, H, W)
        self.backbone = timm.create_model(
            BACKBONE_NAME, pretrained=PRETRAINED, num_classes=0, global_pool=""
        )

        # Modify the first layer to accept 10 channels (Sensors)
        self._modify_first_layer(in_channels=NUM_SENSORS)

        # Determine backbone output channels (EfficientNet-B0 usually 1280)
        # We run a dummy forward pass to dynamically determine the size
        with torch.no_grad():
            dummy_spec = torch.randn(1, NUM_SENSORS, 128, 128)
            features = self.backbone(dummy_spec)
            self.cnn_out_dim = features.shape[1]

        # Recurrent Head
        # Input: (Batch, Time, CNN_Features)
        self.rnn = nn.GRU(
            input_size=self.cnn_out_dim,
            hidden_size=RNN_HIDDEN_DIM,
            num_layers=RNN_NUM_LAYERS,
            bidirectional=RNN_BIDIRECTIONAL,
            batch_first=True,
            dropout=RNN_DROPOUT if RNN_NUM_LAYERS > 1 else 0.0,
        )

        rnn_out_dim = RNN_HIDDEN_DIM * (2 if RNN_BIDIRECTIONAL else 1)
        self.attention = AttentionPool(rnn_out_dim)

        # ------------------------------------------------------------------
        # Branch 2: Statistical MLP
        # ------------------------------------------------------------------
        self.mlp = nn.Sequential(
            nn.Linear(num_stats_features, MLP_HIDDEN_DIM),
            nn.BatchNorm1d(MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(MLP_HIDDEN_DIM, MLP_HIDDEN_DIM),
            nn.BatchNorm1d(MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )

        # ------------------------------------------------------------------
        # Fusion Head
        # ------------------------------------------------------------------
        fusion_dim = rnn_out_dim + MLP_HIDDEN_DIM
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, 1),
        )

    def _modify_first_layer(self, in_channels):
        """
        Adapts the first convolutional layer to accept `in_channels`.
        Initializes weights by averaging RGB channels and replicating.
        """
        # In timm efficientnet, the first layer is named 'conv_stem'
        old_conv = self.backbone.conv_stem

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Weight Initialization
        with torch.no_grad():
            # old_conv.weight shape: (Out, 3, K, K)
            # Average over the RGB dimension (dim 1)
            weight_avg = old_conv.weight.mean(dim=1, keepdim=True)  # (Out, 1, K, K)
            # Replicate for new input channels
            weight_new = weight_avg.repeat(1, in_channels, 1, 1)  # (Out, 10, K, K)

            new_conv.weight.copy_(weight_new)

            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        self.backbone.conv_stem = new_conv

    def forward(self, spec, stats):
        """
        Args:
            spec (torch.Tensor): Spectrogram input (Batch, 10, Freq, Time)
            stats (torch.Tensor): Statistical features (Batch, 150)
        Returns:
            torch.Tensor: Predicted time to eruption (Batch,)
        """
        # --- Spectrogram Branch ---
        # 1. Extract Spatial/Frequency Features
        # Output shape: (Batch, C, F', T')
        x = self.backbone(spec)

        # 2. Frequency-Global Pooling
        # We want to keep the Time dimension (T') and collapse Frequency (F').
        # x shape: (Batch, Channels, Freq, Time) -> mean over dim 2 (Freq)
        x = x.mean(dim=2)  # Shape: (Batch, Channels, Time)

        # 3. Prepare for RNN
        # RNN expects (Batch, Time, Features)
        x = x.permute(0, 2, 1)  # Shape: (Batch, Time, Channels)

        # 4. Recurrent Processing
        x, _ = self.rnn(x)  # Shape: (Batch, Time, Hidden*Dir)

        # 5. Attention Pooling
        x = self.attention(x)  # Shape: (Batch, Hidden*Dir)

        # --- Statistics Branch ---
        s = self.mlp(stats)  # Shape: (Batch, MLP_Hidden)

        # --- Fusion ---
        combined = torch.cat([x, s], dim=1)
        out = self.regressor(combined)

        return out.squeeze(1)
