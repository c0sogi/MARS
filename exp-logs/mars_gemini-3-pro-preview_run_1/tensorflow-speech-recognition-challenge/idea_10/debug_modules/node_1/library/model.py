import torch
import torch.nn as nn
import timm
from library.config import Config


class Attention(nn.Module):
    """
    Attention mechanism to aggregate temporal features from the GRU.
    Computes a weighted sum of the sequence steps.
    """

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        # x shape: (Batch, Time, Hidden)

        # Calculate attention scores
        # scores shape: (Batch, Time, 1)
        scores = self.attention(x)

        # Normalize scores across the time dimension
        weights = torch.softmax(scores, dim=1)

        # Weighted sum
        # (Batch, Time, Hidden) * (Batch, Time, 1) -> (Batch, Time, Hidden)
        # Sum over Time -> (Batch, Hidden)
        context_vector = torch.sum(x * weights, dim=1)

        return context_vector


class HybridCRNN(nn.Module):
    """
    Hybrid CRNN Architecture:
    1. Dilated EfficientNet-B2 Backbone (Spectral Feature Extraction)
    2. Bi-Directional GRU (Temporal/Phonetic Sequence Modeling)
    3. Attention Mechanism (Temporal Aggregation)
    4. Fine-Grained Classifier
    """

    def __init__(self):
        super(HybridCRNN, self).__init__()

        # 1. Backbone: Dilated EfficientNet-B2
        # in_chans=1: Adapts first layer weights (avg of RGB)
        # output_stride=16: Enforces dilation in the final stage (stride 1, dilation 2)
        # instead of downsampling to stride 32. This preserves temporal resolution (~6-7 frames).
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            in_chans=Config.IN_CHANNELS,
            output_stride=16,
        )

        # Determine the output feature dimension of the backbone dynamically
        # We use a dummy input matching the config specs
        with torch.no_grad():
            # Shape: (Batch, Channels, Freq, Time)
            # Freq = N_MELS (128), Time = ~100 (16000/160)
            dummy_input = torch.zeros(
                1,
                Config.IN_CHANNELS,
                Config.N_MELS,
                int(Config.SAMPLE_RATE * Config.DURATION / Config.HOP_LENGTH) + 1,
            )
            features = self.backbone.forward_features(dummy_input)
            # features shape is (1, C, F', T')
            self.backbone_channels = features.shape[1]

        # 2. Sequential Head (Bi-Directional GRU)
        # Input size is the channel dimension of the backbone output
        self.gru = nn.GRU(
            input_size=self.backbone_channels,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=Config.GRU_LAYERS,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
            dropout=Config.DROPOUT if Config.GRU_LAYERS > 1 else 0.0,
        )

        gru_out_dim = (
            Config.HIDDEN_DIM * 2 if Config.BIDIRECTIONAL else Config.HIDDEN_DIM
        )

        # 3. Attention Mechanism
        self.attention = Attention(gru_out_dim)

        # 4. Classifier
        self.classifier = nn.Linear(gru_out_dim, Config.NUM_CLASSES)

        # Dropout for regularization before classifier
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrograms of shape (Batch, 1, Freq, Time)
        """
        # 1. Backbone Feature Extraction
        # Output: (Batch, Channels, Freq', Time')
        x = self.backbone.forward_features(x)

        # 2. Frequency Pooling
        # We average over the frequency dimension to get a sequence of feature vectors over time.
        # Output: (Batch, Channels, Time')
        x = torch.mean(x, dim=2)

        # 3. Permute for RNN
        # GRU expects (Batch, Time, Input_Size)
        x = x.permute(0, 2, 1)

        # 4. Sequential Modeling (GRU)
        self.gru.flatten_parameters()
        # Output: (Batch, Time, Hidden*Dir)
        x, _ = self.gru(x)

        # 5. Temporal Aggregation (Attention)
        # Output: (Batch, Hidden*Dir)
        x = self.attention(x)

        # 6. Classification
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits
