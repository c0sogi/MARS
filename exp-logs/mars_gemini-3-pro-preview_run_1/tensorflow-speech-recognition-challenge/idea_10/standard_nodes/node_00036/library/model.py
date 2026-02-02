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


class SpeechCommandModel(nn.Module):
    """
    Dilated CNN Architecture with Attentive Pooling (Cite solution_lesson_node_00035):
    1. Dilated EfficientNet-B2 Backbone (Spectral Feature Extraction)
    2. Attention Mechanism (Temporal Aggregation)
    3. Fine-Grained Classifier
    """

    def __init__(self):
        super(SpeechCommandModel, self).__init__()

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
        with torch.no_grad():
            # Shape: (Batch, Channels, Freq, Time)
            dummy_input = torch.zeros(
                1,
                Config.IN_CHANNELS,
                Config.N_MELS,
                int(Config.SAMPLE_RATE * Config.DURATION / Config.HOP_LENGTH) + 1,
            )
            features = self.backbone.forward_features(dummy_input)
            self.backbone_channels = features.shape[1]

        # 2. Attention Mechanism
        # Input dim is the backbone channel count
        self.attention = Attention(self.backbone_channels)

        # 3. Classifier
        self.classifier = nn.Linear(self.backbone_channels, Config.NUM_CLASSES)

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
        # Output: (Batch, Channels, Time')
        x = torch.mean(x, dim=2)

        # 3. Permute for Attention
        # Attention expects (Batch, Time, Input_Size)
        x = x.permute(0, 2, 1)

        # 4. Temporal Aggregation (Attention)
        # Output: (Batch, Channels)
        x = self.attention(x)

        # 5. Classification
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits
