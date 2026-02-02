import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Lightweight Attention Pooling mechanism.
    Computes a weighted sum of the input sequence across the time dimension.
    """

    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        # Learn a weight to score each time step
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Time, Features)

        Returns:
            Tensor of shape (Batch, Features)
        """
        # Calculate attention scores: (Batch, Time, 1)
        scores = self.attention_weights(x)

        # Normalize scores to probabilities over the time dimension
        weights = F.softmax(scores, dim=1)

        # Apply weights to input: (Batch, Time, Features) * (Batch, Time, 1)
        weighted_output = x * weights

        # Sum over time dimension to get context vector
        pooled_output = torch.sum(weighted_output, dim=1)

        return pooled_output


class BiGRUModel(nn.Module):
    """
    Bi-directional GRU model with Attention Pooling for EEG Seizure Detection.
    """

    def __init__(self):
        super(BiGRUModel, self).__init__()

        # Hyperparameters from Config
        self.input_dim = Config.N_CHANNELS
        self.proj_dim = Config.PROJ_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.NUM_LAYERS
        self.num_classes = Config.NUM_CLASSES
        self.dropout = Config.DROPOUT

        # 1. Dimensionality Reduction
        # Projects 19 EEG channels -> 12 dimensions
        # nn.Linear applied to (Batch, Time, Channels) acts as a 1x1 Conv / Time-Distributed Dense
        self.projection = nn.Linear(self.input_dim, self.proj_dim)

        # 2. Bi-directional GRU
        # Captures temporal dependencies
        self.gru = nn.GRU(
            input_size=self.proj_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )

        # 3. Attention Pooling
        # Input dimension is Hidden_Dim * 2 due to bidirectional GRU
        self.attention = AttentionPooling(self.hidden_dim * 2)

        # 4. Output Head
        # Maps pooled representation to class logits
        self.classifier = nn.Linear(self.hidden_dim * 2, self.num_classes)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Time, Channels)

        Returns:
            logits: Output tensor of shape (Batch, Num_Classes)
        """
        # Project channels: (Batch, Time, 19) -> (Batch, Time, 12)
        x = self.projection(x)

        # Pass through GRU
        # Output x shape: (Batch, Time, Hidden_Dim * 2)
        x, _ = self.gru(x)

        # Apply Attention Pooling
        # Output x shape: (Batch, Hidden_Dim * 2)
        x = self.attention(x)

        # Classification
        # Output logits shape: (Batch, 6)
        logits = self.classifier(x)

        return logits
