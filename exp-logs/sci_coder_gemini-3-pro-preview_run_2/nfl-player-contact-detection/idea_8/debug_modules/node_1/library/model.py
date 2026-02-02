import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config, utils


class KCAN(nn.Module):
    """
    Kinematic Center-Attention Network (K-CAN).

    A Time-Resolved Attention Network that replaces flat dense layers with a
    dynamic aggregation mechanism centered on the target frame. It strictly
    enforces 'Center Focus' via a Center-Query Attention mechanism and preserves
    instantaneous physics via an explicit skip connection.
    """

    def __init__(self):
        super(KCAN, self).__init__()

        # Ensure reproducibility
        utils.set_seed()

        # Hyperparameters
        self.input_dim = len(config.INPUT_FEATURES)
        self.hidden_dim = config.HIDDEN_DIM
        self.num_heads = config.NUM_HEADS
        self.dropout_rate = config.DROPOUT
        self.window_size = config.WINDOW_SIZE

        # 1. Feature Embedding (Time-Distributed MLP)
        # Projects features of each time step into a higher-dimensional latent space.
        # Input: (Batch, Window, Input_Features) -> Output: (Batch, Window, Hidden_Dim)
        self.embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )

        # 2. Center-Query Attention (Structural Innovation)
        # Query: Derived exclusively from center frame embedding.
        # Keys/Values: Derived from the entire window sequence.
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True,
        )

        # 3. Unified Classification Head
        # Concatenates Attention Output (Context) with Raw Center Features (Skip Connection).
        # Input: Hidden_Dim + Input_Dim
        classifier_input_dim = self.hidden_dim + self.input_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, 1),  # Single logit output
        )

    def forward(self, x):
        """
        Args:
            x (tuple):
                - sequence (Tensor): Shape (Batch, Window_Size, Input_Dim)
                - center_features (Tensor): Shape (Batch, Input_Dim) - Raw features at t=0

        Returns:
            logits (Tensor): Shape (Batch, 1)
        """
        sequence, center_features = x

        # 1. Embed the sequence
        # Shape: (Batch, Window, Hidden_Dim)
        seq_embed = self.embedding(sequence)

        # 2. Extract Center Embedding for Query
        # The center index is the middle of the window
        center_idx = self.window_size // 2

        # Shape: (Batch, Hidden_Dim) -> (Batch, 1, Hidden_Dim) for Attention
        center_query = seq_embed[:, center_idx, :].unsqueeze(1)

        # 3. Apply Multi-Head Attention
        # Query: Center frame state
        # Key/Value: Full temporal context
        # attn_out Shape: (Batch, 1, Hidden_Dim)
        attn_out, _ = self.attention(query=center_query, key=seq_embed, value=seq_embed)

        # Flatten attention output: (Batch, Hidden_Dim)
        attn_out = attn_out.squeeze(1)

        # 4. Explicit Skip Connection
        # Concatenate context with raw explicit kinematics of the center frame
        # Shape: (Batch, Hidden_Dim + Input_Dim)
        combined = torch.cat([attn_out, center_features], dim=1)

        # 5. Classification
        logits = self.classifier(combined)

        return logits
