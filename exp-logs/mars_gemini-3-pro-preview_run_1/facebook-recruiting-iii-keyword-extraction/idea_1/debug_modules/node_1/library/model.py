import torch
import torch.nn as nn
from library.config import Config


class BiGRUClassifier(nn.Module):
    """
    Bi-Directional GRU with Global Max Pooling for Text Classification.

    Architecture:
    1. Embedding Layer: Converts integer tokens to dense vectors.
    2. Bi-Directional GRU: Captures contextual information from both directions.
    3. Global Max Pooling: Extracts the most salient features across the time dimension.
    4. Dropout: Regularization.
    5. Linear Layer: Maps features to output class logits.
    """

    def __init__(
        self,
        vocab_size=Config.VOCAB_SIZE,
        embed_dim=Config.EMBED_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        num_classes=Config.NUM_TAGS,
        dropout=Config.DROPOUT,
        padding_idx=0,
    ):
        """
        Initialize the BiGRUClassifier.

        Args:
            vocab_size (int): Size of the vocabulary (number of unique tokens).
            embed_dim (int): Dimension of the word embeddings.
            hidden_dim (int): Dimension of the GRU hidden state.
            num_layers (int): Number of stacked GRU layers.
            num_classes (int): Number of output classes (tags).
            dropout (float): Dropout probability.
            padding_idx (int): Index used for padding in the embedding layer.
        """
        super(BiGRUClassifier, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=padding_idx
        )

        # Bi-Directional GRU
        # Note: Dropout is only applied between layers if num_layers > 1
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Dropout layer before the final classifier
        self.dropout = nn.Dropout(p=dropout)

        # Final Classification Layer
        # The input dimension is hidden_dim * 2 because the GRU is bidirectional
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len) containing token IDs.

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # 1. Embedding
        # Output shape: (batch_size, seq_len, embed_dim)
        x = self.embedding(x)

        # 2. Bi-Directional GRU
        # output shape: (batch_size, seq_len, hidden_dim * 2)
        # h_n is ignored
        x, _ = self.gru(x)

        # 3. Global Max Pooling
        # We take the maximum value over the sequence dimension (dim=1)
        # This captures the strongest signal for each feature across the entire text.
        # Output shape: (batch_size, hidden_dim * 2)
        x, _ = torch.max(x, dim=1)

        # 4. Dropout
        x = self.dropout(x)

        # 5. Classifier
        # Output shape: (batch_size, num_classes)
        logits = self.fc(x)

        return logits
