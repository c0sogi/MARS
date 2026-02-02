import torch
import torch.nn as nn
from library.config import Config


class NBOWModel(nn.Module):
    """
    Neural Bag-of-Words (NBOW) model for multi-label text classification.

    Architecture:
    1. EmbeddingBag (mode='mean'): Maps token IDs to dense vectors and averages them.
    2. Dropout: Regularization.
    3. Linear: Maps context vector to tag logits.
    """

    def __init__(
        self,
        vocab_size=Config.VOCAB_SIZE,
        embed_dim=Config.EMBED_DIM,
        num_classes=Config.NUM_TAGS,
        dropout=Config.DROPOUT,
        padding_idx=1,  # Matches pad_index=1 in data_processing.py Vocabulary
    ):
        super(NBOWModel, self).__init__()

        # EmbeddingBag is efficient for variable length sequences.
        # It performs the lookup and the reduction (mean) in a fused kernel.
        self.embedding = nn.EmbeddingBag(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            mode="mean",
            padding_idx=padding_idx,
        )

        self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Linear(in_features=embed_dim, out_features=num_classes)

    def forward(self, input, offsets):
        """
        Forward pass.

        Args:
            input (torch.Tensor): 1D tensor of concatenated token indices.
            offsets (torch.Tensor): 1D tensor of starting indices for each sequence.

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        # x shape: (batch_size, embed_dim)
        x = self.embedding(input, offsets)

        x = self.dropout(x)

        # logits shape: (batch_size, num_classes)
        logits = self.fc(x)

        return logits
