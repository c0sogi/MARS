import torch
import torch.nn as nn
from library.config import Config


class DANRegressor(nn.Module):
    """
    Deep Averaging Network (DAN) for regression tasks.

    Architecture:
    1. Embedding Layer: Maps token IDs to dense vectors.
    2. Global Average Pooling: Averages word embeddings across the sequence, ignoring padding.
    3. MLP Head: Projects the averaged embedding to a scalar score.
    """

    def __init__(self, config=Config):
        """
        Args:
            config: Configuration class containing hyperparameters.
        """
        super(DANRegressor, self).__init__()

        self.vocab_size = config.VOCAB_SIZE
        self.embed_dim = config.EMBED_DIM
        self.hidden_dim = config.HIDDEN_DIM
        self.output_dim = config.OUTPUT_DIM
        self.dropout_prob = config.DROPOUT_PROB

        # Padding index is 0 based on the Tokenizer in library/data.py
        self.padding_idx = 0

        # 1. Embedding Layer
        # padding_idx=0 ensures the vector for the padding token is always zero,
        # which helps in the summation part of the average pooling.
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_dim,
            padding_idx=self.padding_idx,
        )

        # 3. MLP Head (Regressor)
        self.regressor = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

        # Initialize weights (optional but good practice)
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) containing token IDs.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, 1) containing predicted scores.
        """
        # input_ids: (batch_size, seq_len)

        # 1. Embed tokens
        # embedded: (batch_size, seq_len, embed_dim)
        embedded = self.embedding(input_ids)

        # 2. Global Average Pooling (Masked)
        # We need to average only non-padding tokens.

        # Create a mask: 1 for valid tokens, 0 for padding tokens
        # mask: (batch_size, seq_len, 1)
        mask = (input_ids != self.padding_idx).float().unsqueeze(-1)

        # Sum embeddings along the sequence dimension (dim=1)
        # Since padding embeddings are 0 (via padding_idx in nn.Embedding) and we multiply by mask,
        # this safely sums only valid token embeddings.
        # sum_embeddings: (batch_size, embed_dim)
        sum_embeddings = torch.sum(embedded * mask, dim=1)

        # Count the number of valid tokens in each sequence
        # token_counts: (batch_size, 1)
        token_counts = torch.sum(mask, dim=1)

        # Avoid division by zero for empty sequences (though unlikely with proper data cleaning)
        token_counts = torch.clamp(token_counts, min=1e-9)

        # Compute the mean
        # mean_embeddings: (batch_size, embed_dim)
        mean_embeddings = sum_embeddings / token_counts

        # 3. Pass through MLP Regressor
        # output: (batch_size, output_dim)
        output = self.regressor(mean_embeddings)

        return output
