import torch
import torch.nn as nn
from library.config import Config


class DualBranchDAN(nn.Module):
    """
    Dual-Branch Deep Averaging Network (DAN) for Question-Answer pair classification.

    This model processes the Question and Answer text independently using a shared
    embedding layer and global average pooling. The resulting representations are
    concatenated and passed through an MLP to predict multiple target probabilities.
    """

    def __init__(
        self,
        vocab_size=Config.VOCAB_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT,
        num_targets=len(Config.TARGET_COLS),
    ):
        """
        Initializes the DualBranchDAN model.

        Args:
            vocab_size (int): Size of the vocabulary. Defaults to Config.VOCAB_SIZE.
            embedding_dim (int): Dimension of the embedding vectors. Defaults to Config.EMBEDDING_DIM.
            hidden_dim (int): Dimension of the hidden layer in the MLP. Defaults to Config.HIDDEN_DIM.
            dropout (float): Dropout probability. Defaults to Config.DROPOUT.
            num_targets (int): Number of target labels to predict. Defaults to len(Config.TARGET_COLS).
        """
        super(DualBranchDAN, self).__init__()

        # Shared Embedding Layer
        # padding_idx=0 ensures that the padding token (index 0) maps to a zero vector
        # and does not accumulate gradients.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )

        # MLP Head
        # The input dimension is 2 * embedding_dim because we concatenate the
        # averaged embeddings of the Question and the Answer.
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_targets)
        self.sigmoid = nn.Sigmoid()

    def forward(self, q_seq, a_seq):
        """
        Forward pass of the model.

        Args:
            q_seq (torch.Tensor): Tensor containing Question sequences.
                                  Shape: (batch_size, max_len_q)
            a_seq (torch.Tensor): Tensor containing Answer sequences.
                                  Shape: (batch_size, max_len_a)

        Returns:
            torch.Tensor: Predicted probabilities for the targets.
                          Shape: (batch_size, num_targets)
        """
        # 1. Embedding
        # Embed the input sequences.
        # Output Shape: (batch_size, sequence_length, embedding_dim)
        q_emb = self.embedding(q_seq)
        a_emb = self.embedding(a_seq)

        # 2. Global Average Pooling
        # Compute the mean of the embeddings across the sequence dimension (dim=1).
        # Since padding_idx=0, pad tokens are zero vectors.
        # Output Shape: (batch_size, embedding_dim)
        q_avg = torch.mean(q_emb, dim=1)
        a_avg = torch.mean(a_emb, dim=1)

        # 3. Concatenation
        # Concatenate the pooled Question and Answer representations.
        # Output Shape: (batch_size, 2 * embedding_dim)
        combined = torch.cat([q_avg, a_avg], dim=1)

        # 4. MLP Head
        # Pass through the fully connected layers with non-linearity and dropout.
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # 5. Output Activation
        # Apply Sigmoid to ensure predictions are in the range [0, 1].
        out = self.sigmoid(x)

        return out
