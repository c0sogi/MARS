import torch
import torch.nn as nn
import torch.nn.functional as F


class FastTextClassifier(nn.Module):
    """
    A FastText-style architecture for efficient text classification.

    This model represents a document as the average of its word embeddings,
    which is then fed into a linear classifier. It is computationally efficient
    and effective for high-dimensional sparse data like text.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int,
        dropout: float = 0.0,
    ):
        """
        Initializes the FastTextClassifier.

        Args:
            vocab_size (int): The size of the vocabulary (number of unique tokens).
            num_classes (int): The number of target labels (tags).
            embedding_dim (int): The dimension of the dense embedding vectors.
            dropout (float): The dropout probability for regularization.
        """
        super(FastTextClassifier, self).__init__()

        # EmbeddingBag computes the mean of embeddings for a sequence of indices.
        # It is much faster than embedding + mean/sum operations.
        # mode='mean' ensures the representation is invariant to sequence length.
        self.embedding = nn.EmbeddingBag(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, mode="mean"
        )

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(p=dropout)

        # Linear layer mapping the document embedding to class logits
        self.fc = nn.Linear(embedding_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights to small uniform values to aid convergence.
        """
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            text (torch.Tensor): 1D tensor containing the concatenated indices of tokens
                                 across the batch.
            offsets (torch.Tensor): 1D tensor containing the starting index of each
                                    sample in the 'text' tensor.

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        # 1. Compute averaged embeddings
        # Input: (total_tokens,), (batch_size,)
        # Output: (batch_size, embedding_dim)
        embedded = self.embedding(text, offsets)

        # 2. Apply dropout
        embedded = self.dropout(embedded)

        # 3. Project to output space
        # Output: (batch_size, num_classes)
        logits = self.fc(embedded)

        return logits
