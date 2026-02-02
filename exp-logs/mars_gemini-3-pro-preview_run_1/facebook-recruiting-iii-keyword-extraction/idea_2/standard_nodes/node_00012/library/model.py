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
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        """
        Initializes the FastTextClassifier (Deep Averaging Network).

        Args:
            vocab_size (int): The size of the vocabulary (number of unique tokens).
            num_classes (int): The number of target labels (tags).
            embedding_dim (int): The dimension of the dense embedding vectors.
            hidden_dim (int): The dimension of the hidden layer.
            dropout (float): The dropout probability for regularization.
        """
        super(FastTextClassifier, self).__init__()

        # EmbeddingBag computes the mean of embeddings for a sequence of indices.
        # mode='mean' ensures the representation is invariant to sequence length.
        self.embedding = nn.EmbeddingBag(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, mode="mean"
        )

        # Cite solution_lesson_node_00005: Deep Averaging Network (DAN) structure
        # Cite solution_lesson_node_00011: Separate embedding streams (Title + Body)
        # We concatenate two embeddings, so input to hidden is embedding_dim * 2
        self.hidden = nn.Linear(embedding_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights to small uniform values to aid convergence.
        """
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.hidden.weight.data.uniform_(-initrange, initrange)
        self.hidden.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(
        self,
        title_text: torch.Tensor,
        title_offsets: torch.Tensor,
        body_text: torch.Tensor,
        body_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with separate embeddings for Title and Body.
        """
        # 1. Compute averaged embeddings separately
        e_title = self.embedding(title_text, title_offsets)
        e_body = self.embedding(body_text, body_offsets)

        # 2. Concatenate embeddings (Late Fusion)
        embedded = torch.cat([e_title, e_body], dim=1)

        # 3. Hidden Layer (Non-linearity)
        h = self.hidden(embedded)
        h = self.relu(h)
        h = self.dropout(h)

        # 4. Project to output space
        logits = self.fc(h)

        return logits
