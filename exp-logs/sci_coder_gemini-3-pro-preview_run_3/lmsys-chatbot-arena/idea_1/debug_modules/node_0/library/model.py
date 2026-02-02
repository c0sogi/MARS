import torch
import torch.nn as nn
from library.config import Config


class DeepAveragingNetwork(nn.Module):
    """
    Deep Averaging Network (DAN) for Chatbot Arena preference prediction.

    Architecture:
    1. Shared Embedding Layer: Maps token indices to dense vectors.
    2. Global Average Pooling: Averages word embeddings for Prompt, Response A, and Response B.
    3. Feature Concatenation: Combines [Prompt, ResA, ResB, ResA - ResB].
    4. MLP Classifier: Projects features to class logits (Winner A, Winner B, Tie).
    """

    def __init__(self, config: Config):
        super(DeepAveragingNetwork, self).__init__()

        self.embedding_dim = config.EMBEDDING_DIM
        self.vocab_size = config.VOCAB_SIZE
        self.hidden_dim = config.HIDDEN_DIM
        self.dropout_prob = config.DROPOUT
        self.num_classes = config.NUM_CLASSES

        # Shared Embedding Layer
        # padding_idx=0 ensures the padding token vector is initialized to zero
        # and ignored during backprop if gradients are sparse, though we also mask it manually.
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0,
        )

        # MLP Layers
        # Input dimension: Prompt + Response A + Response B + (Response A - Response B)
        # Each is a vector of size embedding_dim, so total is 4 * embedding_dim
        input_dim = self.embedding_dim * 4

        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, prompt, response_a, response_b):
        """
        Forward pass of the DAN model.

        Args:
            prompt (torch.Tensor): Tensor of shape (batch_size, seq_len) containing token indices.
            response_a (torch.Tensor): Tensor of shape (batch_size, seq_len) containing token indices.
            response_b (torch.Tensor): Tensor of shape (batch_size, seq_len) containing token indices.

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """

        # Compute average embeddings for each input sequence
        # Vectors shape: (batch_size, embedding_dim)
        v_p = self._global_average_pooling(prompt)
        v_a = self._global_average_pooling(response_a)
        v_b = self._global_average_pooling(response_b)

        # Compute difference vector to explicitly capture comparison features
        v_diff = v_a - v_b

        # Concatenate features
        # Shape: (batch_size, embedding_dim * 4)
        features = torch.cat([v_p, v_a, v_b, v_diff], dim=1)

        # MLP Pass
        x = self.fc1(features)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits

    def _global_average_pooling(self, x):
        """
        Computes the unweighted average of word embeddings, ignoring padding tokens.

        Args:
            x (torch.Tensor): Input tensor of token indices (batch_size, seq_len).

        Returns:
            torch.Tensor: Averaged embedding vector (batch_size, embedding_dim).
        """
        # Get embeddings: (batch_size, seq_len, embedding_dim)
        embeds = self.embedding(x)

        # Create mask for non-padding tokens (padding_idx is 0)
        # Shape: (batch_size, seq_len)
        mask = (x != 0).float()

        # Expand mask to match embedding dimensions for broadcasting
        # Shape: (batch_size, seq_len, 1) -> broadcasts to (batch_size, seq_len, embedding_dim)
        mask = mask.unsqueeze(-1)

        # Sum masked embeddings
        # Shape: (batch_size, embedding_dim)
        sum_embeds = torch.sum(embeds * mask, dim=1)

        # Count non-padding tokens
        # Shape: (batch_size, 1)
        # Clamp min=1 to avoid division by zero for empty sequences (though data processing should prevent this)
        counts = torch.sum(mask, dim=1).clamp(min=1)

        # Compute average
        avg_embeds = sum_embeds / counts

        return avg_embeds
