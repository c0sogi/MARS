import torch
import torch.nn as nn
import numpy as np
from library import config
from library.layers import KMaxPooling


class KMaxInteractionRanker(nn.Module):
    """
    K-Max Interaction Ranker Model.

    This model ranks candidate paragraphs based on their relevance to a question.
    It computes a word-to-word interaction matrix, selects the top-K strongest
    interactions for each query term (K-Max Pooling), aggregates them, and
    scores the resulting feature vector using an MLP.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        pretrained_embeddings=None,
        k=None,
        hidden_dim=None,
        dropout_rate=None,
    ):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of word embeddings.
            pretrained_embeddings (np.ndarray, optional): Pre-trained embedding matrix
                                                          to initialize the layer.
            k (int, optional): The number of top interactions to pool.
                               Defaults to config.K_MAX.
            hidden_dim (int, optional): Hidden dimension size for the MLP.
                                        Defaults to config.HIDDEN_DIM.
            dropout_rate (float, optional): Dropout probability.
                                            Defaults to config.DROPOUT_RATE.
        """
        super(KMaxInteractionRanker, self).__init__()

        # Set hyperparameters with defaults from config if not provided
        self.k = k if k is not None else config.K_MAX
        self.hidden_dim = hidden_dim if hidden_dim is not None else config.HIDDEN_DIM
        self.dropout_rate = (
            dropout_rate if dropout_rate is not None else config.DROPOUT_RATE
        )

        # 1. Embedding Layer
        # Padding index is assumed to be 0 based on text_utils.text_to_indices
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if pretrained_embeddings is not None:
            # Initialize with pre-trained embeddings (e.g., GloVe)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        # 2. Interaction & Pooling Layer
        # Extracts the top-k interaction signals
        self.kmax_pooling = KMaxPooling(k=self.k)

        # 3. Scoring MLP
        # The input to the MLP is the pooled feature vector of size K
        self.mlp = nn.Sequential(
            nn.Linear(self.k, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, query_indices, candidate_indices):
        """
        Computes the relevance score for a (query, candidate) pair.

        Args:
            query_indices (torch.Tensor): Tensor of shape (Batch, Q_Len) containing token indices.
            candidate_indices (torch.Tensor): Tensor of shape (Batch, C_Len) containing token indices.

        Returns:
            torch.Tensor: Relevance scores of shape (Batch,).
        """
        # Generate masks for padding tokens (index 0)
        # Shape: (Batch, Seq_Len)
        query_mask = query_indices != 0
        candidate_mask = candidate_indices != 0

        # Look up embeddings
        # Shape: (Batch, Seq_Len, Embedding_Dim)
        q_embed = self.embedding(query_indices)
        c_embed = self.embedding(candidate_indices)

        # Apply K-Max Interaction Pooling
        # This computes the interaction matrix, selects top-k values per query token,
        # and sum-pools them to get a global matching profile.
        # Output Shape: (Batch, K)
        pooled_features = self.kmax_pooling(
            q_embed, c_embed, query_mask, candidate_mask
        )

        # Pass through MLP to get a scalar score
        # Output Shape: (Batch, 1)
        scores = self.mlp(pooled_features)

        # Squeeze to return shape (Batch,)
        return scores.squeeze(-1)
