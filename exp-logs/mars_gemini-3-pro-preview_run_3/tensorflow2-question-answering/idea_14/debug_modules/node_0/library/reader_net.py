import torch
import torch.nn as nn
import numpy as np
from library import config
from library.layers import HighwayLayer, CoAttention


class HighwayCoAttentionReader(nn.Module):
    """
    Highway Co-Attention Reader Model.

    This model extracts short answer spans from a candidate paragraph given a question.
    It uses Co-Attention to fuse query information into the context, processes the
    sequence through a stack of Highway layers, and predicts start and end token probabilities.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        pretrained_embeddings=None,
        hidden_dim=None,
        num_highway_layers=None,
        dropout_rate=None,
    ):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of word embeddings.
            pretrained_embeddings (np.ndarray, optional): Pre-trained embedding matrix
                                                          to initialize the layer.
            hidden_dim (int, optional): Hidden dimension size for the internal layers.
                                        Defaults to config.HIDDEN_DIM.
            num_highway_layers (int, optional): Number of highway layers to stack.
                                                Defaults to config.HIGHWAY_LAYERS.
            dropout_rate (float, optional): Dropout probability.
                                            Defaults to config.DROPOUT_RATE.
        """
        super(HighwayCoAttentionReader, self).__init__()

        # Set hyperparameters with defaults from config if not provided
        self.hidden_dim = hidden_dim if hidden_dim is not None else config.HIDDEN_DIM
        self.num_highway_layers = (
            num_highway_layers
            if num_highway_layers is not None
            else config.HIGHWAY_LAYERS
        )
        self.dropout_rate = (
            dropout_rate if dropout_rate is not None else config.DROPOUT_RATE
        )

        # 1. Embedding Layer
        # Padding index is assumed to be 0 based on text_utils.text_to_indices
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if pretrained_embeddings is not None:
            # Initialize with pre-trained embeddings
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        # 2. Co-Attention Layer
        # Fuses query info into context. Output dim is 2 * embedding_dim.
        self.co_attention = CoAttention(input_dim=embedding_dim)

        # 3. Projection Layer
        # Projects the fused representation (2 * embed_dim) to the hidden dimension
        # required for the Highway layers.
        self.projection = nn.Sequential(
            nn.Linear(2 * embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )

        # 4. Highway Modeling Layers
        # A stack of Highway layers for deep contextual modeling
        self.highway_layers = nn.ModuleList(
            [
                HighwayLayer(self.hidden_dim, dropout_rate=self.dropout_rate)
                for _ in range(self.num_highway_layers)
            ]
        )

        # 5. Output Layers
        # Independent linear layers for Start and End logits
        self.start_output = nn.Linear(self.hidden_dim, 1)
        self.end_output = nn.Linear(self.hidden_dim, 1)

    def forward(self, query_indices, context_indices):
        """
        Computes the start and end logits for the answer span in the context.

        Args:
            query_indices (torch.Tensor): Tensor of shape (Batch, Q_Len) containing token indices.
            context_indices (torch.Tensor): Tensor of shape (Batch, C_Len) containing token indices.

        Returns:
            tuple: (start_logits, end_logits)
                - start_logits: Tensor of shape (Batch, C_Len)
                - end_logits: Tensor of shape (Batch, C_Len)
        """
        # Generate masks for padding tokens (index 0)
        # Shape: (Batch, Seq_Len)
        query_mask = query_indices != 0
        context_mask = context_indices != 0

        # Look up embeddings
        # Shape: (Batch, Seq_Len, Embedding_Dim)
        q_embed = self.embedding(query_indices)
        c_embed = self.embedding(context_indices)

        # Apply Co-Attention
        # Fuses query information into the context representation.
        # Output Shape: (Batch, C_Len, 2 * Embedding_Dim)
        fused_context = self.co_attention(q_embed, c_embed, query_mask)

        # Project to hidden dimension
        # Output Shape: (Batch, C_Len, Hidden_Dim)
        hidden_repr = self.projection(fused_context)

        # Pass through Highway Layers
        for layer in self.highway_layers:
            hidden_repr = layer(hidden_repr)

        # Compute Start and End Logits
        # Output Shape: (Batch, C_Len, 1)
        start_logits = self.start_output(hidden_repr)
        end_logits = self.end_output(hidden_repr)

        # Squeeze the last dimension to get (Batch, C_Len)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # Mask logits corresponding to padding tokens in the context
        # We set them to a very large negative number so they have ~0 probability after softmax
        if context_mask is not None:
            start_logits = start_logits.masked_fill(context_mask == 0, -1e9)
            end_logits = end_logits.masked_fill(context_mask == 0, -1e9)

        return start_logits, end_logits
