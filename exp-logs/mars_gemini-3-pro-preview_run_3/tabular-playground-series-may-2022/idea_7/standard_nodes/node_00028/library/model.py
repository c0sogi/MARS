import torch
import torch.nn as nn
from library.config import Config


class FunnelMLP(nn.Module):
    """
    Standard Funnel MLP Architecture with Entity Embeddings.

    Structure:
    1. Embeddings: Lookup for categorical features.
    2. Concatenation: Flattened embeddings + Continuous features.
    3. Backbone: MLP with decreasing layer widths (Funnel) using ReLU and Dropout.
    4. Head: Linear projection to output logits.
    """

    def __init__(self, vocab_sizes):
        super().__init__()

        # Hyperparameters from Config
        embedding_dim = Config.EMBEDDING_DIM
        dropout_rate = Config.DROPOUT_RATE
        hidden_layers = Config.HIDDEN_LAYERS
        num_cont_features = Config.NUM_CONT_FEATURES

        # Create embedding layers
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=size, embedding_dim=embedding_dim)
                for size in vocab_sizes
            ]
        )

        # Calculate input dimension
        self.cat_flatten_dim = len(vocab_sizes) * embedding_dim
        current_dim = self.cat_flatten_dim + num_cont_features

        # Build Funnel MLP
        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        self.funnel = nn.Sequential(*layers)
        self.head = nn.Linear(current_dim, Config.OUTPUT_DIM)

    def forward(self, x_cat, x_cont):
        # Embeddings
        emb_list = []
        for i, emb_layer in enumerate(self.embeddings):
            emb_list.append(emb_layer(x_cat[:, i]))

        cat_emb = torch.cat(emb_list, dim=1)

        # Concatenate with continuous
        x = torch.cat([cat_emb, x_cont], dim=1)

        # Forward pass
        features = self.funnel(x)
        logits = self.head(features)

        return logits
