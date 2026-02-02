import torch
import torch.nn as nn
from library.config import Config


class FunnelMLP(nn.Module):
    """
    Standard Funnel MLP architecture with Entity Embeddings.
    Removed Gaussian Noise Injection based on Lesson 00053.
    """

    def __init__(
        self,
        vocab_sizes,
        num_cont_features,
        embed_dim=Config.EMBEDDING_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        super().__init__()

        # 1. Entity Embeddings for Categorical Features
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=v, embedding_dim=embed_dim)
                for v in vocab_sizes
            ]
        )

        # Calculate the total dimension
        cat_input_dim = len(vocab_sizes) * embed_dim
        total_input_dim = cat_input_dim + num_cont_features

        # 2. Funnel MLP Backbone
        layers = []
        in_dim = total_input_dim

        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        self.mlp = nn.Sequential(*layers)

        # 3. Output Head
        self.head = nn.Linear(in_dim, 1)

    def forward(self, cat_x, cont_x):
        # Process Embeddings
        embedded_list = []
        for i, emb_layer in enumerate(self.embeddings):
            val = cat_x[:, i]
            embedded_list.append(emb_layer(val))

        cat_features = torch.cat(embedded_list, dim=1)
        x = torch.cat([cat_features, cont_x], dim=1)

        # Pass through Funnel MLP
        x = self.mlp(x)

        # Output Logits
        x = self.head(x)

        return x
