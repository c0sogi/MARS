import torch
import torch.nn as nn
from library.config import Config


class ManufacturingMLP(nn.Module):
    """
    Standard Funnel MLP with Entity Embeddings.

    Architecture:
    1. Inputs: Categorical Entity Embeddings + Normalized Continuous Features.
    2. Funnel MLP: 512 -> 256 -> 128 -> 1

    This simple architecture is preferred over complex ensembles for this high-signal tabular task.
    Cite solution_lesson_node_00047, solution_lesson_node_00004.
    """

    def __init__(self, vocab_sizes, cont_dim):
        super(ManufacturingMLP, self).__init__()

        # Hyperparameters from Config
        embed_dim = Config.EMBED_DIM
        hidden_dims = Config.HIDDEN_DIMS
        dropout_rate = Config.DROPOUT_RATE

        # 1. Entity Embeddings
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=size, embedding_dim=embed_dim)
                for size in vocab_sizes
            ]
        )

        # Calculate total input dimension
        cat_input_dim = len(vocab_sizes) * embed_dim
        input_dim = cat_input_dim + cont_dim

        # 2. Funnel MLP
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = dim

        # Final Output
        layers.append(nn.Linear(input_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, cat_x, cont_x):
        # Process Embeddings
        embeds = []
        for i, emb_layer in enumerate(self.embeddings):
            embeds.append(emb_layer(cat_x[:, i]))

        cat_features = torch.cat(embeds, dim=1)

        # Combine Features
        x = torch.cat([cat_features, cont_x], dim=1)

        # Pass through MLP
        return self.mlp(x)
