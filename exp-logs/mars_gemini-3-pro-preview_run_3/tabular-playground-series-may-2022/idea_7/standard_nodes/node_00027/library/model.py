import torch
import torch.nn as nn
from library.config import Config


class ManufacturingMLP(nn.Module):
    """
    Standard Funnel MLP Architecture with Entity Embeddings.
    Cite solution_lesson_node_00004: Funnel MLP preferred over ResNets.
    Cite solution_lesson_node_00019: Standard MLP preferred over Gated/GLU.

    Structure:
    1. Embeddings for categorical features.
    2. Concatenation with continuous features.
    3. Funnel MLP (Linear -> ReLU -> Dropout).
    """

    def __init__(self, vocab_sizes):
        super().__init__()

        # Hyperparameters from Config
        embedding_dim = Config.EMBEDDING_DIM
        dropout_rate = Config.DROPOUT_RATE
        hidden_layers = Config.HIDDEN_LAYERS
        num_cont_features = Config.NUM_CONT_FEATURES

        # Embeddings
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=size, embedding_dim=embedding_dim)
                for size in vocab_sizes
            ]
        )

        # Calculate input dimension
        cat_dim = len(vocab_sizes) * embedding_dim
        input_dim = cat_dim + num_cont_features

        # Build MLP
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output Head
        self.head = nn.Linear(current_dim, Config.OUTPUT_DIM)

    def forward(self, x_cat, x_cont):
        # Embeddings
        emb_list = []
        for i, emb_layer in enumerate(self.embeddings):
            emb_list.append(emb_layer(x_cat[:, i]))

        cat_emb = torch.cat(emb_list, dim=1)

        # Concatenate
        x = torch.cat([cat_emb, x_cont], dim=1)

        # Forward pass
        x = self.mlp(x)
        logits = self.head(x)

        return logits
