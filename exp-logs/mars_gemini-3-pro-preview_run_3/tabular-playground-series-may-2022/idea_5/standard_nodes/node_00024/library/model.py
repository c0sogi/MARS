import torch
import torch.nn as nn
import torch.nn.functional as F


class FunnelMLP(nn.Module):
    """
    Standard Funnel MLP.
    Cite solution_lesson_node_00004: Simplicity Over Complexity.
    Structure: [Linear -> ReLU -> Dropout] x N -> Linear.
    """

    def __init__(
        self, num_cont, cat_cardinalities, embedding_dim, hidden_layers, dropout
    ):
        super().__init__()

        # Entity Embeddings for categorical features
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, embedding_dim) for card in cat_cardinalities]
        )

        # Calculate input dimension: Continuous features + Flattened embeddings
        self.input_dim = num_cont + len(cat_cardinalities) * embedding_dim

        # Build the backbone (Funnel structure)
        layers = []
        in_dim = self.input_dim

        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # Output Head
        self.head = nn.Linear(hidden_layers[-1], 1)

    def forward(self, x_cont, x_cat):
        # Process Embeddings
        emb_list = []
        for i, emb_layer in enumerate(self.embeddings):
            emb_list.append(emb_layer(x_cat[:, i]))

        # Concatenate all embeddings
        x_emb = torch.cat(emb_list, dim=1)

        # Concatenate continuous features with embeddings
        x = torch.cat([x_cont, x_emb], dim=1)

        # Pass through Backbone
        x = self.backbone(x)

        # Final classification head
        return self.head(x)
