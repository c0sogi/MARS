import torch
import torch.nn as nn
from library.config import EMBED_DIM, BACKBONE_LAYERS, DROPOUT, OUTPUT_DIM


class ManufacturingMLP(nn.Module):
    """
    Standard Funnel MLP with Early Fusion.

    Concatenates Entity Embeddings and Continuous features directly at the input level.
    Cite solution_lesson_node_00029: Prefer Early Fusion over separated streams.
    Cite solution_lesson_node_00004: Prefer Funnel MLP over ResNets.
    """

    def __init__(
        self,
        vocab_sizes,
        cont_dim,
        embed_dim=EMBED_DIM,
        backbone_layers=BACKBONE_LAYERS,
        dropout=DROPOUT,
        output_dim=OUTPUT_DIM,
    ):
        super(ManufacturingMLP, self).__init__()

        self.vocab_sizes = vocab_sizes
        self.keys = list(vocab_sizes.keys())  # Ensure deterministic order

        # -------------------------------------------------------
        # 1. Embeddings
        # -------------------------------------------------------
        self.embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(num_embeddings=size, embedding_dim=embed_dim)
                for col, size in vocab_sizes.items()
            }
        )

        # -------------------------------------------------------
        # 2. Funnel Backbone
        # -------------------------------------------------------
        # Input dimension is sum of flattened embeddings and continuous features
        total_embed_dim = len(vocab_sizes) * embed_dim
        input_dim = total_embed_dim + cont_dim

        layers = []
        current_dim = input_dim

        for hidden_dim in backbone_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # -------------------------------------------------------
        # 3. Output Head
        # -------------------------------------------------------
        self.head = nn.Linear(current_dim, output_dim)

    def forward(self, cat_x, cont_x):
        # Retrieve and concatenate embeddings
        embed_list = []
        for i, key in enumerate(self.keys):
            val = cat_x[:, i]
            emb = self.embeddings[key](val)
            embed_list.append(emb)

        cat_flat = torch.cat(embed_list, dim=1)

        # Early Fusion: Concatenate embeddings and continuous features
        x = torch.cat([cat_flat, cont_x], dim=1)

        # Pass through backbone
        features = self.backbone(x)
        logits = self.head(features)

        return logits
