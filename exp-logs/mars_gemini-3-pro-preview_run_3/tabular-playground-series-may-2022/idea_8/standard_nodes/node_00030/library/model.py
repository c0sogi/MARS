import torch
import torch.nn as nn
from library.config import EMBED_DIM, BACKBONE_LAYERS, DROPOUT, OUTPUT_DIM


class ManufacturingMLP(nn.Module):
    """
    Standard Funnel MLP Architecture with Early Fusion.

    Concatenates flattened categorical embeddings and continuous features directly
    at the input level before passing them through a funnel-shaped MLP backbone.
    Cite {solution_lesson_node_00029}
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
        """
        Args:
            vocab_sizes (dict): Dictionary mapping categorical column names to vocabulary sizes.
            cont_dim (int): Number of continuous input features.
            embed_dim (int): Dimension of entity embeddings.
            backbone_layers (list): List of hidden layer dimensions for the funnel backbone.
            dropout (float): Dropout probability.
            output_dim (int): Dimension of the output (1 for binary classification).
        """
        super(ManufacturingMLP, self).__init__()

        self.vocab_sizes = vocab_sizes
        self.keys = list(
            vocab_sizes.keys()
        )  # Ensure deterministic order matching data loader

        # 1. Embeddings
        self.embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(num_embeddings=size, embedding_dim=embed_dim)
                for col, size in vocab_sizes.items()
            }
        )

        # Calculate input dimension for the backbone
        # (Num_Cats * Embed_Dim) + Num_Cont
        total_embed_dim = len(vocab_sizes) * embed_dim
        input_dim = total_embed_dim + cont_dim

        # 2. Funnel Backbone
        layers = []
        current_dim = input_dim

        for hidden_dim in backbone_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # 3. Output Head
        self.head = nn.Linear(current_dim, output_dim)

    def forward(self, cat_x, cont_x):
        """
        Args:
            cat_x (torch.Tensor): Categorical indices of shape (Batch, Num_Cat_Features).
            cont_x (torch.Tensor): Continuous values of shape (Batch, Num_Cont_Features).

        Returns:
            torch.Tensor: Logits of shape (Batch, Output_Dim).
        """
        # --- Embeddings ---
        # Retrieve embeddings for each column
        embed_list = []
        for i, key in enumerate(self.keys):
            val = cat_x[:, i]
            emb = self.embeddings[key](val)
            embed_list.append(emb)

        # Concatenate all embeddings: (Batch, Num_Cats * Embed_Dim)
        cat_flat = torch.cat(embed_list, dim=1)

        # --- Early Fusion ---
        # Concatenate embeddings and continuous features
        fused = torch.cat([cat_flat, cont_x], dim=1)

        # --- Backbone & Output ---
        features = self.backbone(fused)
        logits = self.head(features)

        return logits
