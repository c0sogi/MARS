import torch
import torch.nn as nn
from library.config import EMBED_DIM, BACKBONE_LAYERS, DROPOUT, OUTPUT_DIM


class DualStreamFunnelMLP(nn.Module):
    """
    Dual-Stream Funnel MLP Architecture.

    Separates categorical (State) and continuous (Sensor) features into two distinct
    processing streams before fusing them into a funnel-shaped MLP backbone.
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
        super(DualStreamFunnelMLP, self).__init__()

        self.vocab_sizes = vocab_sizes
        self.keys = list(
            vocab_sizes.keys()
        )  # Ensure deterministic order matching data loader

        # -------------------------------------------------------
        # 1. State Stream (Categorical)
        # -------------------------------------------------------
        # Create embeddings for each categorical feature
        self.embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(num_embeddings=size, embedding_dim=embed_dim)
                for col, size in vocab_sizes.items()
            }
        )

        # Calculate total flattened embedding size
        total_embed_dim = len(vocab_sizes) * embed_dim

        # Determine projection dimension (half of the first backbone layer width)
        # This ensures the fused vector matches the input of the backbone
        self.proj_dim = backbone_layers[0] // 2

        # Projection layer for State Stream
        self.state_projection = nn.Sequential(
            nn.Linear(total_embed_dim, self.proj_dim), nn.ReLU()
        )

        # -------------------------------------------------------
        # 2. Sensor Stream (Continuous)
        # -------------------------------------------------------
        # Projection layer for Sensor Stream
        self.sensor_projection = nn.Sequential(
            nn.Linear(cont_dim, self.proj_dim), nn.ReLU()
        )

        # -------------------------------------------------------
        # 3. Funnel Backbone
        # -------------------------------------------------------
        # Input to backbone is the concatenation of state and sensor projections
        input_dim = self.proj_dim * 2

        layers = []
        current_dim = input_dim

        for hidden_dim in backbone_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # -------------------------------------------------------
        # 4. Output Head
        # -------------------------------------------------------
        self.head = nn.Linear(current_dim, output_dim)

    def forward(self, cat_x, cont_x):
        """
        Args:
            cat_x (torch.Tensor): Categorical indices of shape (Batch, Num_Cat_Features).
            cont_x (torch.Tensor): Continuous values of shape (Batch, Num_Cont_Features).

        Returns:
            torch.Tensor: Logits of shape (Batch, Output_Dim).
        """
        # --- Process State Stream ---
        # Retrieve embeddings for each column and concatenate
        # cat_x columns correspond to self.keys order
        embed_list = []
        for i, key in enumerate(self.keys):
            # Extract column i, lookup embedding
            val = cat_x[:, i]
            emb = self.embeddings[key](val)
            embed_list.append(emb)

        # Concatenate all embeddings: (Batch, Num_Cats * Embed_Dim)
        cat_flat = torch.cat(embed_list, dim=1)

        # Project to latent State space
        state_latent = self.state_projection(cat_flat)

        # --- Process Sensor Stream ---
        # Project to latent Sensor space
        sensor_latent = self.sensor_projection(cont_x)

        # --- Fusion ---
        # Concatenate state and sensor representations
        fused = torch.cat([state_latent, sensor_latent], dim=1)

        # --- Backbone & Output ---
        features = self.backbone(fused)
        logits = self.head(features)

        return logits
