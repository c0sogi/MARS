import torch
import torch.nn as nn
from library.config import Config


class GatedBlock(nn.Module):
    """
    A Gated Linear Unit block consisting of:
    LayerNorm -> Linear (projection to 2x width) -> GLU -> Dropout.

    This block serves as the fundamental building unit for the Dual-Stream network,
    providing gated feature selection and non-linear processing with stable gradients.
    """

    def __init__(self, in_features, out_features, dropout_rate):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        # GLU requires the input to be split in half, so we project to 2 * out_features
        self.linear = nn.Linear(in_features, out_features * 2)
        self.glu = nn.GLU(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear(x)
        x = self.glu(x)
        x = self.dropout(x)
        return x


class DualStreamGatedNetwork(nn.Module):
    """
    Dual-Stream Gated Funnel Network Architecture.

    Structure:
    1. Categorical Stream: Embeddings -> Flatten -> Gated Feature Selection
    2. Continuous Stream: Raw Features -> Gated Noise Filtering
    3. Fusion: Concatenation of both streams
    4. Backbone: Funnel MLP (decreasing widths) using GatedBlocks
    5. Head: Linear projection to output logits
    """

    def __init__(self, vocab_sizes):
        """
        Args:
            vocab_sizes (list[int]): A list of integers representing the vocabulary size
                                     for each categorical feature. Order must match Config.CAT_FEATURES.
        """
        super().__init__()

        # Hyperparameters from Config
        embedding_dim = Config.EMBEDDING_DIM
        dropout_rate = Config.DROPOUT_RATE
        hidden_layers = Config.HIDDEN_LAYERS  # e.g., [512, 256, 128]
        num_cont_features = Config.NUM_CONT_FEATURES

        # ---------------------------------------------------------------------
        # Stream 1: Categorical Processing
        # ---------------------------------------------------------------------
        # Create embedding layers for each categorical feature
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=size, embedding_dim=embedding_dim)
                for size in vocab_sizes
            ]
        )

        # Calculate size after flattening all embeddings
        self.cat_flatten_dim = len(vocab_sizes) * embedding_dim

        # Gated Feature Selection for categorical data
        # Keeps dimension constant, acts as a filter
        self.cat_gate = GatedBlock(
            in_features=self.cat_flatten_dim,
            out_features=self.cat_flatten_dim,
            dropout_rate=dropout_rate,
        )

        # ---------------------------------------------------------------------
        # Stream 2: Continuous Processing
        # ---------------------------------------------------------------------
        # Gated Noise Filtering for continuous data
        # Keeps dimension constant
        self.cont_gate = GatedBlock(
            in_features=num_cont_features,
            out_features=num_cont_features,
            dropout_rate=dropout_rate,
        )

        # ---------------------------------------------------------------------
        # Fusion & Funnel Backbone
        # ---------------------------------------------------------------------
        # Input dimension to the funnel is the sum of both streams
        current_dim = self.cat_flatten_dim + num_cont_features

        funnel_layers = []
        for hidden_dim in hidden_layers:
            funnel_layers.append(
                GatedBlock(
                    in_features=current_dim,
                    out_features=hidden_dim,
                    dropout_rate=dropout_rate,
                )
            )
            current_dim = hidden_dim

        self.funnel = nn.Sequential(*funnel_layers)

        # ---------------------------------------------------------------------
        # Output Head
        # ---------------------------------------------------------------------
        # Direct linear projection from the final funnel layer to the target
        self.head = nn.Linear(current_dim, Config.OUTPUT_DIM)

    def forward(self, x_cat, x_cont):
        """
        Args:
            x_cat (torch.Tensor): Categorical indices [Batch, Num_Cat_Features]
            x_cont (torch.Tensor): Continuous values [Batch, Num_Cont_Features]
        """

        # --- Stream 1: Categorical ---
        # 1. Lookup embeddings for each feature
        # x_cat is [Batch, Num_Features], we iterate over columns
        emb_list = []
        for i, emb_layer in enumerate(self.embeddings):
            # Select column i, result is [Batch, Emb_Dim]
            emb_list.append(emb_layer(x_cat[:, i]))

        # 2. Flatten/Concatenate embeddings -> [Batch, Num_Features * Emb_Dim]
        cat_emb = torch.cat(emb_list, dim=1)

        # 3. Apply Gated Feature Selection
        cat_stream_out = self.cat_gate(cat_emb)

        # --- Stream 2: Continuous ---
        # 1. Apply Gated Noise Filtering -> [Batch, Num_Cont_Features]
        cont_stream_out = self.cont_gate(x_cont)

        # --- Fusion ---
        # Concatenate processed streams
        fusion = torch.cat([cat_stream_out, cont_stream_out], dim=1)

        # --- Funnel Backbone ---
        features = self.funnel(fusion)

        # --- Output ---
        logits = self.head(features)

        return logits
