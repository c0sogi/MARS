import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max
from library.config import (
    ATOMIC_INPUT_DIM,
    ATOMIC_HIDDEN_DIM,
    ATOMIC_LAYERS,
    ATOMIC_DROPOUT,
    ATOMIC_EMBEDDING_DIM,
    GLOBAL_INPUT_DIM,
    GLOBAL_HIDDEN_DIM,
    GLOBAL_LAYERS,
    GLOBAL_DROPOUT,
    GLOBAL_EMBEDDING_DIM,
    FUSION_HIDDEN_DIMS,
    FUSION_DROPOUT,
    OUTPUT_DIM,
)


class WideBlock(nn.Module):
    """
    A single block of a Wide MLP: Linear -> BN -> ReLU -> Dropout.
    Used to ensure consistent regularization and capacity scaling.
    """

    def __init__(self, input_dim, output_dim, dropout_prob):
        super(WideBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class AtomicStream(nn.Module):
    """
    Wide Point Processor for the Atomic Stream.
    Processes individual atoms with a wide MLP to capture local chemical and spatial environments.
    """

    def __init__(self):
        super(AtomicStream, self).__init__()

        layers = []
        # Immediate expansion to hidden dimension
        input_dim = ATOMIC_INPUT_DIM

        for _ in range(ATOMIC_LAYERS):
            layers.append(WideBlock(input_dim, ATOMIC_HIDDEN_DIM, ATOMIC_DROPOUT))
            input_dim = ATOMIC_HIDDEN_DIM  # Next layer input is current hidden

        self.encoder = nn.Sequential(*layers)

        # Final projection to embedding space (no activation/bn/dropout usually for the embedding itself,
        # or kept simple linear as per 'Output' description in prompt)
        self.projection = nn.Linear(ATOMIC_HIDDEN_DIM, ATOMIC_EMBEDDING_DIM)

    def forward(self, atomic_feats):
        # atomic_feats: (Total_Atoms, ATOMIC_INPUT_DIM)
        x = self.encoder(atomic_feats)
        x = self.projection(x)
        return x


class GlobalStream(nn.Module):
    """
    Physics-Enhanced Context Encoder for the Global Stream.
    Processes macroscopic properties including physics-injected features.
    """

    def __init__(self):
        super(GlobalStream, self).__init__()

        layers = []
        input_dim = GLOBAL_INPUT_DIM

        for _ in range(GLOBAL_LAYERS):
            layers.append(WideBlock(input_dim, GLOBAL_HIDDEN_DIM, GLOBAL_DROPOUT))
            input_dim = GLOBAL_HIDDEN_DIM

        self.encoder = nn.Sequential(*layers)
        self.projection = nn.Linear(GLOBAL_HIDDEN_DIM, GLOBAL_EMBEDDING_DIM)

    def forward(self, global_feats):
        # global_feats: (Batch_Size, GLOBAL_INPUT_DIM)
        x = self.encoder(global_feats)
        x = self.projection(x)
        return x


class DualPooling(nn.Module):
    """
    Aggregates atomic embeddings using both Mean and Max pooling.
    """

    def __init__(self):
        super(DualPooling, self).__init__()

    def forward(self, x, batch_indices, batch_size):
        # x: (Total_Atoms, Embedding_Dim)
        # batch_indices: (Total_Atoms,)

        # Mean Pooling
        mean_pool = scatter_mean(x, batch_indices, dim=0, dim_size=batch_size)

        # Max Pooling (scatter_max returns values and indices, we need values)
        max_pool, _ = scatter_max(x, batch_indices, dim=0, dim_size=batch_size)

        # Concatenate: (Batch_Size, Embedding_Dim * 2)
        return torch.cat([mean_pool, max_pool], dim=1)


class GPA_WDS(nn.Module):
    """
    Global-Physics Augmented Wide Deep Sets.
    Main architecture combining Atomic and Global streams via Late Fusion.
    """

    def __init__(self):
        super(GPA_WDS, self).__init__()

        self.atomic_stream = AtomicStream()
        self.global_stream = GlobalStream()
        self.pooling = DualPooling()

        # Calculate fusion input dimension
        # Atomic: Embedding * 2 (Mean + Max)
        # Global: Embedding
        fusion_input_dim = (ATOMIC_EMBEDDING_DIM * 2) + GLOBAL_EMBEDDING_DIM

        # Fusion Head (Regressor)
        fusion_layers = []
        in_dim = fusion_input_dim

        for hidden_dim in FUSION_HIDDEN_DIMS:
            fusion_layers.append(WideBlock(in_dim, hidden_dim, FUSION_DROPOUT))
            in_dim = hidden_dim

        self.fusion_encoder = nn.Sequential(*fusion_layers)

        # Final prediction layer
        self.output_layer = nn.Linear(in_dim, OUTPUT_DIM)

    def forward(self, atomic_feats, global_feats, batch_indices, batch_ids):
        """
        Args:
            atomic_feats: (Total_Atoms, ATOMIC_INPUT_DIM)
            global_feats: (Batch_Size, GLOBAL_INPUT_DIM)
            batch_indices: (Total_Atoms,) mapping atoms to crystals
            batch_ids: (Batch_Size,) crystal IDs (unused in forward, but passed by collate)

        Returns:
            predictions: (Batch_Size, OUTPUT_DIM)
        """
        batch_size = global_feats.size(0)

        # 1. Atomic Stream
        atom_emb = self.atomic_stream(
            atomic_feats
        )  # (Total_Atoms, ATOMIC_EMBEDDING_DIM)

        # 2. Aggregation
        atom_pooled = self.pooling(
            atom_emb, batch_indices, batch_size
        )  # (Batch_Size, ATOMIC_EMBEDDING_DIM * 2)

        # 3. Global Stream
        global_emb = self.global_stream(
            global_feats
        )  # (Batch_Size, GLOBAL_EMBEDDING_DIM)

        # 4. Late Fusion
        fused = torch.cat(
            [atom_pooled, global_emb], dim=1
        )  # (Batch_Size, Fusion_Input_Dim)

        # 5. Regression
        features = self.fusion_encoder(fused)
        predictions = self.output_layer(features)

        return predictions
