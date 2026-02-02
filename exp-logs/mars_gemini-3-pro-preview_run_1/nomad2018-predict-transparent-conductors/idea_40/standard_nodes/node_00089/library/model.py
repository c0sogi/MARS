import torch
import torch.nn as nn
from library.config import Config


class AtomicStream(nn.Module):
    """
    Context-Aware Point Processor for the Atomic Stream.
    Projects dense atomic features (Identity, Coords, d_min, d_mean, Context)
    into a high-dimensional latent space using a Wide MLP.
    """

    def __init__(self):
        super().__init__()
        input_dim = Config.ATOMIC_FEATURE_DIM
        hidden_dim = Config.ATOMIC_HIDDEN_DIM
        dropout = Config.DROPOUT

        # Layer 1: Expansion
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        # Layer 2: Deep Processing
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        # Layer 3: Projection to Embedding Space (No activation)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, max_atoms, atomic_feature_dim)
        Returns:
            x: (batch_size, max_atoms, atomic_hidden_dim)
        """
        b, n, c = x.shape

        # Handle empty graph case if n=0
        if n == 0:
            return torch.zeros((b, n, Config.ATOMIC_HIDDEN_DIM), device=x.device)

        # Flatten for linear layers: (batch_size * max_atoms, feature_dim)
        x = x.view(-1, c)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop2(x)

        x = self.fc3(x)

        # Reshape back to sequence format
        x = x.view(b, n, -1)
        return x


class GlobalStream(nn.Module):
    """
    Thermodynamic Context Processor for the Global Stream.
    Encodes macroscopic features (Lattice, Volume, Density, Stoichiometry).
    """

    def __init__(self):
        super().__init__()
        input_dim = Config.GLOBAL_FEATURE_DIM
        hidden_dim = Config.GLOBAL_HIDDEN_DIM
        dropout = Config.DROPOUT

        # Layer 1: Expansion
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        # Layer 2: Projection (No activation)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, global_feature_dim)
        Returns:
            x: (batch_size, global_hidden_dim)
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        return x


class DC3_WDS(nn.Module):
    """
    Density-Calibrated Chemically-Contextualized Wide Deep Sets.
    Fuses aggregated atomic embeddings with global embeddings to predict material properties.
    """

    def __init__(self):
        super().__init__()
        self.atomic_stream = AtomicStream()
        self.global_stream = GlobalStream()

        atomic_dim = Config.ATOMIC_HIDDEN_DIM
        global_dim = Config.GLOBAL_HIDDEN_DIM

        # Fusion Input Dimension:
        # Mean Pooling (atomic_dim) + Max Pooling (atomic_dim) + Global Embedding (global_dim)
        fusion_input_dim = (2 * atomic_dim) + global_dim

        # Regressor Head
        fusion_hidden_dim = 512

        self.fusion_fc1 = nn.Linear(fusion_input_dim, fusion_hidden_dim)
        self.fusion_bn1 = nn.BatchNorm1d(fusion_hidden_dim)
        self.fusion_act1 = nn.ReLU()
        self.fusion_drop1 = nn.Dropout(Config.DROPOUT)

        self.fusion_fc2 = nn.Linear(fusion_hidden_dim, Config.NUM_TARGETS)

    def forward(self, atomic_feats, global_feats, mask):
        """
        Args:
            atomic_feats: (batch_size, max_atoms, atomic_feat_dim)
            global_feats: (batch_size, global_feat_dim)
            mask: (batch_size, max_atoms) - Boolean mask, True for valid atoms
        Returns:
            out: (batch_size, num_targets)
        """
        # 1. Process Atomic Stream
        # (B, N, H_atomic)
        atomic_emb = self.atomic_stream(atomic_feats)

        # 2. Aggregation (Dual Pooling)
        # Expand mask for broadcasting: (B, N, 1)
        mask_expanded = mask.unsqueeze(-1).float()

        # Apply mask to zero out padding contributions
        atomic_emb_masked = atomic_emb * mask_expanded

        # Mean Pooling
        sum_emb = torch.sum(atomic_emb_masked, dim=1)  # (B, H_atomic)
        counts = torch.sum(mask_expanded, dim=1)  # (B, 1)
        counts = torch.clamp(counts, min=1.0)  # Avoid div by zero
        mean_pool = sum_emb / counts

        # Max Pooling
        # Replace padding zeros with a large negative number to ignore them in max
        # (B, N, H_atomic)
        neg_inf = torch.ones_like(atomic_emb) * -1e9
        atomic_emb_for_max = torch.where(mask.unsqueeze(-1), atomic_emb, neg_inf)

        # Handle case where N=0 (though unlikely with collate_fn unless batch is empty)
        if atomic_emb.shape[1] > 0:
            max_pool, _ = torch.max(atomic_emb_for_max, dim=1)  # (B, H_atomic)

            # Fix for Debug Lesson 9: Explicitly Handle Empty Sets
            # If a graph is empty (all mask False), max_pool is -1e9. Reset to 0.
            graph_has_atoms = mask.sum(dim=1, keepdim=True) > 0  # (B, 1)
            max_pool = torch.where(
                graph_has_atoms, max_pool, torch.zeros_like(max_pool)
            )
        else:
            max_pool = torch.zeros(
                (atomic_emb.shape[0], atomic_emb.shape[2]), device=atomic_emb.device
            )

        # 3. Process Global Stream
        # (B, H_global)
        global_emb = self.global_stream(global_feats)

        # 4. Late Fusion
        # Concatenate: [Mean, Max, Global]
        fused = torch.cat([mean_pool, max_pool, global_emb], dim=1)

        # 5. Regression
        out = self.fusion_fc1(fused)
        out = self.fusion_bn1(out)
        out = self.fusion_act1(out)
        out = self.fusion_drop1(out)
        out = self.fusion_fc2(out)

        return out
