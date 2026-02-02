import torch
import torch.nn as nn
from library.config import Config


class AtomicStream(nn.Module):
    """
    Chemically-Aware Point Processor.
    Processes per-atom features using a Wide MLP with Immediate Expansion.
    Aggregates atomic embeddings using Dual Pooling (Global Mean + Global Max).
    """

    def __init__(self):
        super().__init__()
        self.input_dim = Config.ATOMIC_FEATURE_DIM
        self.hidden_dim = Config.ATOMIC_HIDDEN_DIM
        # Project to a compact embedding space before pooling
        self.output_dim = 256
        self.dropout_rate = Config.DROPOUT_RATE

        # Wide MLP Layers
        # Layer 1: Expansion
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.dropout1 = nn.Dropout(self.dropout_rate)

        # Layer 2: Deep processing
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)
        self.dropout2 = nn.Dropout(self.dropout_rate)

        # Projection Layer (Linear, no activation as per strategy)
        self.fc_out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, mask):
        """
        Args:
            x: (Batch, N_atoms, Features)
            mask: (Batch, N_atoms) - True for valid atoms, False for padding
        """
        B, N, F = x.shape

        # Flatten to (B*N, F) for Batch Normalization
        x_flat = x.view(-1, F)

        # Block 1
        x_flat = self.fc1(x_flat)
        x_flat = self.bn1(x_flat)
        x_flat = torch.relu(x_flat)
        x_flat = self.dropout1(x_flat)

        # Block 2
        x_flat = self.fc2(x_flat)
        x_flat = self.bn2(x_flat)
        x_flat = torch.relu(x_flat)
        x_flat = self.dropout2(x_flat)

        # Projection
        x_flat = self.fc_out(x_flat)  # (B*N, output_dim)

        # Reshape back to (B, N, output_dim)
        x_out = x_flat.view(B, N, -1)

        # Apply Masking
        # Expand mask to match feature dim: (B, N, 1)
        mask_expanded = mask.unsqueeze(-1).float()
        x_masked = x_out * mask_expanded

        # --- Dual Pooling ---

        # 1. Global Mean Pooling
        # Sum valid vectors and divide by the number of valid atoms
        sum_pooled = torch.sum(x_masked, dim=1)  # (B, output_dim)
        # Avoid division by zero
        counts = torch.sum(mask_expanded, dim=1).clamp(min=1e-9)
        mean_pooled = sum_pooled / counts

        # 2. Global Max Pooling
        # Replace padded positions with a large negative value so they don't affect max
        neg_inf = torch.full_like(x_masked, -1e9)
        # Use x_masked where mask is True, else -1e9
        x_for_max = torch.where(mask.unsqueeze(-1), x_masked, neg_inf)
        max_pooled, _ = torch.max(x_for_max, dim=1)  # (B, output_dim)

        # Concatenate pooled representations
        return torch.cat([mean_pooled, max_pooled], dim=1)


class GlobalStream(nn.Module):
    """
    Thermodynamic Context Processor.
    Processes macroscopic features using a High-Capacity MLP.
    """

    def __init__(self):
        super().__init__()
        self.input_dim = Config.GLOBAL_FEATURE_DIM
        self.hidden_dim = Config.GLOBAL_HIDDEN_DIM
        self.dropout_rate = Config.DROPOUT_RATE

        # MLP Layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.dropout1 = nn.Dropout(self.dropout_rate)

        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)
        self.dropout2 = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        # x: (Batch, Global_Features)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        return x


class ChemicallyWeightedDeepSets(nn.Module):
    """
    Main CW-DS Model.
    Integrates Atomic and Global streams via Late Fusion.
    """

    def __init__(self):
        super().__init__()
        self.atomic_stream = AtomicStream()
        self.global_stream = GlobalStream()

        # Fusion Head Dimensions
        # Atomic: 256 (Mean) + 256 (Max) = 512
        # Global: 256
        # Total Fusion Input: 768
        self.fusion_input_dim = (
            self.atomic_stream.output_dim * 2
        ) + self.global_stream.hidden_dim
        self.fusion_hidden_dim = Config.FUSION_HIDDEN_DIM

        # Fusion Layers
        self.fusion_fc1 = nn.Linear(self.fusion_input_dim, self.fusion_hidden_dim)
        self.fusion_bn1 = nn.BatchNorm1d(self.fusion_hidden_dim)
        self.fusion_dropout = nn.Dropout(Config.DROPOUT_RATE)

        # Final Regression Head (Predicts 2 targets)
        self.output_layer = nn.Linear(self.fusion_hidden_dim, 2)

    def forward(self, atomic_features, atomic_mask, global_features):
        """
        Args:
            atomic_features: (B, N, 12)
            atomic_mask: (B, N)
            global_features: (B, 12)
        """
        # Process Atomic Stream
        atomic_embedding = self.atomic_stream(atomic_features, atomic_mask)

        # Process Global Stream
        global_embedding = self.global_stream(global_features)

        # Late Fusion
        combined = torch.cat([atomic_embedding, global_embedding], dim=1)

        # Regression
        x = self.fusion_fc1(combined)
        x = self.fusion_bn1(x)
        x = torch.relu(x)
        x = self.fusion_dropout(x)

        output = self.output_layer(x)
        return output
