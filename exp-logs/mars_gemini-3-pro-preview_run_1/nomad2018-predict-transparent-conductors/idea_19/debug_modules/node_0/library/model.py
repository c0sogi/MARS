import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class AtomicStream(nn.Module):
    """
    Chemically-Aware Point Processor.
    Processes per-atom features using a Wide MLP with rigorous regularization.
    """

    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(AtomicStream, self).__init__()

        # Wide MLP layers
        # Input -> Hidden
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Hidden -> Hidden (Deepening the wide representation)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Output Projection (No activation as per design)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask):
        """
        Args:
            x: (Batch, Length, Feat) Atomic features
            mask: (Batch, Length) Boolean mask (True for valid atoms)
        Returns:
            aggregated: (Batch, 2 * Hidden) Concatenated Mean and Max pooled embeddings
        """
        B, L, F = x.shape

        # Flatten for processing: (B*L, F)
        # We process all atoms (including padding) then mask later to keep batch norm statistics stable-ish
        # or we could mask first. Standard approach for 3D input in PyTorch Linear is supported.

        # Layer 1
        x = self.fc1(x)
        # Permute for BN: (B, C, L)
        x = x.permute(0, 2, 1)
        x = self.bn1(x)
        x = x.permute(0, 2, 1)
        x = F.relu(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        x = self.bn2(x)
        x = x.permute(0, 2, 1)
        x = F.relu(x)
        x = self.dropout2(x)

        # Projection Layer (No activation/BN/Dropout on the final projection before pooling)
        x = self.fc3(x)  # (B, L, H)

        # --- Dual Pooling ---

        # Expand mask for broadcasting: (B, L, 1)
        mask_expanded = mask.unsqueeze(-1).float()

        # 1. Global Mean Pooling
        # Sum valid embeddings
        sum_pooled = torch.sum(x * mask_expanded, dim=1)  # (B, H)
        # Count valid atoms (avoid division by zero)
        counts = torch.sum(mask_expanded, dim=1).clamp(min=1.0)  # (B, 1)
        mean_pooled = sum_pooled / counts

        # 2. Global Max Pooling
        # Set padding to large negative value
        # Clone x to avoid modifying inplace if needed, though x is intermediate here
        x_max = x.clone()
        x_max[~mask] = -1e9
        max_pooled = torch.max(x_max, dim=1)[0]  # (B, H)

        # Concatenate
        aggregated = torch.cat([mean_pooled, max_pooled], dim=1)  # (B, 2*H)

        return aggregated


class GlobalStream(nn.Module):
    """
    Thermodynamic Context Encoder.
    Processes macroscopic features using a High-Capacity MLP.
    """

    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(GlobalStream, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Final projection
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: (Batch, Global_Feat)
        Returns:
            embedding: (Batch, Hidden)
        """
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Projection
        x = self.fc3(x)

        return x


class CR_WDS(nn.Module):
    """
    Chemically-Resolved Wide Deep Sets.
    Orchestrates the Atomic and Global streams and performs Late Fusion.
    """

    def __init__(self):
        super(CR_WDS, self).__init__()

        # Hyperparameters from Config
        atomic_in = Config.ATOMIC_FEATURE_DIM
        atomic_hidden = Config.ATOMIC_HIDDEN_DIM
        global_in = Config.GLOBAL_FEATURE_DIM
        global_hidden = Config.GLOBAL_HIDDEN_DIM
        dropout = Config.DROPOUT_RATE
        num_targets = Config.NUM_TARGETS

        # Streams
        self.atomic_stream = AtomicStream(atomic_in, atomic_hidden, dropout)
        self.global_stream = GlobalStream(global_in, global_hidden, dropout)

        # Fusion Head
        # Input: (Atomic_Mean + Atomic_Max) + Global
        fusion_input_dim = (atomic_hidden * 2) + global_hidden

        self.fusion_fc1 = nn.Linear(fusion_input_dim, atomic_hidden)
        self.fusion_bn1 = nn.BatchNorm1d(atomic_hidden)
        self.fusion_dropout = nn.Dropout(dropout)

        self.output_layer = nn.Linear(atomic_hidden, num_targets)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict: Dictionary containing:
                - 'atomic_features': (B, L, 11)
                - 'mask': (B, L)
                - 'global_features': (B, 12)
        Returns:
            predictions: (B, 2)
        """
        atomic_x = batch_dict["atomic_features"]
        mask = batch_dict["mask"]
        global_x = batch_dict["global_features"]

        # 1. Process Atomic Stream
        # Returns (B, 2*Atomic_Hidden)
        atomic_emb = self.atomic_stream(atomic_x, mask)

        # 2. Process Global Stream
        # Returns (B, Global_Hidden)
        global_emb = self.global_stream(global_x)

        # 3. Late Fusion
        combined = torch.cat([atomic_emb, global_emb], dim=1)

        # 4. Regression Head
        x = self.fusion_fc1(combined)
        x = self.fusion_bn1(x)
        x = F.relu(x)
        x = self.fusion_dropout(x)

        predictions = self.output_layer(x)

        return predictions
