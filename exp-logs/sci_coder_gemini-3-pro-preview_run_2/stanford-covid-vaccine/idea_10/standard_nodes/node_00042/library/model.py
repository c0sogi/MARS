import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    INPUT_CHANNELS,
    NUM_TARGETS,
    HIDDEN_DIM,
    KERNEL_SIZE,
    DROPOUT,
    DILATIONS,
    GRU_HIDDEN_DIM,
    GRU_LAYERS,
    SEQ_LEN,
)


class ResidualDilatedBlock(nn.Module):
    """
    A single dilated convolutional block with a residual connection.
    Cite solution_lesson_node_00011: Single-Layer Dilated Residual Blocks
    Cite solution_lesson_node_00017: No Batch Normalization to avoid overfitting
    """

    def __init__(self, channels, dilation, kernel_size=3, dropout=0.1):
        super().__init__()

        # Calculate padding to maintain sequence length
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)
        return x + out


class ResidualTCNStack(nn.Module):
    """
    A stack of ResidualDilatedBlocks.
    """

    def __init__(self, channels, dilations, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(
                ResidualDilatedBlock(channels, dilation=d, dropout=dropout)
            )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class StagedInteractiveDenseNet(nn.Module):
    """
    Renamed to keep compatibility with runfile, but implements:
    Interactive Residual TCN (Cite solution_lesson_node_00031)

    Architecture:
    1. Embedding
    2. TCN Stage 1 (Local Context)
    3. Latent Gather (Interaction)
    4. TCN Stage 2 (Pair Context)
    5. BiGRU (Global Aggregation)
    6. Head
    """

    def __init__(self):
        super().__init__()

        # 1. Input Projection
        self.embedding = nn.Sequential(
            nn.Conv1d(INPUT_CHANNELS, HIDDEN_DIM, kernel_size=1),
            nn.ReLU(),
        )

        # 2. Stage 1: Local Context
        self.stage1 = ResidualTCNStack(
            channels=HIDDEN_DIM,
            dilations=DILATIONS,
            dropout=DROPOUT,
        )

        # 3. Latent Gather (Implemented in method)
        # Output dim will be HIDDEN_DIM * 2

        # Projection after gather to return to HIDDEN_DIM
        self.post_gather_proj = nn.Sequential(
            nn.Conv1d(HIDDEN_DIM * 2, HIDDEN_DIM, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

        # 4. Stage 2: Pair Context
        self.stage2 = ResidualTCNStack(
            channels=HIDDEN_DIM,
            dilations=DILATIONS,
            dropout=DROPOUT,
        )

        # 5. Global Aggregation (BiGRU)
        self.gru = nn.GRU(
            input_size=HIDDEN_DIM,
            hidden_size=GRU_HIDDEN_DIM,
            num_layers=GRU_LAYERS,
            batch_first=True,
            bidirectional=True,
        )

        # 6. Output Head
        self.head = nn.Linear(GRU_HIDDEN_DIM * 2, NUM_TARGETS)

    def latent_gather(self, x, partner_indices):
        """
        Gathers features from partner positions.
        Cite solution_lesson_node_00031
        """
        B, C, L = x.shape

        # Create a dummy column of zeros for unpaired bases (-1 indices)
        dummy = torch.zeros(B, C, 1, device=x.device, dtype=x.dtype)

        # Concatenate dummy to the end of sequence
        x_padded = torch.cat([x, dummy], dim=2)

        # Adjust indices: replace -1 with L
        indices = partner_indices.clone()
        indices[indices == -1] = L

        # Expand indices for gather
        indices_expanded = indices.unsqueeze(1).expand(-1, C, -1)

        # Gather features
        x_gathered = torch.gather(x_padded, 2, indices_expanded)

        # Concatenate original and gathered features
        return torch.cat([x, x_gathered], dim=1)

    def forward(self, x, partner_indices):
        # 1. Embedding
        x = self.embedding(x)

        # 2. Stage 1
        x = self.stage1(x)

        # 3. Interaction
        x = self.latent_gather(x, partner_indices)
        x = self.post_gather_proj(x)

        # 4. Stage 2
        x = self.stage2(x)

        # 5. Global Aggregation
        x = x.permute(0, 2, 1)  # (B, L, C)
        x, _ = self.gru(x)

        # 6. Head
        out = self.head(x)

        return out
