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


class DenseDilatedBlock(nn.Module):
    """
    A single dilated convolutional block for the DenseNet architecture.
    It projects the accumulated dense input to a bottleneck, applies dilated conv,
    and returns the new feature map.
    """

    def __init__(self, in_channels, out_channels, dilation, kernel_size=3, dropout=0.1):
        super().__init__()

        # Calculate padding to maintain sequence length
        # padding = (kernel_size - 1) * dilation // 2
        padding = (kernel_size - 1) * dilation // 2

        self.net = nn.Sequential(
            # Bottleneck 1x1 Conv to reduce channel explosion from dense connections
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            # Dilated Convolution
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DenseDilatedStage(nn.Module):
    """
    A stage consisting of multiple DenseDilatedBlocks.
    Implements dense connections: Input to layer i is concatenation of all 0..i-1 outputs.
    """

    def __init__(self, in_channels, growth_rate, dilations, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()

        current_in_channels = in_channels

        for d in dilations:
            block = DenseDilatedBlock(
                in_channels=current_in_channels,
                out_channels=growth_rate,
                dilation=d,
                kernel_size=KERNEL_SIZE,
                dropout=dropout,
            )
            self.blocks.append(block)
            current_in_channels += growth_rate

        self.out_channels = current_in_channels

    def forward(self, x):
        # x shape: (B, C_in, L)
        features = [x]

        for block in self.blocks:
            # Concatenate all previous features
            in_tensor = torch.cat(features, dim=1)
            # Compute new features
            out = block(in_tensor)
            # Append to list
            features.append(out)

        # Return concatenation of all features (DenseNet style)
        return torch.cat(features, dim=1)


class StagedInteractiveDenseNet(nn.Module):
    """
    The main architecture:
    1. Input Projection
    2. Stage 1 (Local Dense Context)
    3. Latent Gather (Interaction via Partner Indices)
    4. Stage 2 (Pair Dense Context)
    5. BiGRU (Global Aggregation)
    6. Output Head
    """

    def __init__(self):
        super().__init__()

        # 1. Input Projection
        self.embedding = nn.Sequential(
            nn.Conv1d(INPUT_CHANNELS, HIDDEN_DIM, kernel_size=1),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
        )

        # 2. Stage 1: Local Context
        # Input: HIDDEN_DIM
        # Output: HIDDEN_DIM + (len(DILATIONS) * HIDDEN_DIM)
        self.stage1 = DenseDilatedStage(
            in_channels=HIDDEN_DIM,
            growth_rate=HIDDEN_DIM,
            dilations=DILATIONS,
            dropout=DROPOUT,
        )

        stage1_out_dim = self.stage1.out_channels

        # 3. Inter-Stage Projection (Optional but good for parameter control)
        # We project the concatenated Stage 1 output back to a manageable size
        # before the interaction/Stage 2 to keep memory in check,
        # or we feed the full dense representation.
        # Given the "Dense" philosophy, we keep the features but let's ensure
        # the next stage can handle 2x input size (due to gather).

        # 4. Stage 2: Pair Context
        # Input size will be 2 * stage1_out_dim (Original + Partner)
        self.stage2 = DenseDilatedStage(
            in_channels=stage1_out_dim * 2,
            growth_rate=HIDDEN_DIM,
            dilations=DILATIONS,
            dropout=DROPOUT,
        )

        stage2_out_dim = self.stage2.out_channels

        # 5. Global Aggregation (BiGRU)
        # We project Stage 2 output to a smaller dim for GRU efficiency
        self.pre_gru_proj = nn.Sequential(
            nn.Conv1d(stage2_out_dim, HIDDEN_DIM, kernel_size=1),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size=HIDDEN_DIM,
            hidden_size=GRU_HIDDEN_DIM,
            num_layers=GRU_LAYERS,
            batch_first=True,
            bidirectional=True,
        )

        # 6. Output Head
        # BiGRU outputs 2 * GRU_HIDDEN_DIM
        self.head = nn.Linear(GRU_HIDDEN_DIM * 2, NUM_TARGETS)

    def latent_gather(self, x, partner_indices):
        """
        Gathers features from partner positions.
        x: (B, C, L)
        partner_indices: (B, L) with -1 for unpaired.
        """
        B, C, L = x.shape

        # Create a dummy column of zeros for unpaired bases (-1 indices)
        # Shape: (B, C, 1)
        dummy = torch.zeros(B, C, 1, device=x.device, dtype=x.dtype)

        # Concatenate dummy to the end of sequence
        # Shape: (B, C, L+1)
        x_padded = torch.cat([x, dummy], dim=2)

        # Adjust indices: replace -1 with L (index of dummy column)
        # partner_indices is (B, L)
        # We clone to ensure we don't modify the input tensor in place
        indices = partner_indices.clone()
        indices[indices == -1] = L

        # Expand indices for gather: (B, C, L)
        indices_expanded = indices.unsqueeze(1).expand(-1, C, -1)

        # Gather features
        # Shape: (B, C, L)
        x_gathered = torch.gather(x_padded, 2, indices_expanded)

        # Concatenate original and gathered features
        # Shape: (B, 2*C, L)
        return torch.cat([x, x_gathered], dim=1)

    def forward(self, x, partner_indices):
        """
        Args:
            x: (B, InputChannels, L)
            partner_indices: (B, L)
        Returns:
            (B, L, NumTargets)
        """
        # 1. Embedding
        x = self.embedding(x)  # (B, Hidden, L)

        # 2. Stage 1
        x_s1 = self.stage1(x)  # (B, Stage1_Out, L)

        # 3. Interaction
        x_inter = self.latent_gather(x_s1, partner_indices)  # (B, 2*Stage1_Out, L)

        # 4. Stage 2
        x_s2 = self.stage2(x_inter)  # (B, Stage2_Out, L)

        # 5. Global Aggregation
        # Project before GRU
        x_gru_in = self.pre_gru_proj(x_s2)  # (B, Hidden, L)

        # Permute for GRU: (B, L, Hidden)
        x_gru_in = x_gru_in.permute(0, 2, 1)

        # GRU
        x_gru_out, _ = self.gru(x_gru_in)  # (B, L, 2*GruHidden)

        # 6. Head
        out = self.head(x_gru_out)  # (B, L, NumTargets)

        return out
