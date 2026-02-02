import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DecoupledInteractionModule(nn.Module):
    """
    Stabilized Bias-Refined Decoupled Interaction Module.

    Implements the structural interaction mechanism with specific stability fixes:
    1. Point-to-point gathering with explicit zero-masking for unpaired bases.
    2. Bias-refined message computation allowing bias to act as loop embedding.
    3. Stabilized MLP Gate with internal LayerNorm but NO logit normalization.
    4. Residual injection with Post-LayerNorm.
    """

    def __init__(self, hidden_dim):
        super().__init__()

        # Message Pathway
        # m_ij = GELU(W_msg * h_j + b_msg)
        # When h_j is masked to 0 (unpaired), m_ij = GELU(b_msg)
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)

        # Gate Pathway
        # Projects concatenated context [h_i; h_j] -> hidden_dim
        self.gate_proj1 = nn.Linear(2 * hidden_dim, hidden_dim)

        # Internal Normalization for Stability (Lesson 75)
        self.gate_norm = nn.LayerNorm(hidden_dim)

        # Project to logits
        self.gate_proj2 = nn.Linear(hidden_dim, hidden_dim)

        # Output Normalization
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, pair_indices):
        """
        Args:
            x: Tensor of shape (N, L, D)
            pair_indices: LongTensor of shape (N, L) with -1 for unpaired.
        """
        N, L, D = x.shape

        # 1. Gather Neighbor States (h_j)
        # Handle -1 indices by replacing with 0 temporarily, then masking
        # valid_mask is 1 where paired, 0 where unpaired
        valid_mask = (pair_indices != -1).unsqueeze(-1).float()  # (N, L, 1)

        safe_indices = pair_indices.clone()
        safe_indices[pair_indices == -1] = 0

        # Expand indices for gather: (N, L, D)
        gather_idx = safe_indices.unsqueeze(-1).expand(-1, -1, D)

        # Gather h_j
        h_j = torch.gather(x, 1, gather_idx)

        # Apply Zero-Masking for unpaired bases (Lesson 64)
        # If unpaired, h_j becomes 0 vector.
        h_j = h_j * valid_mask

        # 2. Compute Message (Bias-Refined)
        # If h_j is 0, this becomes GELU(bias), serving as a learned embedding for loops.
        m = F.gelu(self.msg_proj(h_j))

        # 3. Compute Gate (Stabilized)
        # Concatenate source and neighbor
        cat_input = torch.cat([x, h_j], dim=-1)

        # Project
        z_raw = self.gate_proj1(cat_input)

        # Internal Normalization (Stabilization - Lesson 75)
        z_norm = self.gate_norm(z_raw)

        # Activation
        z_act = F.gelu(z_norm)

        # Logits and Sigmoid (No Logit Norm - Lesson 78)
        logits = self.gate_proj2(z_act)
        g = torch.sigmoid(logits)

        # 4. Injection and Residual
        h_res = x + g * m

        # 5. Post-Normalization (Lesson 68)
        h_out = self.out_norm(h_res)

        return h_out


class DeepStabilizedBiGRU(nn.Module):
    """
    4-Layer Backbone with Stabilized Decoupled Interaction Modules.

    Structure:
    - Conv1d Stem
    - Block 1: BiGRU + Interaction
    - Block 2: BiGRU + Interaction
    - Block 3: BiGRU + Interaction
    - Block 4: BiGRU (No Interaction)
    - Linear Head
    """

    def __init__(self):
        super().__init__()

        # Hyperparameters
        self.hidden_dim = Config.HIDDEN_DIM  # 384 (GRU hidden size)
        self.gru_output_dim = self.hidden_dim * 2  # Bidirectional -> 768

        # 1. Convolutional Stem
        self.stem = nn.Conv1d(
            in_channels=Config.INPUT_DIM,
            out_channels=Config.CNN_FILTERS,
            kernel_size=Config.CNN_KERNEL_SIZE,
            padding=Config.CNN_KERNEL_SIZE // 2,
        )
        self.stem_act = nn.GELU()
        self.dropout = nn.Dropout(Config.DROPOUT)

        # 2. Deep Backbone (4 Blocks)
        # Block 1
        self.gru1 = nn.GRU(
            input_size=Config.CNN_FILTERS,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.inter1 = DecoupledInteractionModule(self.gru_output_dim)

        # Block 2
        self.gru2 = nn.GRU(
            input_size=self.gru_output_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.inter2 = DecoupledInteractionModule(self.gru_output_dim)

        # Block 3
        self.gru3 = nn.GRU(
            input_size=self.gru_output_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.inter3 = DecoupledInteractionModule(self.gru_output_dim)

        # Block 4 (No Interaction Module as per instructions)
        self.gru4 = nn.GRU(
            input_size=self.gru_output_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # 3. Output Head
        self.head = nn.Linear(self.gru_output_dim, Config.NUM_TARGETS)

    def forward(self, sequence, pair_indices):
        """
        Args:
            sequence: (N, L, 14)
            pair_indices: (N, L)
        """
        # Permute for Conv1d: (N, C, L)
        x = sequence.transpose(1, 2)

        # Stem
        x = self.stem(x)
        x = self.stem_act(x)
        x = self.dropout(x)

        # Permute back for GRU: (N, L, C)
        x = x.transpose(1, 2)

        # Block 1
        x, _ = self.gru1(x)
        x = self.inter1(x, pair_indices)
        x = self.dropout(x)

        # Block 2
        x, _ = self.gru2(x)
        x = self.inter2(x, pair_indices)
        x = self.dropout(x)

        # Block 3
        x, _ = self.gru3(x)
        x = self.inter3(x, pair_indices)
        x = self.dropout(x)

        # Block 4 (No Interaction)
        x, _ = self.gru4(x)
        x = self.dropout(x)

        # Head
        out = self.head(x)

        return out
