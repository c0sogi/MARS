import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class StabilizedWideMLP(nn.Module):
    """
    A Wide MLP with internal LayerNorm and GELU activation, used as a gate.
    Structure: Linear -> LayerNorm -> GELU -> Linear -> Sigmoid
    """

    def __init__(self, input_dim, output_dim, expansion=2):
        super(StabilizedWideMLP, self).__init__()
        hidden_dim = int(input_dim * expansion)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.sigmoid(x)


class GLU(nn.Module):
    """
    Gated Linear Unit.
    """

    def __init__(self, input_dim, output_dim):
        super(GLU, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        out = self.fc(x)
        a, b = out.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class DualPathInteractionModule(nn.Module):
    """
    Topology-Aware Dual-Path Interaction Module.
    Routes information differently for paired (stem) and unpaired (loop) bases.
    """

    def __init__(self, hidden_dim):
        super(DualPathInteractionModule, self).__init__()
        self.hidden_dim = hidden_dim

        # Path A: Paired Interaction (Stems)
        # Message: GLU based on neighbor state h_j
        self.msg_paired = GLU(hidden_dim, hidden_dim)
        # Gate: Stabilized Wide MLP taking [h_i; h_j]
        self.gate_paired = StabilizedWideMLP(hidden_dim * 2, hidden_dim)

        # Path B: Unpaired Refinement (Loops)
        # Message: GLU based on self state h_i
        self.msg_unpaired = GLU(hidden_dim, hidden_dim)
        # Gate: Stabilized Wide MLP taking h_i
        self.gate_unpaired = StabilizedWideMLP(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, pair_indices, pair_mask):
        """
        Args:
            h: (Batch, Seq, Hidden)
            pair_indices: (Batch, Seq) - Indices of paired bases
            pair_mask: (Batch, Seq, 1) - 1.0 if paired, 0.0 if unpaired
        """
        B, L, D = h.shape

        # 1. Gather Neighbor States for Paired Path
        # Create batch indices for gathering: (B, L)
        batch_idx = torch.arange(B, device=h.device).unsqueeze(1).expand(B, L)
        # Gather: h_neighbor[b, i] = h[b, pair_indices[b, i]]
        h_neighbor = h[batch_idx, pair_indices]  # (B, L, D)

        # 2. Path A: Paired (Stems)
        # Message from neighbor
        m_paired = self.msg_paired(h_neighbor)
        # Gate based on self and neighbor
        g_paired_in = torch.cat([h, h_neighbor], dim=-1)
        g_paired = self.gate_paired(g_paired_in)
        # Update
        u_paired = g_paired * m_paired

        # 3. Path B: Unpaired (Loops)
        # Message from self
        m_unpaired = self.msg_unpaired(h)
        # Gate based on self
        g_unpaired = self.gate_unpaired(h)
        # Update
        u_unpaired = g_unpaired * m_unpaired

        # 4. Fusion
        # Apply mask: Use u_paired where mask=1, u_unpaired where mask=0
        u_total = pair_mask * u_paired + (1.0 - pair_mask) * u_unpaired

        # 5. Residual + Norm
        out = self.norm(h + u_total)

        return out


class HCTADPBiGRU(nn.Module):
    """
    High-Capacity Topology-Aware Dual-Path BiGRU.
    """

    def __init__(self):
        super(HCTADPBiGRU, self).__init__()
        self.config = Config()

        # 1. Convolutional Stem
        # Projects sparse one-hot inputs (14 channels) to dense embedding (256)
        self.stem = nn.Sequential(
            nn.Conv1d(self.config.input_dim, 256, kernel_size=3, padding=1), nn.GELU()
        )

        # 2. Backbone
        self.num_layers = self.config.num_layers  # 4
        self.gru_hidden = self.config.hidden_dim  # 384
        self.total_hidden = self.gru_hidden * 2  # 768 (Bidirectional)

        self.grus = nn.ModuleList()
        self.interactions = nn.ModuleList()

        current_dim = 256

        for i in range(self.num_layers):
            # BiGRU Layer
            self.grus.append(
                nn.GRU(
                    input_size=current_dim,
                    hidden_size=self.gru_hidden,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            current_dim = self.total_hidden

            # Interaction Module (Blocks 1, 2, 3 only)
            if i < self.num_layers - 1:
                self.interactions.append(DualPathInteractionModule(self.total_hidden))

        # 3. Output Head
        self.dropout = nn.Dropout(self.config.dropout)
        self.head = nn.Linear(self.total_hidden, self.config.num_classes)

    def forward(self, x, pair_indices, pair_mask):
        """
        Args:
            x: (Batch, Seq, 14)
            pair_indices: (Batch, Seq)
            pair_mask: (Batch, Seq, 1)
        """
        # Permute for Conv1d: (B, 14, L)
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        # Permute back: (B, L, 256)
        x = x.permute(0, 2, 1)

        for i in range(self.num_layers):
            # BiGRU
            x, _ = self.grus[i](x)

            # Interaction (if present for this block)
            if i < len(self.interactions):
                x = self.interactions[i](x, pair_indices, pair_mask)

        x = self.dropout(x)
        out = self.head(x)

        return out
