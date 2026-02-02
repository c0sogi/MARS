import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library.config import Config


class ModalityStem(nn.Module):
    """
    Independent processing stem for a single modality.
    Structure: Linear -> Temporal Conv1d(k=7) -> ReLU -> Dropout
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, dropout):
        super(ModalityStem, self).__init__()
        self.project = nn.Linear(input_dim, hidden_dim)

        # Padding to maintain temporal dimension: (k-1)//2 for odd k
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Time, InputDim)

        # 1. Linear Projection
        x = self.project(x)  # (B, T, HiddenDim)

        # 2. Temporal Conv1d
        # Conv1d expects (Batch, Channels, Length)
        x = x.permute(0, 2, 1)  # (B, HiddenDim, T)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Restore shape (Batch, Time, HiddenDim)
        x = x.permute(0, 2, 1)
        return x


class DAGINet(nn.Module):
    """
    Decoupled-Anchor Gated-Injection Network (DAGI-Net).
    Features:
    - Independent Modality Stems
    - Decoupled Global Anchors (GAP on stems)
    - Gated Fusion conditioned on Local + Global
    - Dual-Injection BiGRU Backbone
    """

    def __init__(self):
        super(DAGINet, self).__init__()

        # Hyperparameters from Config
        self.hidden_size = Config.HIDDEN_SIZE
        self.num_classes = Config.NUM_CLASSES
        self.dropout_rate = Config.DROPOUT
        self.kernel_size = Config.CNN_KERNEL_SIZE

        # --- 1. Decoupled Input Stems ---
        # We project each modality to 'hidden_size'.
        # The fused representation will be 2 * hidden_size.
        self.stem_dim = self.hidden_size

        self.skeleton_stem = ModalityStem(
            input_dim=Config.SKELETON_INPUT_DIM,
            hidden_dim=self.stem_dim,
            kernel_size=self.kernel_size,
            dropout=self.dropout_rate,
        )

        self.audio_stem = ModalityStem(
            input_dim=Config.N_MFCC,
            hidden_dim=self.stem_dim,
            kernel_size=self.kernel_size,
            dropout=self.dropout_rate,
        )

        self.fused_dim = self.stem_dim * 2

        # --- 2. Gated Fusion ---
        self.layer_norm = nn.LayerNorm(self.fused_dim)

        # Gate components: Sigmoid(W_x * X_local + W_g * G_global + b)
        self.gate_x_proj = nn.Linear(self.fused_dim, self.fused_dim)
        self.gate_g_proj = nn.Linear(self.fused_dim, self.fused_dim)

        # --- 3. Dual-Injected BiGRU Backbone ---

        # Layer 1: Processes Gated Fusion
        self.gru1 = nn.GRU(
            input_size=self.fused_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.gru_out_dim = self.hidden_size * 2

        # Injection Projections for Layer 2
        # Input to Layer 2 = H1 + Proj_local(Y) + Proj_global(G)
        self.proj_local = nn.Linear(self.fused_dim, self.gru_out_dim)
        self.proj_global = nn.Linear(self.fused_dim, self.gru_out_dim)

        # Layer 2: Processes Injected Context
        self.gru2 = nn.GRU(
            input_size=self.gru_out_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # --- 4. Non-Linear Output Head ---
        self.head = nn.Sequential(
            nn.Linear(self.gru_out_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.num_classes),
        )

    def forward(self, skeleton, audio, lengths):
        """
        Args:
            skeleton: (Batch, Time, 60)
            audio: (Batch, Time, 13)
            lengths: (Batch,) - Valid lengths of sequences
        Returns:
            logits: (Batch, Time, NumClasses)
        """

        # --- 1. Decoupled Stems ---
        skel_feat = self.skeleton_stem(skeleton)  # (B, T, stem_dim)
        audio_feat = self.audio_stem(audio)  # (B, T, stem_dim)

        # --- 2. Decoupled Anchors (GAP) ---
        # Create mask for valid frames to ignore padding in GAP
        max_len = skeleton.size(1)
        # mask: (B, T), True where valid
        mask = torch.arange(max_len, device=skeleton.device)[None, :] < lengths[:, None]
        mask_float = mask.float().unsqueeze(-1)  # (B, T, 1)

        # Compute GAP (Sum / Length)
        # Avoid division by zero for safety (though lengths >= 1 usually)
        len_float = lengths.unsqueeze(-1).float().clamp(min=1.0)

        skel_gap = (skel_feat * mask_float).sum(dim=1) / len_float  # (B, stem_dim)
        audio_gap = (audio_feat * mask_float).sum(dim=1) / len_float  # (B, stem_dim)

        decoupled_anchors = torch.cat([skel_gap, audio_gap], dim=1)  # (B, fused_dim)

        # --- 3. Gated Fusion ---
        x_fused = torch.cat([skel_feat, audio_feat], dim=2)  # (B, T, fused_dim)
        x_fused = self.layer_norm(x_fused)

        # Gate Calculation
        # Broadcast anchors: (B, fused_dim) -> (B, 1, fused_dim)
        gate_content = self.gate_x_proj(x_fused) + self.gate_g_proj(
            decoupled_anchors
        ).unsqueeze(1)
        gate = torch.sigmoid(gate_content)

        # Apply Gate
        y = x_fused * gate  # (B, T, fused_dim)

        # --- 4. Dual-Injection Backbone ---

        # Layer 1
        packed_input = pack_padded_sequence(
            y, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_h1, _ = self.gru1(packed_input)
        h1, _ = pad_packed_sequence(packed_h1, batch_first=True)  # (B, T, 2*hidden)

        # Refined Global Context (Cite solution_lesson_node_00106)
        # Re-compute anchors from the gated output 'y' to suppress noise
        refined_anchors = (y * mask_float).sum(dim=1) / len_float  # (B, fused_dim)

        # Injection Composition for Layer 2
        # Input2 = H1 + Proj_local(Y) + Proj_global(Refined_G)
        inj_local = self.proj_local(y)  # (B, T, 2*hidden)
        inj_global = self.proj_global(refined_anchors).unsqueeze(1)  # (B, 1, 2*hidden)

        input2 = h1 + inj_local + inj_global

        # Layer 2
        packed_input2 = pack_padded_sequence(
            input2, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_h2, _ = self.gru2(packed_input2)
        h2, _ = pad_packed_sequence(packed_h2, batch_first=True)  # (B, T, 2*hidden)

        # --- 5. Output Head ---
        logits = self.head(h2)  # (B, T, num_classes)

        return logits
