import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library.config import Config


class InputStem(nn.Module):
    """
    Decoupled input processing stem for a single modality.
    Structure: Linear -> Permute -> Conv1d(k=7) -> Permute -> ReLU -> Dropout
    """

    def __init__(self, input_dim, embed_dim, kernel_size, dropout):
        super().__init__()
        self.fc = nn.Linear(input_dim, embed_dim)
        # Temporal Convolution with padding to maintain length
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        x = self.fc(x)  # (B, T, EmbedDim)

        # Permute for Conv1d: (B, EmbedDim, T)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # Back to (B, T, EmbedDim)

        x = self.act(x)
        x = self.dropout(x)
        return x


class ContextGating(nn.Module):
    """
    Context Gating Mechanism: Y = X * Sigmoid(W*X + b)
    Acts as a learnable gate to filter noise.
    """

    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, T, Dim)
        gate = torch.sigmoid(self.fc(x))
        return x * gate


class GCA_IIN(nn.Module):
    """
    Global-Context Anchored Input-Injected Network.
    Features:
    - Decoupled Input Stems
    - Gated Fusion
    - Global Anchor Extraction (Global Average Pooling)
    - Dual-Injected BiGRU Backbone (Layer 1 -> Injection -> Layer 2)
    """

    def __init__(self):
        super().__init__()

        # 1. Decoupled Input Stems
        self.skel_stem = InputStem(
            input_dim=Config.INPUT_DIM_SKELETON,
            embed_dim=Config.SKELETON_EMBED_DIM,
            kernel_size=Config.KERNEL_SIZE,
            dropout=Config.DROPOUT,
        )

        self.audio_stem = InputStem(
            input_dim=Config.INPUT_DIM_AUDIO,
            embed_dim=Config.AUDIO_EMBED_DIM,
            kernel_size=Config.KERNEL_SIZE,
            dropout=Config.DROPOUT,
        )

        # Fusion Components
        self.fused_dim = Config.SKELETON_EMBED_DIM + Config.AUDIO_EMBED_DIM
        self.fusion_norm = nn.LayerNorm(self.fused_dim)
        self.context_gating = ContextGating(self.fused_dim)

        # 2. Dual-Injected Recurrent Backbone
        self.hidden_dim = Config.HIDDEN_DIM
        self.bidirectional = Config.BIDIRECTIONAL
        self.num_directions = 2 if self.bidirectional else 1
        self.rnn_output_dim = self.hidden_dim * self.num_directions

        # Layer 1: Processes the fused local context
        self.gru1 = nn.GRU(
            input_size=self.fused_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        # Injection Projections
        # We project Y (Local) and G (Global) to match the RNN hidden state dimension
        self.proj_local = nn.Linear(self.fused_dim, self.rnn_output_dim)
        self.proj_global = nn.Linear(self.fused_dim, self.rnn_output_dim)

        # Layer 2: Processes the composite injected input
        # Input dim is rnn_output_dim because we sum H1 + Proj(Y) + Proj(G)
        self.gru2 = nn.GRU(
            input_size=self.rnn_output_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        # 3. Output Head
        self.head = nn.Sequential(
            nn.Linear(self.rnn_output_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.hidden_dim, Config.NUM_CLASSES),
        )

    def forward(self, skeleton, audio, lengths):
        """
        Args:
            skeleton: (Batch, Time, SkelDim)
            audio: (Batch, Time, AudioDim)
            lengths: (Batch,) Tensor containing valid sequence lengths
        Returns:
            logits: (Batch, Time, NumClasses)
        """
        # --- 1. Input Stems ---
        skel_feat = self.skel_stem(skeleton)
        audio_feat = self.audio_stem(audio)

        # --- 2. Fusion & Gating ---
        # Concatenate along feature dimension
        y = torch.cat([skel_feat, audio_feat], dim=-1)  # (B, T, FusedDim)
        y = self.fusion_norm(y)
        y = self.context_gating(y)

        # --- 3. Global Anchor Extraction ---
        # Compute Global Average Pooling while ignoring padding
        device = y.device
        max_len = y.size(1)

        # Create mask: (B, T) - True for valid frames, False for padding
        mask = torch.arange(max_len, device=device).expand(
            len(lengths), max_len
        ) < lengths.unsqueeze(1)
        mask = mask.float().unsqueeze(-1)  # (B, T, 1)

        # Sum valid frames and divide by length
        sum_y = (y * mask).sum(dim=1)  # (B, FusedDim)
        g = sum_y / (lengths.unsqueeze(1).float() + 1e-8)  # (B, FusedDim)

        # --- 4. Backbone Layer 1 ---
        # Pack sequence for efficient RNN processing
        packed_y = pack_padded_sequence(
            y, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_h1, _ = self.gru1(packed_y)

        # Unpack to get full sequence for element-wise addition
        h1, _ = pad_packed_sequence(
            packed_h1, batch_first=True, total_length=max_len
        )  # (B, T, H*2)

        # --- 5. Dual Injection ---
        # Construct Input for Layer 2: H1 + Proj(Local) + Proj(Global)
        p_local = self.proj_local(y)  # (B, T, H*2)
        p_global = self.proj_global(g).unsqueeze(
            1
        )  # (B, 1, H*2) - Broadcast across time

        # Element-wise addition (Broadcasting handles p_global)
        input_2 = h1 + p_local + p_global

        # --- 6. Backbone Layer 2 ---
        packed_input_2 = pack_padded_sequence(
            input_2, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_h2, _ = self.gru2(packed_input_2)

        # Unpack final hidden states
        h2, _ = pad_packed_sequence(packed_h2, batch_first=True, total_length=max_len)

        # --- 7. Output Head ---
        logits = self.head(h2)  # (B, T, NumClasses)

        return logits
