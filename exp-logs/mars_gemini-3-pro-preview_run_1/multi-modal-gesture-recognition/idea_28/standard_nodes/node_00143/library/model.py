import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library.config import Config


class GatedConvStem(nn.Module):
    """
    Independent modality processing stream using Gated Convolution.
    Structure: Linear -> GatedConv1d (GLU) -> Dropout
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, dropout):
        super(GatedConvStem, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GLU Topology: Conv produces 2 * hidden_dim channels (A and B)
        # Padding is calculated to maintain temporal dimension: (k-1)//2
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            hidden_dim,
            hidden_dim * 2,
            kernel_size=kernel_size,
            padding=padding,
            groups=1,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, InputDim)

        # Linear Projection
        x = self.input_proj(x)  # (B, T, Hidden)

        # Transpose for Conv1d: (B, Hidden, T)
        x = x.transpose(1, 2)

        # Gated Convolution
        out = self.conv(x)  # (B, 2*Hidden, T)
        a, b = out.chunk(2, dim=1)
        x = a * torch.sigmoid(b)  # GLU

        # Transpose back: (B, T, Hidden)
        x = x.transpose(1, 2)

        x = self.dropout(x)
        return x


class BAMPNet(nn.Module):
    """
    Boundary-Aware Magnitude-Preserving Network.
    """

    def __init__(self):
        super(BAMPNet, self).__init__()

        # ==========================================
        # 1. Gated Convolutional Stems
        # ==========================================
        # Skeleton Stem
        self.skel_stem = GatedConvStem(
            input_dim=Config.INPUT_DIM_SKELETON,
            hidden_dim=Config.HIDDEN_DIM,
            kernel_size=Config.STEM_KERNEL_SIZE,
            dropout=Config.DROPOUT_RATE,
        )

        # Audio Stem
        self.audio_stem = GatedConvStem(
            input_dim=Config.INPUT_DIM_AUDIO,
            hidden_dim=Config.HIDDEN_DIM,
            kernel_size=Config.STEM_KERNEL_SIZE,
            dropout=Config.DROPOUT_RATE,
        )

        # Fusion Dimension (Concatenation)
        self.fused_dim = Config.HIDDEN_DIM * 2

        # ==========================================
        # 2. Magnitude-Preserving Gated Fusion
        # ==========================================
        # Gate weights: W_x * X + W_g * G + b
        self.gate_x_proj = nn.Linear(self.fused_dim, self.fused_dim)
        self.gate_g_proj = nn.Linear(self.fused_dim, self.fused_dim)

        # ==========================================
        # 3. Refined-Context Input-Injected Backbone
        # ==========================================
        self.rnn_hidden = Config.HIDDEN_DIM
        self.rnn_bidir = True
        self.rnn_out_dim = self.rnn_hidden * 2 if self.rnn_bidir else self.rnn_hidden

        # Layer 1
        self.gru1 = nn.GRU(
            input_size=self.fused_dim,
            hidden_size=self.rnn_hidden,
            bidirectional=self.rnn_bidir,
            batch_first=True,
        )

        # Injection Projections
        # Map original fused input (Y) to RNN output space
        self.proj_y = nn.Linear(self.fused_dim, self.rnn_out_dim)
        # Map refined global context (G_refined) to RNN output space
        self.proj_g = nn.Linear(self.fused_dim, self.rnn_out_dim)

        self.injection_dropout = nn.Dropout(Config.DROPOUT_RATE)

        # Layer 2
        self.gru2 = nn.GRU(
            input_size=self.rnn_out_dim,
            hidden_size=self.rnn_hidden,
            bidirectional=self.rnn_bidir,
            batch_first=True,
        )

        # ==========================================
        # 4. Multi-Task Output Heads
        # ==========================================
        # Classification Head (Classes + Background)
        # Output size is NUM_CLASSES + 1 because class IDs are 1-20, 0 is background.
        # We need indices 0..20, so 21 outputs.
        self.class_head = nn.Sequential(
            nn.Linear(self.rnn_out_dim, self.rnn_hidden),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(self.rnn_hidden, Config.NUM_CLASSES + 1),
        )

        # Boundary Head (Binary)
        self.boundary_head = nn.Sequential(
            nn.Linear(self.rnn_out_dim, self.rnn_hidden // 2),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(self.rnn_hidden // 2, 1),
        )

    def _masked_gap(self, x, mask):
        """
        Computes Global Average Pooling ignoring masked (padded) steps.
        x: (B, T, C)
        mask: (B, T) - True for valid, False for pad
        """
        # Expand mask to (B, T, C)
        mask_expanded = mask.unsqueeze(-1).float()

        # Zero out padded values
        x_masked = x * mask_expanded

        # Sum over time
        sum_x = x_masked.sum(dim=1)  # (B, C)

        # Count valid steps
        lengths = mask_expanded.sum(dim=1)  # (B, C)
        lengths = torch.clamp(lengths, min=1.0)  # Avoid div by zero

        return sum_x / lengths

    def forward(self, skeleton, audio, lengths, mask):
        # skeleton: (B, T, 60)
        # audio: (B, T, 13)
        # lengths: (B,)
        # mask: (B, T)

        # 1. Process Stems
        skel_feat = self.skel_stem(skeleton)  # (B, T, 256)
        audio_feat = self.audio_stem(audio)  # (B, T, 256)

        # 2. Fusion
        # Concatenate
        x_raw = torch.cat([skel_feat, audio_feat], dim=2)  # (B, T, 512)

        # Global Context (G_raw)
        g_raw = self._masked_gap(x_raw, mask)  # (B, 512)

        # Gating
        # Gate = sigmoid(Wx * X + Wg * G + b)
        # Broadcast G: (B, 1, 512)
        gate = torch.sigmoid(
            self.gate_x_proj(x_raw) + self.gate_g_proj(g_raw).unsqueeze(1)
        )

        # Apply Gate (Magnitude Preserving - No LayerNorm)
        y = x_raw * gate  # (B, T, 512)

        # 3. Backbone
        # Refined Anchor (G_refined)
        g_refined = self._masked_gap(y, mask)  # (B, 512)

        # Pack sequence for RNN
        # lengths must be on CPU for pack_padded_sequence in some versions,
        # but usually tensor is fine in newer pytorch. Safe to use .cpu()
        packed_y = pack_padded_sequence(
            y, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Layer 1
        packed_h1, _ = self.gru1(packed_y)
        h1, _ = pad_packed_sequence(packed_h1, batch_first=True)  # (B, T, 512)

        # Input Injection
        # Input_2 = Dropout(H1 + Proj(Y) + Proj(G_refined))
        # We need to handle padding in H1 (pad_packed_sequence handles it, returns 0 for pads)
        # Y is also padded with something (from stem), but we should mask operations if needed.
        # However, since we pack again, values at pad positions don't affect RNN state,
        # but might affect gradients if not careful.
        # Since we use packed sequence for Layer 2, the input values at pad positions are ignored by GRU.

        term_h1 = h1
        term_y = self.proj_y(y)
        term_g = self.proj_g(g_refined).unsqueeze(1)  # Broadcast T

        input_2 = term_h1 + term_y + term_g
        input_2 = self.injection_dropout(input_2)

        # Layer 2
        packed_input_2 = pack_padded_sequence(
            input_2, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_h2, _ = self.gru2(packed_input_2)
        h2, _ = pad_packed_sequence(packed_h2, batch_first=True)  # (B, T, 512)

        # 4. Heads
        logits_class = self.class_head(h2)  # (B, T, 21)
        logits_boundary = self.boundary_head(h2)  # (B, T, 1)

        return {"logits": logits_class, "boundary": logits_boundary}
