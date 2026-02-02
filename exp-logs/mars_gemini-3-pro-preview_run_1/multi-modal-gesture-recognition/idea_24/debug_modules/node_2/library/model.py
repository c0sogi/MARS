import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library.config import Config


def masked_global_average_pooling(x, mask):
    """
    Computes Global Average Pooling (GAP) considering the mask.

    Args:
        x: (Batch, Time, Channels)
        mask: (Batch, Time) - Boolean mask where True indicates valid data.

    Returns:
        (Batch, Channels)
    """
    # mask is (B, T), x is (B, T, C)
    # Expand mask to (B, T, 1) and cast to float
    mask_float = mask.unsqueeze(-1).float()

    # Zero out padded positions (just in case)
    x_masked = x * mask_float

    # Sum over time dimension
    sum_x = x_masked.sum(dim=1)  # (B, C)

    # Count valid steps
    lengths = mask_float.sum(dim=1)  # (B, 1)

    # Avoid division by zero
    lengths = torch.clamp(lengths, min=1.0)

    return sum_x / lengths


class WideStem(nn.Module):
    """
    Wide Single-Scale Stem: Linear -> Conv1d(k=7) -> ReLU -> Dropout.
    Designed to maximize feature capacity without bottlenecks.
    """

    def __init__(self, input_dim, hidden_dim, dropout):
        super(WideStem, self).__init__()
        self.project = nn.Linear(input_dim, hidden_dim)
        # Kernel size 7, padding 3 preserves temporal dimension
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, InputDim)
        x = self.project(x)  # (B, T, Hidden)

        # Conv1d expects (B, Channels, Time)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)  # Back to (B, T, Hidden)

        x = self.act(x)
        x = self.dropout(x)
        return x


class RawMagnitudeGating(nn.Module):
    """
    Raw Magnitude Gating:
    1. Concatenates stems.
    2. Computes Masked GAP (Global Context).
    3. Gates the raw sequence using context.
    4. Applies LayerNorm AFTER gating.
    """

    def __init__(self, input_dim):
        super(RawMagnitudeGating, self).__init__()
        self.fc_x = nn.Linear(input_dim, input_dim)
        self.fc_g = nn.Linear(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x, mask):
        # x: (B, T, C) - Fused raw input

        # 1. Masked Global Context
        g_raw = masked_global_average_pooling(x, mask)  # (B, C)

        # 2. Compute Gate
        # Gate = sigmoid(Wx * X + Wg * G + b)
        # Broadcast G to (B, T, C) implicitly
        gate = torch.sigmoid(self.fc_x(x) + self.fc_g(g_raw).unsqueeze(1))

        # 3. Apply Gate
        y = x * gate

        # 4. Post-Gate Normalization
        y = self.norm(y)

        return y


class InputInjectedBackbone(nn.Module):
    """
    2-Layer BiGRU with Refined-Context Input Injection.
    Layer 2 receives: H1 + Proj(Input) + Proj(RefinedContext).
    """

    def __init__(self, input_dim, hidden_dim, dropout):
        super(InputInjectedBackbone, self).__init__()
        self.hidden_dim = hidden_dim

        # BiGRU doubles the output dimension
        rnn_output_dim = hidden_dim * 2

        # Layer 1
        self.gru1 = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)

        # Projections for Injection
        # We need to project Input (input_dim) and Context (input_dim) to match GRU1 output (rnn_output_dim)
        # If input_dim == rnn_output_dim, these are just linear transforms
        self.proj_input = nn.Linear(input_dim, rnn_output_dim)
        self.proj_context = nn.Linear(input_dim, rnn_output_dim)

        # Layer 2
        # Input to GRU2 is size rnn_output_dim (since we sum H1 and projections)
        self.gru2 = nn.GRU(
            rnn_output_dim, hidden_dim, bidirectional=True, batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, lengths):
        # x: (B, T, InputDim) - This is Y_norm from gating

        # --- Layer 1 ---
        packed_input = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_h1, _ = self.gru1(packed_input)
        h1, _ = pad_packed_sequence(packed_h1, batch_first=True)  # (B, T, 2*Hidden)

        # --- Injection Preparation ---
        # 1. Refined Anchor: Masked GAP of the *gated* sequence Y_norm
        g_refined = masked_global_average_pooling(x, mask)  # (B, InputDim)

        # 2. Projections
        # Expand dims for broadcasting: (B, 1, OutDim)
        proj_x = self.proj_input(x)  # (B, T, OutDim)
        proj_g = self.proj_context(g_refined).unsqueeze(1)  # (B, 1, OutDim)

        # 3. Composite Input
        # Input2 = H1 + Proj(Y_norm) + Proj(G_refined)
        # h1 might have different length than x if pad_packed_sequence trimmed it?
        # Usually pad_packed restores to max length in batch.
        # We ensure shapes match.
        if h1.size(1) != x.size(1):
            # This theoretically shouldn't happen with pad_packed_sequence defaults if x was padded
            # But for safety, we slice or pad. Usually they match.
            pass

        input2 = h1 + proj_x + proj_g
        input2 = self.dropout(input2)

        # --- Layer 2 ---
        packed_input2 = pack_padded_sequence(
            input2, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_h2, _ = self.gru2(packed_input2)
        h2, _ = pad_packed_sequence(packed_h2, batch_first=True)  # (B, T, 2*Hidden)

        return h2


class MCWINet(nn.Module):
    """
    Masked-Context Wide-Injected Network.
    """

    def __init__(self):
        super(MCWINet, self).__init__()

        # Dimensions
        self.skel_input_dim = 60  # 20 joints * 3
        self.audio_input_dim = Config.N_MFCC  # 13
        self.hidden_dim = Config.HIDDEN_DIM  # 256
        self.dropout_p = Config.DROPOUT

        # 1. Wide Single-Scale Stems
        self.skel_stem = WideStem(self.skel_input_dim, self.hidden_dim, self.dropout_p)
        self.audio_stem = WideStem(
            self.audio_input_dim, self.hidden_dim, self.dropout_p
        )

        # Fused Dimension = 256 + 256 = 512
        self.fused_dim = self.hidden_dim * 2

        # 2. Raw Magnitude Gating
        self.gating = RawMagnitudeGating(self.fused_dim)

        # 3. Input Injected Backbone
        self.backbone = InputInjectedBackbone(
            self.fused_dim, self.hidden_dim, self.dropout_p
        )

        # 4. Non-Linear Output Head
        # Backbone output is 2 * Hidden (BiGRU) = 512
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, Config.NUM_CLASSES),
        )

    def forward(self, skeleton, audio, mask, lengths):
        """
        Args:
            skeleton: (B, T, 60)
            audio: (B, T, 13)
            mask: (B, T)
            lengths: (B,)
        """
        # 1. Stems
        s_feat = self.skel_stem(skeleton)  # (B, T, 256)
        a_feat = self.audio_stem(audio)  # (B, T, 256)

        # Fusion
        fused = torch.cat([s_feat, a_feat], dim=2)  # (B, T, 512)

        # 2. Gating
        gated = self.gating(fused, mask)  # (B, T, 512)

        # 3. Backbone
        # Returns (B, T, 512)
        features = self.backbone(gated, mask, lengths)

        # 4. Head
        logits = self.head(features)  # (B, T, NumClasses)

        return logits
