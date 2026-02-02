import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from library.config import HYPERPARAMS, TOTAL_CLASSES


class FeatureStem(nn.Module):
    """
    Independent processing stem for a single modality.
    Linear -> Permute -> Conv1d -> ReLU -> Dropout -> Permute
    """

    def __init__(self, input_dim, output_dim, kernel_size=7, dropout=0.3):
        super(FeatureStem, self).__init__()
        self.project = nn.Linear(input_dim, output_dim)
        # Padding to maintain temporal length: (k-1)//2
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(output_dim, output_dim, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        x = self.project(x)

        # Conv1d expects (Batch, Channel, Time)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Back to (Batch, Time, Channel)
        x = x.permute(0, 2, 1)
        return x


class GlobalConditionedGate(nn.Module):
    """
    Applies a gating mask conditioned on both local features and global context.
    Gate_t = sigmoid(Wx * Xt + Wc * C_raw + b)
    Y_t = Xt * Gate_t
    """

    def __init__(self, feature_dim):
        super(GlobalConditionedGate, self).__init__()
        self.gate_x = nn.Linear(feature_dim, feature_dim)
        self.gate_c = nn.Linear(feature_dim, feature_dim)
        self.bias = nn.Parameter(torch.zeros(feature_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        # x: (Batch, Time, Dim)
        # lengths: (Batch)

        # Compute Global Average Pooling (C_raw) masking padding
        batch_size, max_len, _ = x.size()

        # Create mask: (Batch, Time, 1)
        # lengths needs to be on same device for comparison
        mask = torch.arange(max_len, device=x.device).expand(
            batch_size, max_len
        ) < lengths.unsqueeze(1)
        mask = mask.float().unsqueeze(2)

        # Sum and divide by length (GAP)
        sum_x = torch.sum(x * mask, dim=1)  # (Batch, Dim)
        c_raw = sum_x / (lengths.unsqueeze(1).float() + 1e-8)

        # Calculate Gate
        # Wx * Xt -> (Batch, Time, Dim)
        term_x = self.gate_x(x)
        # Wc * C_raw -> (Batch, Dim) -> Broadcast to (Batch, Time, Dim)
        term_c = self.gate_c(c_raw).unsqueeze(1)

        gate = self.sigmoid(term_x + term_c + self.bias)

        # Apply Gate
        y = x * gate
        return y


class DualInjectedBiGRU(nn.Module):
    """
    2-Layer BiGRU where the second layer receives injected inputs from
    the first layer, the original cleaned input, and the refined global context.
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super(DualInjectedBiGRU, self).__init__()

        # Layer 1: Standard BiGRU
        # Bidirectional output size = hidden_dim (hidden_dim//2 per direction)
        self.gru1 = nn.GRU(
            input_dim, hidden_dim // 2, bidirectional=True, batch_first=True
        )

        # Projections for Injection
        self.proj_local = nn.Linear(input_dim, hidden_dim)
        self.proj_global = nn.Linear(input_dim, hidden_dim)

        # Layer 2: BiGRU receiving injected sum
        # Input size is hidden_dim (sum of H1 and projections)
        self.gru2 = nn.GRU(
            hidden_dim, hidden_dim // 2, bidirectional=True, batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, y, lengths):
        # y: (Batch, Time, Dim) - Gated Input

        # 1. Layer 1 Forward
        # Pack sequence for correct RNN handling
        packed_y = pack_padded_sequence(
            y, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_h1, _ = self.gru1(packed_y)
        h1, _ = pad_packed_sequence(packed_h1, batch_first=True, total_length=y.size(1))
        # h1: (Batch, Time, HiddenDim)

        # 2. Prepare Injection for Layer 2
        # Calculate G_clean = GAP(y)
        batch_size, max_len, _ = y.size()
        mask = torch.arange(max_len, device=y.device).expand(
            batch_size, max_len
        ) < lengths.unsqueeze(1)
        mask = mask.float().unsqueeze(2)

        sum_y = torch.sum(y * mask, dim=1)
        g_clean = sum_y / (lengths.unsqueeze(1).float() + 1e-8)  # (Batch, Dim)

        # Input_2 = H1 + Proj_local(Y) + Proj_global(G_clean)
        inj_local = self.proj_local(y)
        inj_global = self.proj_global(g_clean).unsqueeze(1)  # Broadcast to time

        input_2 = h1 + inj_local + inj_global
        input_2 = self.dropout(input_2)

        # 3. Layer 2 Forward
        packed_input_2 = pack_padded_sequence(
            input_2, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_h2, _ = self.gru2(packed_input_2)
        h2, _ = pad_packed_sequence(packed_h2, batch_first=True, total_length=y.size(1))

        return h2


class GCINet(nn.Module):
    """
    Global-Conditioned Input-Injected Network.
    """

    def __init__(self):
        super(GCINet, self).__init__()

        hidden_dim = HYPERPARAMS["hidden_dim"]
        dropout = HYPERPARAMS["dropout"]
        kernel_size = HYPERPARAMS["kernel_size_temporal"]

        # Input Dimensions
        skel_in = 60  # 20 joints * 3
        audio_in = HYPERPARAMS["n_mfcc"]  # 13

        # Feature Stems
        # Project each modality to half the hidden dimension
        stem_dim = hidden_dim // 2

        self.skel_stem = FeatureStem(skel_in, stem_dim, kernel_size, dropout)
        self.audio_stem = FeatureStem(audio_in, stem_dim, kernel_size, dropout)

        # Fusion Dimension (Concatenation)
        fusion_dim = stem_dim * 2

        # Global Conditioned Gate
        self.gate = GlobalConditionedGate(fusion_dim)

        # Dual Injected Backbone
        self.backbone = DualInjectedBiGRU(fusion_dim, hidden_dim, dropout)

        # Non-Linear Output Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, TOTAL_CLASSES),
        )

    def forward(self, skeleton, audio, lengths):
        # skeleton: (Batch, Time, 60)
        # audio: (Batch, Time, 13)
        # lengths: (Batch)

        # 1. Independent Feature Stems
        skel_feat = self.skel_stem(skeleton)
        audio_feat = self.audio_stem(audio)

        # 2. Fusion (Concatenation)
        x = torch.cat([skel_feat, audio_feat], dim=2)  # (Batch, Time, FusionDim)

        # 3. Global Conditioned Gating
        y = self.gate(x, lengths)

        # 4. Dual Injected Backbone
        h2 = self.backbone(y, lengths)

        # 5. Output Head
        logits = self.head(h2)  # (Batch, Time, NumClasses)

        return logits
