import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library.config import Config


class WideStem(nn.Module):
    """
    Independent modality processing stem.
    Structure: Linear -> Conv1d(k=7) -> ReLU -> Dropout.
    """

    def __init__(self, input_dim, output_dim, kernel_size=7, dropout=0.3):
        super().__init__()
        self.project = nn.Linear(input_dim, output_dim)
        # Padding ensures output length matches input length
        self.conv = nn.Conv1d(
            output_dim, output_dim, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, InputDim)
        x = self.project(x)  # (B, T, OutDim)

        # Permute for Conv1d: (B, OutDim, T)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Permute back: (B, T, OutDim)
        x = x.permute(0, 2, 1)
        return x


class MagnitudePreservingFusion(nn.Module):
    """
    Fuses modalities without normalization to preserve signal magnitude.
    Uses Masked GAP for global context gating.
    """

    def __init__(self, dim1, dim2):
        super().__init__()
        self.concat_dim = dim1 + dim2

        # Gating weights
        self.gate_x = nn.Linear(self.concat_dim, self.concat_dim)
        self.gate_g = nn.Linear(self.concat_dim, self.concat_dim)

    def forward(self, x1, x2, lengths):
        # x1: (B, T, D1), x2: (B, T, D2)

        # 1. Concatenate
        x_raw = torch.cat([x1, x2], dim=2)  # (B, T, D1+D2)

        # 2. Masked Global Average Pooling (G_raw)
        # Create mask: (B, T, 1)
        max_len = x_raw.size(1)
        mask = torch.arange(max_len, device=x_raw.device)[None, :] < lengths[:, None]
        mask = mask.float().unsqueeze(2)  # (B, T, 1)

        # Sum over time, divide by length
        sum_pooled = torch.sum(x_raw * mask, dim=1)  # (B, D)
        # Avoid division by zero
        len_safe = lengths.float().unsqueeze(1).clamp(min=1.0)
        g_raw = sum_pooled / len_safe  # (B, D)

        # 3. Conditioned Gating
        # Gate = Sigmoid(W_x * X_raw + W_g * G_raw + b)
        # W_g * G_raw needs broadcasting to (B, T, D)
        gate_feat = self.gate_x(x_raw) + self.gate_g(g_raw).unsqueeze(1)
        gate = torch.sigmoid(gate_feat)

        # 4. Apply Gate (Magnitude Preserving: No LayerNorm)
        y = x_raw * gate

        return y


class InputInjectedBiGRU(nn.Module):
    """
    2-Layer BiGRU with explicit input injection and vertical dropout.
    Layer 2 Input = Dropout(H1 + Proj(Y) + Proj(G_refined))
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim * 2  # Bidirectional

        # Layer 1
        self.gru1 = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Projections for Injection
        # We project Y (input_dim) and G_refined (input_dim) to match GRU output (hidden*2)
        self.proj_y = nn.Linear(input_dim, self.output_dim)
        self.proj_g = nn.Linear(input_dim, self.output_dim)

        # Vertical Dropout (applied after summation)
        self.dropout = nn.Dropout(dropout)

        # Layer 2
        # Input to layer 2 is size output_dim (from layer 1)
        self.gru2 = nn.GRU(
            self.output_dim, hidden_dim, batch_first=True, bidirectional=True
        )

    def forward(self, y, lengths):
        # y: (B, T, InputDim)

        # 1. Refined Anchor (Masked GAP of Y)
        max_len = y.size(1)
        mask = torch.arange(max_len, device=y.device)[None, :] < lengths[:, None]
        mask = mask.float().unsqueeze(2)

        sum_pooled = torch.sum(y * mask, dim=1)
        len_safe = lengths.float().unsqueeze(1).clamp(min=1.0)
        g_refined = sum_pooled / len_safe  # (B, InputDim)

        # 2. Layer 1
        packed_input = pack_padded_sequence(
            y, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_h1, _ = self.gru1(packed_input)
        h1, _ = pad_packed_sequence(
            packed_h1, batch_first=True, total_length=max_len
        )  # (B, T, H*2)

        # 3. Injection Logic
        # Input_2 = Dropout(H_1 + Proj(Y) + Proj(G_refined))
        proj_y_seq = self.proj_y(y)  # (B, T, H*2)
        proj_g_vec = self.proj_g(g_refined).unsqueeze(1)  # (B, 1, H*2)

        # Summation
        in2 = h1 + proj_y_seq + proj_g_vec

        # Vertical Dropout
        in2 = self.dropout(in2)

        # 4. Layer 2
        packed_in2 = pack_padded_sequence(
            in2, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru2(packed_in2)
        output, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=max_len
        )

        return output


class BS_MPII(nn.Module):
    """
    Boundary-Supervised Magnitude-Preserving Input-Injected Network.
    """

    def __init__(self):
        super().__init__()

        # Dimensions
        self.skel_input_dim = Config.SKELETON_JOINTS * Config.SKELETON_CHANNELS
        self.audio_input_dim = Config.AUDIO_N_MFCC
        self.stem_dim = Config.STEM_CHANNELS

        # 1. Wide Independent Stems
        self.skel_stem = WideStem(
            self.skel_input_dim,
            self.stem_dim,
            kernel_size=Config.STEM_KERNEL_SIZE,
            dropout=0.3,
        )
        self.audio_stem = WideStem(
            self.audio_input_dim,
            self.stem_dim,
            kernel_size=Config.STEM_KERNEL_SIZE,
            dropout=0.3,
        )

        # 2. Magnitude-Preserving Gated Fusion
        self.fusion = MagnitudePreservingFusion(self.stem_dim, self.stem_dim)
        self.fused_dim = self.stem_dim * 2

        # 3. Regularized Input-Injected Backbone
        self.backbone = InputInjectedBiGRU(
            self.fused_dim, Config.HIDDEN_DIM, dropout=Config.DROPOUT
        )
        self.backbone_out_dim = Config.HIDDEN_DIM * 2

        # 4. Multi-Task Output Heads
        # Classification Head
        self.class_head = nn.Sequential(
            nn.Linear(self.backbone_out_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(Config.HIDDEN_DIM, Config.NUM_CLASSES),
        )

        # Boundary Head (Binary Classification)
        self.boundary_head = nn.Sequential(
            nn.Linear(self.backbone_out_dim, Config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(Config.HIDDEN_DIM // 2, 1),  # Logits for BCEWithLogitsLoss
        )

    def forward(self, skeleton, audio, lengths):
        # skeleton: (B, T, J, 3) -> Flatten to (B, T, J*3)
        B, T, J, C = skeleton.shape
        skel_flat = skeleton.view(B, T, J * C)

        # 1. Stems
        skel_feat = self.skel_stem(skel_flat)
        audio_feat = self.audio_stem(audio)

        # 2. Fusion
        fused = self.fusion(skel_feat, audio_feat, lengths)

        # 3. Backbone
        backbone_out = self.backbone(fused, lengths)

        # 4. Heads
        class_logits = self.class_head(backbone_out)  # (B, T, NumClasses)
        boundary_logits = self.boundary_head(backbone_out).squeeze(-1)  # (B, T)

        return {"class_logits": class_logits, "boundary_logits": boundary_logits}
