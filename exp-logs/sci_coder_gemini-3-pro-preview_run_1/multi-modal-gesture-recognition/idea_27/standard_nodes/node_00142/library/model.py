import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library.config import Config


class WideStem(nn.Module):
    """
    Wide Independent Stem for modality processing.
    Structure: Linear -> Permute -> Conv1d(k=7) -> ReLU -> Dropout.
    """

    def __init__(self, input_dim, output_dim, kernel_size=7, dropout=0.3):
        super(WideStem, self).__init__()
        self.project = nn.Linear(input_dim, output_dim)
        # Padding to keep length consistent: (k-1)/2
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            output_dim, output_dim, kernel_size=kernel_size, padding=padding
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Time, InputDim)
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
    Fuses modalities preserving magnitude (no LayerNorm).
    Includes Masked GAP and Gating.
    """

    def __init__(self, input_dim):
        super(MagnitudePreservingFusion, self).__init__()

        # Gating weights
        # Gate = sigmoid(Wx * X + Wg * G + b)
        self.fc_x = nn.Linear(input_dim, input_dim)
        self.fc_g = nn.Linear(input_dim, input_dim)

    def forward(self, x_skel, x_audio, mask):
        """
        Args:
            x_skel: (B, T, DimSkel)
            x_audio: (B, T, DimAudio)
            mask: (B, T) Boolean mask where True is valid data, False is padding.
        """
        # 1. Concatenate
        # x_raw: (B, T, DimTotal)
        x_raw = torch.cat([x_skel, x_audio], dim=-1)

        # 2. Masked Global Average Pooling (GAP)
        # mask: (B, T) -> (B, T, 1)
        mask_expanded = mask.unsqueeze(-1).float()

        # Sum valid time steps
        sum_features = torch.sum(x_raw * mask_expanded, dim=1)  # (B, DimTotal)
        sum_mask = torch.sum(mask_expanded, dim=1)  # (B, 1)

        # Avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1.0)
        g_raw = sum_features / sum_mask  # (B, DimTotal)

        # 3. Gating Mechanism
        # Gate_t = sigmoid(Wx(X_t) + Wg(G_raw))
        # Broadcast G_raw to (B, T, DimTotal)
        gate_input = self.fc_x(x_raw) + self.fc_g(g_raw).unsqueeze(1)
        gate = torch.sigmoid(gate_input)

        # 4. Apply Gate
        y = x_raw * gate

        return y


class InputInjectedBiGRU(nn.Module):
    """
    2-Layer BiGRU with Regularized Input Injection.
    Layer 2 Input = Dropout(Layer1_Output + Proj(Input) + Proj(GlobalContext))
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super(InputInjectedBiGRU, self).__init__()
        self.hidden_dim = hidden_dim

        # Layer 1
        self.gru1 = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Injection Projections
        # Layer 1 output dim is 2 * hidden_dim
        self.proj_input = nn.Linear(input_dim, 2 * hidden_dim)
        self.proj_context = nn.Linear(input_dim, 2 * hidden_dim)

        # Vertical Dropout
        self.dropout = nn.Dropout(dropout)

        # Layer 2
        # Input to Layer 2 is size 2 * hidden_dim
        self.gru2 = nn.GRU(
            2 * hidden_dim, hidden_dim, batch_first=True, bidirectional=True
        )

    def forward(self, x, mask, lengths):
        """
        Args:
            x: (B, T, InputDim) - The gated fused features
            mask: (B, T) - Valid mask
            lengths: (B) - Sequence lengths (CPU tensor)
        """
        # 1. Compute Refined Anchor (Masked GAP of input x)
        mask_expanded = mask.unsqueeze(-1).float()
        sum_features = torch.sum(x * mask_expanded, dim=1)
        sum_mask = torch.clamp(torch.sum(mask_expanded, dim=1), min=1.0)
        g_refined = sum_features / sum_mask  # (B, InputDim)

        # 2. Layer 1 Execution
        packed_input = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_h1, _ = self.gru1(packed_input)
        h1, _ = pad_packed_sequence(packed_h1, batch_first=True)  # (B, T, 2*Hidden)

        # 3. Injection & Regularization
        # Sum_2 = H1 + Proj(X) + Proj(G_refined)
        # Note: h1 may have different padded length than x if pad_packed_sequence trims it,
        # but usually with batch_first=True and total_length argument or default behavior it matches max length in batch.
        # To be safe, we ensure alignment.

        # Align shapes if necessary (usually not needed if lengths are correct)
        if h1.size(1) != x.size(1):
            # This happens if the batch max length < tensor max length
            # We slice x to match h1
            x = x[:, : h1.size(1), :]
            mask = mask[:, : h1.size(1)]

        proj_x = self.proj_input(x)  # (B, T, 2*Hidden)
        proj_g = self.proj_context(g_refined).unsqueeze(1)  # (B, 1, 2*Hidden)

        sum_2 = h1 + proj_x + proj_g
        input_2 = self.dropout(sum_2)

        # 4. Layer 2 Execution
        packed_input_2 = pack_padded_sequence(
            input_2, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_h2, _ = self.gru2(packed_input_2)
        h2, _ = pad_packed_sequence(packed_h2, batch_first=True)  # (B, T, 2*Hidden)

        return h2


class MPCNet(nn.Module):
    """
    Magnitude-Preserving Context-Injected Network.
    """

    def __init__(self):
        super(MPCNet, self).__init__()

        # --- 1. Wide Stems ---
        self.skel_stem = WideStem(
            input_dim=Config.SKELETON_CHANNELS,
            output_dim=Config.STEM_CHANNELS,
            kernel_size=Config.STEM_KERNEL_SIZE,
            dropout=Config.DROPOUT_RATE,
        )

        self.audio_stem = WideStem(
            input_dim=Config.N_MFCC,
            output_dim=Config.STEM_CHANNELS,
            kernel_size=Config.STEM_KERNEL_SIZE,
            dropout=Config.DROPOUT_RATE,
        )

        # --- 2. Fusion ---
        fusion_dim = Config.STEM_CHANNELS * 2
        self.fusion = MagnitudePreservingFusion(input_dim=fusion_dim)

        # --- 3. Backbone ---
        self.backbone = InputInjectedBiGRU(
            input_dim=fusion_dim,
            hidden_dim=Config.BACKBONE_HIDDEN_DIM,
            dropout=Config.DROPOUT_RATE,
        )

        # --- 4. Output Head ---
        # Input: 2 * Hidden (BiGRU output)
        head_input_dim = Config.BACKBONE_HIDDEN_DIM * 2
        self.classifier = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(head_input_dim // 2, Config.NUM_CLASSES),
        )

    def forward(self, skeleton, audio, length):
        """
        Args:
            skeleton: (B, T, 60)
            audio: (B, T, 20)
            length: (B)
        """
        # Create mask based on length
        # (B, T)
        max_len = skeleton.size(1)
        batch_size = skeleton.size(0)

        # Create range (0, 1, ..., max_len-1)
        # Expand to (B, T) and compare with length
        indices = (
            torch.arange(max_len, device=skeleton.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        mask = indices < length.unsqueeze(1)

        # 1. Process Stems
        skel_feat = self.skel_stem(skeleton)
        audio_feat = self.audio_stem(audio)

        # 2. Fusion
        fused_feat = self.fusion(skel_feat, audio_feat, mask)

        # 3. Backbone
        # Returns (B, T, 2*Hidden)
        backbone_out = self.backbone(fused_feat, mask, length)

        # 4. Classification
        logits = self.classifier(backbone_out)

        return logits
