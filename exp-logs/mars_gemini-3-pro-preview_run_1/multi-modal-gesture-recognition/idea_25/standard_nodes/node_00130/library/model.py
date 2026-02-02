import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library.config import Config


class WideStem(nn.Module):
    """
    Wide Independent Stem for modality processing.
    Structure: Linear -> Conv1d(k=7, c=256) -> ReLU -> Dropout
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, dropout_rate):
        super(WideStem, self).__init__()
        # Initial projection to hidden_dim
        self.linear = nn.Linear(input_dim, hidden_dim)
        # Wide convolution
        # Padding = kernel_size // 2 to maintain temporal length
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (Batch, Time, InputDim)

        # Linear Projection
        x = self.linear(x)  # (B, T, Hidden)

        # Prepare for Conv1d: (B, Hidden, T)
        x = x.permute(0, 2, 1)

        # Conv1d
        x = self.conv(x)

        # Back to (B, T, Hidden)
        x = x.permute(0, 2, 1)

        # Activation & Dropout
        x = self.relu(x)
        x = self.dropout(x)

        return x


class MagnitudePreservingFusion(nn.Module):
    """
    Magnitude-Preserving Gated Fusion.
    Concatenates stems, computes masked GAP, and applies gating without normalization.
    """

    def __init__(self, input_dim):
        super(MagnitudePreservingFusion, self).__init__()

        # Gating weights
        # Gate = Sigmoid(Wx * X + Wg * G + b)
        self.wx = nn.Linear(input_dim, input_dim)
        self.wg = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_skel, x_audio, mask):
        # x_skel, x_audio: (B, T, Hidden)
        # mask: (B, T) - 1 for valid, 0 for pad

        # 1. Concatenate
        x_raw = torch.cat([x_skel, x_audio], dim=-1)  # (B, T, 2*Hidden)

        # 2. Masked Global Average Pooling (GAP)
        # Expand mask for element-wise multiplication: (B, T, 1)
        mask_expanded = mask.unsqueeze(-1).float()

        # Sum valid time steps
        sum_pooled = torch.sum(x_raw * mask_expanded, dim=1)  # (B, 2*Hidden)

        # Count valid time steps (avoid div by zero)
        lengths = torch.sum(mask_expanded, dim=1)
        lengths = torch.clamp(lengths, min=1.0)

        g_raw = sum_pooled / lengths  # (B, 2*Hidden)

        # 3. Gating
        # Project sequence and context
        gate_seq = self.wx(x_raw)  # (B, T, 2*Hidden)
        gate_ctx = self.wg(g_raw).unsqueeze(1)  # (B, 1, 2*Hidden)

        gate = self.sigmoid(gate_seq + gate_ctx)

        # 4. Apply Gate
        y = x_raw * gate

        return y


class InputInjectedBiGRU(nn.Module):
    """
    Refined-Context Input-Injected Backbone.
    2-layer BiGRU with injection mechanism between layers.
    """

    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(InputInjectedBiGRU, self).__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1

        # Layer 1
        self.gru1 = nn.GRU(
            input_dim, hidden_dim, batch_first=True, bidirectional=self.bidirectional
        )

        # Projections for Injection
        # Layer 1 output is (B, T, 2*Hidden)
        # Y is (B, T, InputDim) -> Project to 2*Hidden
        # G_refined is (B, InputDim) -> Project to 2*Hidden
        # Layer 2 Input size = 2*Hidden (from H1)
        # We sum H1 + Proj(Y) + Proj(G), so dimensions must match H1 output

        gru1_out_dim = hidden_dim * self.num_directions
        self.proj_y = nn.Linear(input_dim, gru1_out_dim)
        self.proj_g = nn.Linear(input_dim, gru1_out_dim)

        # Layer 2
        # Input size matches H1 output size because of the sum injection
        self.gru2 = nn.GRU(
            gru1_out_dim, hidden_dim, batch_first=True, bidirectional=self.bidirectional
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, y, lengths, mask):
        # y: (B, T, InputDim) - Fused input
        # lengths: (B,) CPU tensor for pack_padded_sequence
        # mask: (B, T)

        # ---------------------
        # Layer 1
        # ---------------------
        packed_input = pack_padded_sequence(
            y, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_h1, _ = self.gru1(packed_input)
        h1, _ = pad_packed_sequence(packed_h1, batch_first=True)
        # h1: (B, T, 2*Hidden) assuming sorted/unsorted handled by pad_packed

        # ---------------------
        # Refined Anchor (Masked GAP of Y)
        # ---------------------
        mask_expanded = mask.unsqueeze(-1).float()
        sum_pooled = torch.sum(y * mask_expanded, dim=1)
        len_denom = torch.clamp(torch.sum(mask_expanded, dim=1), min=1.0)
        g_refined = sum_pooled / len_denom  # (B, InputDim)

        # ---------------------
        # Injection
        # ---------------------
        # Input_2 = H1 + Proj(Y) + Proj(G_refined)
        proj_y_seq = self.proj_y(y)  # (B, T, 2*Hidden)
        proj_g_ctx = self.proj_g(g_refined).unsqueeze(1)  # (B, 1, 2*Hidden)

        # Ensure h1 length matches y length (pad_packed restores original length)
        # Just in case of minor mismatches due to packing logic, usually safe.
        input_2 = h1 + proj_y_seq + proj_g_ctx

        # Apply Dropout before Layer 2
        input_2 = self.dropout(input_2)

        # ---------------------
        # Layer 2
        # ---------------------
        packed_input_2 = pack_padded_sequence(
            input_2, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_h2, _ = self.gru2(packed_input_2)
        h2, _ = pad_packed_sequence(packed_h2, batch_first=True)

        return h2


class MPWINet(nn.Module):
    """
    Magnitude-Preserving Wide-Injected Network (MPWI-Net).
    """

    def __init__(self):
        super(MPWINet, self).__init__()

        # Dimensions
        self.skel_input_dim = Config.SKELETON_JOINTS * Config.SKELETON_CHANNELS
        self.audio_input_dim = Config.MFCC_N_MFCC
        self.hidden_dim = Config.HIDDEN_DIM

        # 1. Wide Stems
        self.skel_stem = WideStem(
            self.skel_input_dim,
            self.hidden_dim,
            Config.KERNEL_SIZE,
            Config.DROPOUT_RATE,
        )
        self.audio_stem = WideStem(
            self.audio_input_dim,
            self.hidden_dim,
            Config.KERNEL_SIZE,
            Config.DROPOUT_RATE,
        )

        # 2. Fusion
        # Concatenation of two stems -> 2 * hidden_dim
        fusion_dim = self.hidden_dim * 2
        self.fusion = MagnitudePreservingFusion(fusion_dim)

        # 3. Backbone
        self.backbone = InputInjectedBiGRU(
            fusion_dim, self.hidden_dim, Config.DROPOUT_RATE
        )

        # 4. Output Head
        # Backbone output is 2 * hidden_dim (BiGRU)
        backbone_out_dim = self.hidden_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(backbone_out_dim, backbone_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(backbone_out_dim // 2, Config.NUM_CLASSES),
        )

    def forward(self, skeleton, audio, lengths, mask):
        """
        Args:
            skeleton: (B, T, J, 3)
            audio: (B, T, F)
            lengths: (B,)
            mask: (B, T)
        """
        # Flatten Skeleton: (B, T, J*3)
        B, T, J, C = skeleton.shape
        skel_flat = skeleton.view(B, T, J * C)

        # 1. Stems
        skel_feat = self.skel_stem(skel_flat)  # (B, T, H)
        audio_feat = self.audio_stem(audio)  # (B, T, H)

        # 2. Fusion
        fused_feat = self.fusion(skel_feat, audio_feat, mask)  # (B, T, 2H)

        # 3. Backbone
        backbone_out = self.backbone(fused_feat, lengths, mask)  # (B, T, 2H)

        # 4. Classifier
        logits = self.classifier(backbone_out)  # (B, T, NumClasses)

        return logits
