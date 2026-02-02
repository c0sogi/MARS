import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library.config import INPUT_DIM, HIDDEN_DIM, NUM_CLASSES


class MaskedSE(nn.Module):
    """
    Masked Squeeze-and-Excitation Module.
    Computes global channel descriptors using the sequence mask to exclude padding.
    """

    def __init__(self, channels, reduction=16):
        super(MaskedSE, self).__init__()
        reduced_channels = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, mask):
        """
        Args:
            x: (B, C, T) Feature map
            mask: (B, T) Binary mask (1 for valid, 0 for padding)
        """
        # Mask input features to ensure padding doesn't contribute
        mask_expanded = mask.unsqueeze(1)  # (B, 1, T)
        x_masked = x * mask_expanded

        # Global Average Pooling over valid tokens only
        # Sum over time dimension
        x_sum = x_masked.sum(dim=2)  # (B, C)

        # Count valid tokens
        valid_counts = mask.sum(dim=1).clamp(min=1).unsqueeze(1)  # (B, 1)

        # Average
        x_avg = x_sum / valid_counts  # (B, C)

        # Excitation
        scale = self.fc(x_avg).unsqueeze(2)  # (B, C, 1)

        return x * scale


class GatedResidualBlock(nn.Module):
    """
    Gated Residual Block with Masked SE and 1x1 Projection.
    """

    def __init__(self, channels, kernel_size, dilation):
        super(GatedResidualBlock, self).__init__()

        self.conv_dilated = nn.Conv1d(
            channels, 2 * channels, kernel_size, padding=dilation, dilation=dilation
        )

        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.se = MaskedSE(channels)

    def forward(self, x, mask):
        residual = x

        # Dilated Convolution
        out = self.conv_dilated(x)

        # Gated Activation
        P, Q = out.chunk(2, dim=1)
        Z = torch.tanh(P) * torch.sigmoid(Q)

        # 1x1 Projection (Strictly retained)
        H = self.conv_1x1(Z)

        # Masked Squeeze-and-Excitation
        H_prime = self.se(H, mask)

        # Residual Connection
        return residual + H_prime


class SingleStageTCN(nn.Module):
    """
    Refinement Stage using a stack of GatedResidualBlocks.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=10):
        super(SingleStageTCN, self).__init__()

        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)

        self.layers = nn.ModuleList(
            [
                GatedResidualBlock(hidden_dim, kernel_size=3, dilation=2**i)
                for i in range(num_layers)
            ]
        )

        self.output_proj = nn.Conv1d(hidden_dim, output_dim, 1)

    def forward(self, x, mask):
        # x: (B, C, T)
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x, mask)

        out = self.output_proj(x)
        return out


class MSE_GCN(nn.Module):
    """
    Masked Squeeze-and-Excitation Gated-Cascaded Network.
    Stage 1: Bi-LSTM Encoder
    Stage 2: Masked SE Gated Refinement
    Stage 3: Masked SE Sharpening
    """

    def __init__(self):
        super(MSE_GCN, self).__init__()

        # ================= Stage 1: Encoder =================
        self.embedding = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.lstm = nn.LSTM(
            HIDDEN_DIM, HIDDEN_DIM, num_layers=2, batch_first=True, bidirectional=True
        )

        # Heads for Stage 1
        self.s1_cls = nn.Linear(HIDDEN_DIM * 2, NUM_CLASSES)
        self.s1_bnd = nn.Linear(HIDDEN_DIM * 2, 1)

        # ================= Stage 2: Refinement =================
        # Input: Probabilities from Stage 1 (NUM_CLASSES + 1)
        self.stage2 = SingleStageTCN(NUM_CLASSES + 1, HIDDEN_DIM, NUM_CLASSES + 1)

        # ================= Stage 3: Sharpening =================
        self.stage3 = SingleStageTCN(NUM_CLASSES + 1, HIDDEN_DIM, NUM_CLASSES + 1)

    def forward(self, features, mask, lengths):
        """
        Args:
            features: (B, T, D)
            mask: (B, T)
            lengths: (B,)
        Returns:
            List of dictionaries containing 'cls' and 'bnd' probabilities for each stage.
        """
        # ================= Stage 1 =================
        x = self.embedding(features)

        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        out_packed, _ = self.lstm(packed)

        out, _ = pad_packed_sequence(
            out_packed, batch_first=True, total_length=features.size(1)
        )

        s1_cls_logits = self.s1_cls(out)
        s1_bnd_logits = self.s1_bnd(out)

        s1_cls_prob = F.softmax(s1_cls_logits, dim=2)
        s1_bnd_prob = torch.sigmoid(s1_bnd_logits)

        # Inter-Stage Masking
        mask_expanded = mask.unsqueeze(2)
        s1_cls_prob = s1_cls_prob * mask_expanded
        s1_bnd_prob = s1_bnd_prob * mask_expanded

        # ================= Stage 2 =================
        # Concatenate probs: (B, T, 22) -> Permute to (B, 22, T)
        s1_concat = torch.cat([s1_cls_prob, s1_bnd_prob], dim=2)
        s1_concat_t = s1_concat.permute(0, 2, 1)

        s2_out_t = self.stage2(s1_concat_t, mask)
        s2_out = s2_out_t.permute(0, 2, 1)  # (B, T, 22)

        s2_cls_logits = s2_out[:, :, :NUM_CLASSES]
        s2_bnd_logits = s2_out[:, :, NUM_CLASSES:]

        s2_cls_prob = F.softmax(s2_cls_logits, dim=2) * mask_expanded
        s2_bnd_prob = torch.sigmoid(s2_bnd_logits) * mask_expanded

        # ================= Stage 3 =================
        s2_concat = torch.cat([s2_cls_prob, s2_bnd_prob], dim=2)
        s2_concat_t = s2_concat.permute(0, 2, 1)

        s3_out_t = self.stage3(s2_concat_t, mask)
        s3_out = s3_out_t.permute(0, 2, 1)

        s3_cls_logits = s3_out[:, :, :NUM_CLASSES]
        s3_bnd_logits = s3_out[:, :, NUM_CLASSES:]

        s3_cls_prob = F.softmax(s3_cls_logits, dim=2) * mask_expanded
        s3_bnd_prob = torch.sigmoid(s3_bnd_logits) * mask_expanded

        return [
            {"cls": s1_cls_prob, "bnd": s1_bnd_prob},
            {"cls": s2_cls_prob, "bnd": s2_bnd_prob},
            {"cls": s3_cls_prob, "bnd": s3_bnd_prob},
        ]
