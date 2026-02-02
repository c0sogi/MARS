import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class GatedActivation(nn.Module):
    """
    Gated Activation Unit: tanh(A) * sigmoid(B)
    """

    def __init__(self):
        super(GatedActivation, self).__init__()

    def forward(self, x):
        # x shape: (B, 2*C, T)
        # Split along channel dimension
        chunk_size = x.shape[1] // 2
        a, b = torch.split(x, chunk_size, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)


class DualScaleGatedBlock(nn.Module):
    """
    Dual-Scale Gated Block processing input via Global (dilated) and Local (fixed) branches.
    """

    def __init__(self, in_channels, branch_channels, kernel_size, dilation, dropout):
        super(DualScaleGatedBlock, self).__init__()

        self.in_channels = in_channels
        self.branch_channels = branch_channels

        # Global Branch: Dilated Convolution
        # Output channels = 2 * branch_channels for Gated Activation
        padding_global = (kernel_size - 1) * dilation // 2
        self.global_conv = nn.Conv1d(
            in_channels,
            branch_channels * 2,
            kernel_size,
            padding=padding_global,
            dilation=dilation,
        )

        # Local Branch: Fixed Dilation = 1
        padding_local = (kernel_size - 1) // 2
        self.local_conv = nn.Conv1d(
            in_channels,
            branch_channels * 2,
            kernel_size,
            padding=padding_local,
            dilation=1,
        )

        self.gated_act = GatedActivation()
        self.dropout = nn.Dropout(dropout)

        # Fusion: Project concatenated branches back to in_channels
        # Branch outputs are size branch_channels each -> concat is 2*branch_channels
        self.fusion_conv = nn.Conv1d(branch_channels * 2, in_channels, 1)

    def forward(self, x):
        # x: (B, C, T)

        # Global Branch
        out_global = self.global_conv(x)
        out_global = self.gated_act(out_global)  # (B, branch_channels, T)

        # Local Branch
        out_local = self.local_conv(x)
        out_local = self.gated_act(out_local)  # (B, branch_channels, T)

        # Concatenate
        out_concat = torch.cat(
            [out_global, out_local], dim=1
        )  # (B, 2*branch_channels, T)

        # Fusion
        out_fused = self.fusion_conv(out_concat)  # (B, in_channels, T)
        out_fused = self.dropout(out_fused)

        # Residual Connection
        return x + out_fused


class Stage1_Encoder(nn.Module):
    def __init__(self):
        super(Stage1_Encoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=Config.INPUT_DIM,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
        )

        # Input to heads is 2 * hidden_size due to bidirectional
        feature_dim = Config.HIDDEN_SIZE * 2

        self.cls_head = nn.Linear(feature_dim, Config.NUM_CLASSES)
        self.bnd_head = nn.Linear(feature_dim, 1)

    def forward(self, x, mask):
        # x: (B, T, D)
        # mask: (B, T)

        # Pack sequence for LSTM
        lengths = mask.sum(dim=1).cpu().int()
        # Handle case where lengths might be 0
        lengths = torch.clamp(lengths, min=1)

        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        packed_out, _ = self.lstm(packed_x)

        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # out: (B, T, 2*Hidden)

        # Heads
        cls_logits = self.cls_head(out)  # (B, T, NumClasses)
        bnd_logits = self.bnd_head(out)  # (B, T, 1)

        # Apply Softmax/Sigmoid
        cls_probs = F.softmax(cls_logits, dim=2)
        bnd_probs = torch.sigmoid(bnd_logits)

        return cls_probs, bnd_probs


class RefinementStage(nn.Module):
    def __init__(self, in_channels):
        super(RefinementStage, self).__init__()

        # Project input probabilities to hidden dimension
        # Input is (B, C, T) where C = NumClasses + 1 (Boundary)
        self.input_proj = nn.Conv1d(in_channels, Config.HIDDEN_SIZE, 1)

        layers = []
        num_layers = Config.STAGE_LAYERS
        for i in range(num_layers):
            dilation = 2**i
            layers.append(
                DualScaleGatedBlock(
                    in_channels=Config.HIDDEN_SIZE,
                    branch_channels=Config.BRANCH_CHANNELS,
                    kernel_size=Config.KERNEL_SIZE,
                    dilation=dilation,
                    dropout=Config.DROPOUT,
                )
            )
        self.tcn = nn.Sequential(*layers)

        # Heads
        self.cls_head = nn.Conv1d(Config.HIDDEN_SIZE, Config.NUM_CLASSES, 1)
        self.bnd_head = nn.Conv1d(Config.HIDDEN_SIZE, 1, 1)

    def forward(self, x):
        # x: (B, C_in, T)

        feat = self.input_proj(x)
        feat = self.tcn(feat)

        cls_logits = self.cls_head(feat)
        bnd_logits = self.bnd_head(feat)

        # (B, NumClasses, T) -> (B, T, NumClasses) for consistency with Stage 1 output format
        cls_probs = F.softmax(cls_logits, dim=1).transpose(1, 2)
        bnd_probs = torch.sigmoid(bnd_logits).transpose(1, 2)

        return cls_probs, bnd_probs


class DSG_CRCN(nn.Module):
    def __init__(self):
        super(DSG_CRCN, self).__init__()

        self.stage1 = Stage1_Encoder()

        # Input to refinement stages: Class Probs + Boundary Prob
        refinement_input_dim = Config.NUM_CLASSES + 1

        self.stage2 = RefinementStage(refinement_input_dim)
        self.stage3 = RefinementStage(refinement_input_dim)

    def forward(self, x, mask):
        # x: (B, T, InputDim)
        # mask: (B, T)

        # --- Stage 1 ---
        s1_cls, s1_bnd = self.stage1(x, mask)
        # s1_cls: (B, T, 21), s1_bnd: (B, T, 1)

        # Inter-Stage Masking
        # Expand mask for multiplication
        mask_expanded = mask.unsqueeze(2)  # (B, T, 1)
        s1_cls = s1_cls * mask_expanded
        s1_bnd = s1_bnd * mask_expanded

        # Prepare input for Stage 2
        # Concatenate and transpose to (B, C, T) for Conv1d
        s2_in = torch.cat([s1_cls, s1_bnd], dim=2)  # (B, T, 22)
        s2_in = s2_in.transpose(1, 2)  # (B, 22, T)

        # --- Stage 2 ---
        s2_cls, s2_bnd = self.stage2(s2_in)
        # Outputs are (B, T, C)

        # Masking
        s2_cls = s2_cls * mask_expanded
        s2_bnd = s2_bnd * mask_expanded

        # Prepare input for Stage 3
        s3_in = torch.cat([s2_cls, s2_bnd], dim=2).transpose(1, 2)

        # --- Stage 3 ---
        s3_cls, s3_bnd = self.stage3(s3_in)

        # Final Masking
        s3_cls = s3_cls * mask_expanded
        s3_bnd = s3_bnd * mask_expanded

        return {
            "stage1_cls": s1_cls,
            "stage1_bnd": s1_bnd,
            "stage2_cls": s2_cls,
            "stage2_bnd": s2_bnd,
            "stage3_cls": s3_cls,
            "stage3_bnd": s3_bnd,
        }
