import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    NUM_CLASSES,
    HIDDEN_SIZE,
    NUM_LSTM_LAYERS,
    NUM_TCN_LAYERS,
    TCN_KERNEL_SIZE,
    DROPOUT,
    SELECTED_JOINTS,
    AUDIO_MFCC_N_MFCC,
)


class DualScaleGatedBlock(nn.Module):
    """
    Processes input through parallel Global (dilated) and Local (d=1) branches.
    Each branch uses a Gated Activation Unit (tanh * sigmoid).
    Outputs are concatenated and projected back to input width.
    """

    def __init__(self, in_channels, branch_channels, kernel_size, dilation, dropout):
        super(DualScaleGatedBlock, self).__init__()

        self.padding_global = (kernel_size - 1) * dilation // 2
        self.padding_local = (kernel_size - 1) // 2

        # Global Branch: Dilated Convolution
        # Output channels = 2 * branch_channels for Gated Activation split
        self.conv_global = nn.Conv1d(
            in_channels,
            branch_channels * 2,
            kernel_size,
            padding=self.padding_global,
            dilation=dilation,
        )

        # Local Branch: Standard Convolution (dilation=1)
        self.conv_local = nn.Conv1d(
            in_channels,
            branch_channels * 2,
            kernel_size,
            padding=self.padding_local,
            dilation=1,
        )

        # Fusion: Project concatenated branches back to in_channels
        # Input to fusion is branch_channels (global) + branch_channels (local)
        self.fusion = nn.Conv1d(branch_channels * 2, in_channels, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T)

        # Global Branch
        g = self.conv_global(x)
        g_tanh, g_sigmoid = torch.chunk(g, 2, dim=1)
        out_global = torch.tanh(g_tanh) * torch.sigmoid(g_sigmoid)

        # Local Branch
        l = self.conv_local(x)
        l_tanh, l_sigmoid = torch.chunk(l, 2, dim=1)
        out_local = torch.tanh(l_tanh) * torch.sigmoid(l_sigmoid)

        # Fusion
        fused = torch.cat([out_global, out_local], dim=1)
        out = self.fusion(fused)
        out = self.dropout(out)

        # Residual connection
        return x + out


class BiLSTMEncoder(nn.Module):
    """
    Stage 1: Multi-Task Recurrent Encoder.
    Processes raw features and outputs initial probabilities.
    """

    def __init__(self, input_dim, hidden_size, num_layers, num_classes, dropout):
        super(BiLSTMEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Project from bidirectional hidden size (2 * hidden) to output
        encoder_dim = hidden_size * 2

        self.cls_head = nn.Linear(encoder_dim, num_classes)
        self.bnd_head = nn.Linear(encoder_dim, 1)

    def forward(self, x):
        # x: (B, T, InputDim)

        # LSTM output: (B, T, 2*Hidden)
        features, _ = self.lstm(x)

        # Heads
        cls_logits = self.cls_head(features)  # (B, T, NumClasses)
        bnd_logits = self.bnd_head(features)  # (B, T, 1)

        # Probabilities
        cls_probs = F.softmax(cls_logits, dim=2)
        bnd_probs = torch.sigmoid(bnd_logits)

        return cls_probs, bnd_probs


class RefinementStage(nn.Module):
    """
    Stage 2 & 3: Dual-Scale Gated Refinement.
    Takes probabilities from previous stage, refines them using TCN stacks.
    """

    def __init__(
        self, input_dim, hidden_size, num_layers, kernel_size, num_classes, dropout
    ):
        super(RefinementStage, self).__init__()

        # Input projection
        self.input_proj = nn.Conv1d(input_dim, hidden_size, 1)

        # Stack of Dual-Scale Gated Blocks
        self.layers = nn.ModuleList()
        branch_channels = hidden_size // 2  # Split hidden size between two branches

        for i in range(num_layers):
            dilation = 2**i
            self.layers.append(
                DualScaleGatedBlock(
                    in_channels=hidden_size,
                    branch_channels=branch_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        # Output heads
        self.cls_head = nn.Conv1d(hidden_size, num_classes, 1)
        self.bnd_head = nn.Conv1d(hidden_size, 1, 1)

    def forward(self, x, mask):
        # x: (B, InputDim, T) - Concatenated probs from prev stage
        # mask: (B, 1, T)

        # Project input
        out = self.input_proj(x)

        # Apply blocks
        for layer in self.layers:
            out = layer(out)
            # Apply mask inside the stack to keep zeros zeroed?
            # Usually applied after the stage or before.
            # Prompt says "Inter-Stage Masking", implying between stages.
            # However, for TCNs, it's often good practice to mask padded areas to avoid artifact propagation.
            if mask is not None:
                out = out * mask

        # Heads
        cls_logits = self.cls_head(out)
        bnd_logits = self.bnd_head(out)

        # Probabilities
        # Transpose back to (B, T, C) for consistency with Stage 1 output format in dictionary
        cls_probs = F.softmax(cls_logits, dim=1).transpose(1, 2)
        bnd_probs = torch.sigmoid(bnd_logits).transpose(1, 2)

        return cls_probs, bnd_probs


class DSG_CRCN(nn.Module):
    """
    Dual-Scale Supervised Gated-Cascaded Recurrent-Convolutional Network.
    """

    def __init__(self):
        super(DSG_CRCN, self).__init__()

        # Calculate Input Dimension
        # 12 joints * 3 coords * 2 (pos + vel) + 13 MFCCs
        num_joints = len(SELECTED_JOINTS)
        input_dim = (num_joints * 3 * 2) + AUDIO_MFCC_N_MFCC

        # Stage 1
        self.stage1 = BiLSTMEncoder(
            input_dim=input_dim,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LSTM_LAYERS,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT,
        )

        # Stage 2
        # Input to Stage 2 is [P_cls, P_bnd] from Stage 1
        stage2_input_dim = NUM_CLASSES + 1
        self.stage2 = RefinementStage(
            input_dim=stage2_input_dim,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_TCN_LAYERS,
            kernel_size=TCN_KERNEL_SIZE,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT,
        )

        # Stage 3
        # Input to Stage 3 is [P_cls, P_bnd] from Stage 2
        self.stage3 = RefinementStage(
            input_dim=stage2_input_dim,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_TCN_LAYERS,
            kernel_size=TCN_KERNEL_SIZE,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT,
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, InputDim)
            mask: (B, T) Boolean mask where True indicates valid data.
        """
        outputs = {}

        # --- Stage 1 ---
        s1_cls, s1_bnd = self.stage1(x)
        outputs["stage1_cls"] = s1_cls
        outputs["stage1_bnd"] = s1_bnd

        # Prepare mask for Conv1d (B, 1, T)
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).float()  # (B, 1, T)
        else:
            mask_expanded = torch.ones(x.shape[0], 1, x.shape[1], device=x.device)

        # Inter-Stage Masking 1
        # Concatenate outputs: (B, T, C) -> (B, C, T)
        s1_combined = torch.cat([s1_cls, s1_bnd], dim=2).transpose(1, 2)
        s1_masked = s1_combined * mask_expanded

        # --- Stage 2 ---
        s2_cls, s2_bnd = self.stage2(s1_masked, mask_expanded)
        outputs["stage2_cls"] = s2_cls
        outputs["stage2_bnd"] = s2_bnd

        # Inter-Stage Masking 2
        s2_combined = torch.cat([s2_cls, s2_bnd], dim=2).transpose(1, 2)
        s2_masked = s2_combined * mask_expanded

        # --- Stage 3 ---
        s3_cls, s3_bnd = self.stage3(s2_masked, mask_expanded)
        outputs["stage3_cls"] = s3_cls
        outputs["stage3_bnd"] = s3_bnd

        # Final output for inference is usually Stage 3
        outputs["final_cls"] = s3_cls
        outputs["final_bnd"] = s3_bnd

        return outputs
