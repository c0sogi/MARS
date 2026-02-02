import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedResidualLayer(nn.Module):
    """
    A single dilated residual block for the TCN.
    Uses centered padding to maintain temporal resolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(DilatedResidualLayer, self).__init__()

        # Calculate padding to maintain sequence length (centered/acausal)
        # padding = (kernel_size - 1) * dilation // 2
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=padding
        )
        # Removed InstanceNorm (Cite Lesson 00066)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        # Removed InstanceNorm (Cite Lesson 00066)

        # Downsample if channel dimensions change
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.relu(out + residual)


class SingleStageTCN(nn.Module):
    """
    Refinement network using a stack of dilated residual layers.
    Outputs refined Class logits.
    """

    def __init__(self, input_dim, num_classes):
        super(SingleStageTCN, self).__init__()

        layers = []
        num_channels = Config.TCN_NUM_CHANNELS
        kernel_size = Config.TCN_KERNEL_SIZE
        dropout = Config.TCN_DROPOUT

        in_c = input_dim
        for i, out_c in enumerate(num_channels):
            dilation = 2**i
            layers.append(
                DilatedResidualLayer(in_c, out_c, kernel_size, dilation, dropout)
            )
            in_c = out_c

        self.network = nn.Sequential(*layers)

        # Heads
        self.cls_head = nn.Conv1d(in_c, num_classes, 1)
        self.bnd_head = nn.Conv1d(in_c, 1, 1)

    def forward(self, x):
        # x: (B, C, T)
        feat = self.network(x)
        cls_logits = self.cls_head(feat)
        bnd_logits = self.bnd_head(feat)
        return cls_logits, bnd_logits


class BiLSTMEncoder(nn.Module):
    """
    Stage 1: Recurrent Encoder for initial prediction.
    """

    def __init__(self, input_dim, num_classes):
        super(BiLSTMEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_dim,
            Config.LSTM_HIDDEN_DIM,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
            dropout=Config.LSTM_DROPOUT if Config.LSTM_LAYERS > 1 else 0,
        )

        hidden_dim = (
            Config.LSTM_HIDDEN_DIM * 2
            if Config.BIDIRECTIONAL
            else Config.LSTM_HIDDEN_DIM
        )

        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.bnd_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, InputDim)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)

        cls_logits = self.cls_head(out)  # (B, T, C)
        bnd_logits = self.bnd_head(out)  # (B, T, 1)
        return cls_logits, bnd_logits


class MultiStageModel(nn.Module):
    """
    Boundary-Aware Masked Dual-Stage Cascaded Recurrent-Convolutional Network (BA-MD-CRCN).
    Orchestrates Stage 1 (LSTM) and Stages 2 & 3 (TCN) with explicit masking.
    """

    def __init__(self):
        super(MultiStageModel, self).__init__()

        # Stage 1: Initial Prediction
        self.stage1 = BiLSTMEncoder(Config.INPUT_DIM, Config.NUM_CLASSES)

        # Stage 2 & 3: Refinement
        # Input to TCN is concatenated probabilities: [P_cls (21) + P_bnd (1)]
        tcn_input_dim = Config.NUM_CLASSES + 1

        self.stage2 = SingleStageTCN(tcn_input_dim, Config.NUM_CLASSES)
        self.stage3 = SingleStageTCN(tcn_input_dim, Config.NUM_CLASSES)

    def forward(self, x, mask):
        """
        Args:
            x: (B, T, InputDim) - Input features
            mask: (B, T) - Sequence mask (1 for valid, 0 for padding)
        Returns:
            dict: Outputs from all stages for Deep Supervision.
        """
        # Expand mask for element-wise multiplication: (B, T, 1)
        mask_expanded = mask.unsqueeze(2)

        # ================= Stage 1 (LSTM) =================
        s1_cls_logits, s1_bnd_logits = self.stage1(x)

        # Compute probabilities for next stage
        s1_cls_probs = F.softmax(s1_cls_logits, dim=2)
        s1_bnd_probs = torch.sigmoid(s1_bnd_logits)

        # Explicit Masking: Zero out padding noise
        s1_cls_probs = s1_cls_probs * mask_expanded
        s1_bnd_probs = s1_bnd_probs * mask_expanded

        # ================= Stage 2 (TCN) =================
        # Prepare Input: Concat [P_cls, P_bnd] -> (B, T, C+1)
        s2_input = torch.cat([s1_cls_probs, s1_bnd_probs], dim=2)
        # Permute for TCN (Conv1d expects B, C, T): (B, C+1, T)
        s2_input = s2_input.permute(0, 2, 1)

        s2_cls_logits, s2_bnd_logits = self.stage2(s2_input)

        # Permute back: (B, T, C)
        s2_cls_logits = s2_cls_logits.permute(0, 2, 1)
        s2_bnd_logits = s2_bnd_logits.permute(0, 2, 1)

        # Probabilities & Masking
        s2_cls_probs = F.softmax(s2_cls_logits, dim=2)
        s2_bnd_probs = torch.sigmoid(s2_bnd_logits)

        s2_cls_probs = s2_cls_probs * mask_expanded
        s2_bnd_probs = s2_bnd_probs * mask_expanded

        # ================= Stage 3 (TCN) =================
        # Prepare Input
        s3_input = torch.cat([s2_cls_probs, s2_bnd_probs], dim=2)
        s3_input = s3_input.permute(0, 2, 1)

        s3_cls_logits, s3_bnd_logits = self.stage3(s3_input)

        # Permute back
        s3_cls_logits = s3_cls_logits.permute(0, 2, 1)
        s3_bnd_logits = s3_bnd_logits.permute(0, 2, 1)

        s3_cls_probs = F.softmax(s3_cls_logits, dim=2)
        # Note: No masking strictly needed here for output, but good for consistency if used downstream

        return {
            "stage1": {
                "cls_logits": s1_cls_logits,
                "bnd_logits": s1_bnd_logits,
                "cls_probs": s1_cls_probs,
            },
            "stage2": {
                "cls_logits": s2_cls_logits,
                "bnd_logits": s2_bnd_logits,
                "cls_probs": s2_cls_probs,
            },
            "stage3": {
                "cls_logits": s3_cls_logits,
                "bnd_logits": s3_bnd_logits,
                "cls_probs": s3_cls_probs,
            },
        }
