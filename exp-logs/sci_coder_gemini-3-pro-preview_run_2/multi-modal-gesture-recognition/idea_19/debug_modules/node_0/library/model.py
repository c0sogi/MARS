import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class GatedBlock(nn.Module):
    """
    Gated Activation Block for TCN.
    Computes Z = tanh(W_f * X) * sigmoid(W_g * X).
    Includes a residual connection: Output = X + Z.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(GatedBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation // 2

        # Filter convolution (Tanh stream)
        self.filter_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

        # Gate convolution (Sigmoid stream)
        self.gate_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

        self.dropout = nn.Dropout(dropout)

        # Residual connection projection if dimensions change
        if in_channels != out_channels:
            self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.res_conv = None

    def forward(self, x):
        # x: (B, C, T)

        # Gated Activation Mechanism
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))

        z = filter_out * gate_out
        z = self.dropout(z)

        # Residual Connection
        res = x if self.res_conv is None else self.res_conv(x)

        return res + z


class BiLSTMEncoder(nn.Module):
    """
    Stage 1: Multi-Task Recurrent Encoder.
    Backbone: Bi-Directional LSTM.
    Outputs: Class Probabilities (Softmax) and Boundary Probability (Sigmoid).
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(BiLSTMEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Bidirectional output size is 2 * hidden_size
        lstm_out_size = hidden_size * 2
        self.dropout = nn.Dropout(dropout)

        # Prediction Heads
        self.cls_head = nn.Linear(lstm_out_size, num_classes)
        self.bnd_head = nn.Linear(lstm_out_size, 1)

    def forward(self, x, mask):
        # x: (B, T, F)
        # mask: (B, T) - Not strictly needed for LSTM computation but kept for API consistency

        lstm_out, _ = self.lstm(x)  # (B, T, 2*H)
        lstm_out = self.dropout(lstm_out)

        # Compute Logits
        cls_logits = self.cls_head(lstm_out)  # (B, T, C)
        bnd_logits = self.bnd_head(lstm_out)  # (B, T, 1)

        # Permute to (B, C, T) for compatibility with TCN stages
        cls_logits = cls_logits.permute(0, 2, 1)
        bnd_logits = bnd_logits.permute(0, 2, 1)

        # Apply Activations
        p_cls = F.softmax(cls_logits, dim=1)
        p_bnd = torch.sigmoid(bnd_logits)

        return p_cls, p_bnd


class GatedTCN(nn.Module):
    """
    Stage 2 & 3: Gated MS-TCN for Refinement.
    Uses GatedBlocks to refine probabilities.
    """

    def __init__(
        self,
        input_channels,
        num_layers,
        num_channels,
        kernel_size,
        dropout,
        num_classes,
    ):
        super(GatedTCN, self).__init__()

        layers = []
        # Input projection to hidden size
        self.input_proj = nn.Conv1d(input_channels, num_channels, 1)

        # Stack Gated Blocks with increasing dilation
        for i in range(num_layers):
            dilation = 2**i
            layers.append(
                GatedBlock(num_channels, num_channels, kernel_size, dilation, dropout)
            )
        self.network = nn.Sequential(*layers)

        # Prediction Heads
        self.cls_head = nn.Conv1d(num_channels, num_classes, 1)
        self.bnd_head = nn.Conv1d(num_channels, 1, 1)

    def forward(self, x, mask):
        # x: (B, C_in, T)
        # mask: (B, 1, T)

        # Project and Refine
        out = self.input_proj(x)
        out = self.network(out)

        # Compute Logits
        cls_logits = self.cls_head(out)
        bnd_logits = self.bnd_head(out)

        # Apply Activations
        p_cls = F.softmax(cls_logits, dim=1)
        p_bnd = torch.sigmoid(bnd_logits)

        return p_cls, p_bnd


class RSG_CRCN(nn.Module):
    """
    Robust Supervised Gated-Cascaded Recurrent-Convolutional Network.
    Stage 1: BiLSTM -> [P_cls, P_bnd]
    Stage 2: GatedTCN -> [P'_cls, P'_bnd] (Refinement)
    Stage 3: GatedTCN -> [P''_cls, P''_bnd] (Sharpening)
    """

    def __init__(self):
        super(RSG_CRCN, self).__init__()

        # --- Feature Dimension Calculation ---
        # Skeleton: 12 joints * 3 coords = 36
        # Velocity: 12 joints * 3 coords = 36
        # Audio: 13 MFCCs
        input_dim = (
            Config.NUM_JOINTS * Config.CHANNELS_PER_JOINT * 2
        ) + Config.AUDIO_MFCC_N_MFCC

        # --- Stage 1: Recurrent Encoder ---
        self.stage1 = BiLSTMEncoder(
            input_size=input_dim,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_LAYERS,
            num_classes=Config.NUM_CLASSES,
            dropout=Config.DROPOUT,
        )

        # --- Stage 2 & 3: Gated TCN ---
        # Input to next stages is the concatenation of Class Probs and Boundary Prob
        stage_input_dim = Config.NUM_CLASSES + 1

        self.stage2 = GatedTCN(
            input_channels=stage_input_dim,
            num_layers=Config.TCN_NUM_LAYERS,
            num_channels=Config.TCN_CHANNELS,
            kernel_size=Config.TCN_KERNEL_SIZE,
            dropout=Config.DROPOUT,
            num_classes=Config.NUM_CLASSES,
        )

        self.stage3 = GatedTCN(
            input_channels=stage_input_dim,
            num_layers=Config.TCN_NUM_LAYERS,
            num_channels=Config.TCN_CHANNELS,
            kernel_size=Config.TCN_KERNEL_SIZE,
            dropout=Config.DROPOUT,
            num_classes=Config.NUM_CLASSES,
        )

    def forward(self, x, mask):
        """
        Forward pass with Inter-Stage Masking.

        Args:
            x (torch.Tensor): Input features (B, T, F).
            mask (torch.Tensor): Sequence mask (B, T).

        Returns:
            dict: Outputs from all stages for Deep Supervision.
        """
        # Expand mask for broadcasting over channels: (B, 1, T)
        mask_expanded = mask.unsqueeze(1)

        # --- Stage 1 ---
        # s1_cls: (B, C, T), s1_bnd: (B, 1, T)
        s1_cls, s1_bnd = self.stage1(x, mask)

        # Concatenate and Explicitly Mask
        # This prevents noise from padding regions propagating into the TCN
        s1_out = torch.cat([s1_cls, s1_bnd], dim=1)
        s1_out = s1_out * mask_expanded

        # --- Stage 2 ---
        s2_cls, s2_bnd = self.stage2(s1_out, mask_expanded)

        # Concatenate and Explicitly Mask
        s2_out = torch.cat([s2_cls, s2_bnd], dim=1)
        s2_out = s2_out * mask_expanded

        # --- Stage 3 ---
        s3_cls, s3_bnd = self.stage3(s2_out, mask_expanded)

        # Final masking for safety
        s3_cls = s3_cls * mask_expanded
        s3_bnd = s3_bnd * mask_expanded

        return {
            "stage1": (s1_cls, s1_bnd),
            "stage2": (s2_cls, s2_bnd),
            "stage3": (s3_cls, s3_bnd),
        }
