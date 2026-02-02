import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class GatedActivationUnit(nn.Module):
    """
    Gated Activation Unit for Temporal Convolutional Networks.
    Implements the WaveNet-style gating mechanism:
    Z = tanh(W_f * X) * sigmoid(W_g * X)

    This allows the network to dynamically regulate information flow,
    opening gates during transitions (high boundary prob) and closing them
    for smoothing during stable gestures.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(GatedActivationUnit, self).__init__()

        # Calculate padding for 'same' convolution with dilation
        # (kernel_size - 1) * dilation // 2 assumes odd kernel size
        padding = (kernel_size - 1) * dilation // 2

        # Filter convolution (Tanh path)
        self.conv_filter = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )

        # Gate convolution (Sigmoid path)
        self.conv_gate = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )

        # 1x1 Projection for residual integration
        self.conv_1x1 = nn.Conv1d(out_channels, in_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Channels, Time)

        # Filter: Tanh activation
        f = torch.tanh(self.conv_filter(x))

        # Gate: Sigmoid activation
        g = torch.sigmoid(self.conv_gate(x))

        # Element-wise multiplication (Gating)
        out = f * g

        # Projection and Dropout
        out = self.conv_1x1(out)
        out = self.dropout(out)

        # Residual Connection
        return x + out


class BiLSTMEncoder(nn.Module):
    """
    Stage 1: Multi-Task Recurrent Encoder.

    Backbone: Bi-Directional LSTM (2 layers).
    Heads:
        1. Class Probabilities (Softmax)
        2. Boundary Probability (Sigmoid)
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(BiLSTMEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True
        )

        # BiLSTM outputs 2 * hidden_dim features
        lstm_out_dim = hidden_dim * 2

        # Prediction Heads
        self.cls_head = nn.Linear(lstm_out_dim, num_classes)
        self.bnd_head = nn.Linear(lstm_out_dim, 1)

    def forward(self, x):
        # x shape: (Batch, Time, InputDim)

        # Pass through LSTM
        feat, _ = self.lstm(x)

        # Class Head
        cls_logits = self.cls_head(feat)
        cls_probs = F.softmax(cls_logits, dim=2)  # (Batch, Time, NumClasses)

        # Boundary Head
        bnd_logits = self.bnd_head(feat)
        bnd_probs = torch.sigmoid(bnd_logits)  # (Batch, Time, 1)

        return cls_probs, bnd_probs


class GatedRefinementStage(nn.Module):
    """
    Stage 2 & 3: Supervised Gated Refinement (Gated MS-TCN).

    Takes probability distributions from the previous stage and refines them
    using a stack of Dilated Gated Activation Units.
    """

    def __init__(
        self, input_dim, hidden_dim, num_layers, num_classes, kernel_size, dropout
    ):
        super(GatedRefinementStage, self).__init__()

        # Initial projection to hidden dimension
        self.conv_1x1_in = nn.Conv1d(input_dim, hidden_dim, 1)

        # Stack of Gated Units with increasing dilation
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2**i
            self.layers.append(
                GatedActivationUnit(
                    hidden_dim, hidden_dim, kernel_size, dilation, dropout
                )
            )

        # Output Heads (1x1 Convs)
        self.cls_head = nn.Conv1d(hidden_dim, num_classes, 1)
        self.bnd_head = nn.Conv1d(hidden_dim, 1, 1)

    def forward(self, x):
        # x shape: (Batch, Time, InputDim) -> Transpose to (Batch, InputDim, Time) for Conv1d
        x = x.transpose(1, 2)

        out = self.conv_1x1_in(x)

        for layer in self.layers:
            out = layer(out)

        # Heads
        cls_logits = self.cls_head(out)  # (Batch, NumClasses, Time)
        bnd_logits = self.bnd_head(out)  # (Batch, 1, Time)

        # Transpose back to (Batch, Time, Features)
        cls_logits = cls_logits.transpose(1, 2)
        bnd_logits = bnd_logits.transpose(1, 2)

        cls_probs = F.softmax(cls_logits, dim=2)
        bnd_probs = torch.sigmoid(bnd_logits)

        return cls_probs, bnd_probs


class SG_CRCN(nn.Module):
    """
    Supervised Gated-Cascaded Recurrent-Convolutional Network (SG-CRCN).

    Structure:
    1. Stage 1: BiLSTM Encoder -> Coarse Predictions
    2. Masking
    3. Stage 2: Gated Refinement -> Refined Predictions
    4. Masking
    5. Stage 3: Gated Refinement -> Final Predictions
    """

    def __init__(self):
        super(SG_CRCN, self).__init__()

        self.input_dim = Config.INPUT_DIM
        self.num_classes = Config.NUM_CLASSES
        self.hidden_dim = Config.HIDDEN_DIM

        # --- Stage 1: BiLSTM ---
        self.stage1 = BiLSTMEncoder(
            self.input_dim, self.hidden_dim, Config.LSTM_LAYERS, self.num_classes
        )

        # Intermediate dimension is the concatenation of Class Probs + Boundary Prob
        self.inter_dim = self.num_classes + 1

        # --- Stage 2: Gated Refinement ---
        self.stage2 = GatedRefinementStage(
            self.inter_dim,
            self.hidden_dim,
            Config.NUM_TCN_LAYERS,
            self.num_classes,
            Config.KERNEL_SIZE,
            Config.DROPOUT,
        )

        # --- Stage 3: Gated Refinement ---
        self.stage3 = GatedRefinementStage(
            self.inter_dim,
            self.hidden_dim,
            Config.NUM_TCN_LAYERS,
            self.num_classes,
            Config.KERNEL_SIZE,
            Config.DROPOUT,
        )

    def forward(self, x, mask):
        """
        Args:
            x: Input features (Batch, Time, InputDim)
            mask: Sequence mask (Batch, Time)

        Returns:
            dict: Outputs from all stages for Deep Supervision.
        """
        # Expand mask for element-wise multiplication: (Batch, Time, 1)
        mask_expanded = mask.unsqueeze(2)

        # ---------------- Stage 1 ----------------
        s1_cls, s1_bnd = self.stage1(x)

        # Apply Masking (Inter-Stage Noise Suppression)
        s1_cls = s1_cls * mask_expanded
        s1_bnd = s1_bnd * mask_expanded

        # Concatenate for next stage
        s1_out = torch.cat([s1_cls, s1_bnd], dim=2)

        # ---------------- Stage 2 ----------------
        s2_cls, s2_bnd = self.stage2(s1_out)

        # Apply Masking
        s2_cls = s2_cls * mask_expanded
        s2_bnd = s2_bnd * mask_expanded

        # Concatenate for next stage
        s2_out = torch.cat([s2_cls, s2_bnd], dim=2)

        # ---------------- Stage 3 ----------------
        s3_cls, s3_bnd = self.stage3(s2_out)

        # Apply Masking
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
