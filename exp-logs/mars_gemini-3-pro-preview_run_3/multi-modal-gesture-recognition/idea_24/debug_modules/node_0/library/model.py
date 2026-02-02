import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class InputGatingLayer(nn.Module):
    """
    Learnable Input Gating Layer.
    Applies a feature-wise sigmoid gate to the input: x = x * sigmoid(Wx + b).
    This allows the model to suppress noisy features (e.g., jittery joints)
    before they enter the recurrent backbone.
    """

    def __init__(self, input_dim):
        super(InputGatingLayer, self).__init__()
        self.gate_fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x shape: (Batch, Time, InputDim)
        gate = torch.sigmoid(self.gate_fc(x))
        return x * gate


class GatedDilatedConv1d(nn.Module):
    """
    Single Gated Dilated Convolutional Layer.
    Implements: Output = Conv1x1(Tanh(W_f * x) * Sigmoid(W_g * x)) + x
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(GatedDilatedConv1d, self).__init__()

        # Calculate padding to maintain temporal dimension (Same padding)
        # Assuming odd kernel size k=3. Padding = dilation * (k-1) / 2
        self.padding = dilation * (kernel_size - 1) // 2

        # Convolution producing 2 * out_channels (for filter and gate)
        self.conv = nn.Conv1d(
            in_channels,
            2 * out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

        self.dropout = nn.Dropout(dropout)
        self.conv1x1 = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        residual = x

        out = self.conv(x)

        # Split into filter (P) and gate (Q)
        # Chunk along channel dimension (dim 1)
        P, Q = torch.chunk(out, 2, dim=1)

        # Gated Activation
        out = torch.tanh(P) * torch.sigmoid(Q)

        # Projection and Dropout
        out = self.conv1x1(out)
        out = self.dropout(out)

        return out + residual


class SawtoothTCNStage(nn.Module):
    """
    Refinement Stage using Sawtooth TCN.
    Dilation Schedule: [1, 2, 4, 8, 1, 2, 4, 8]
    """

    def __init__(self, num_classes, hidden_dim, kernel_size, dilations, dropout):
        super(SawtoothTCNStage, self).__init__()

        self.layers = nn.ModuleList()

        # Input Projection: Classes -> Hidden
        self.in_conv = nn.Conv1d(num_classes, hidden_dim, 1)

        # Stacked Dilated Layers
        for dilation in dilations:
            self.layers.append(
                GatedDilatedConv1d(
                    hidden_dim, hidden_dim, kernel_size, dilation, dropout
                )
            )

        # Output Projection: Hidden -> Classes
        self.out_conv = nn.Conv1d(hidden_dim, num_classes, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Time, NumClasses) - Probabilities from previous stage

        # Transpose to (Batch, NumClasses, Time) for Conv1d
        x = x.transpose(1, 2)

        out = self.in_conv(x)
        out = self.dropout(out)

        for layer in self.layers:
            out = layer(out)

        out = self.out_conv(out)

        # Transpose back to (Batch, Time, NumClasses)
        out = out.transpose(1, 2)

        return out


class GI_HCSN(nn.Module):
    """
    Gated-Input High-Capacity Sawtooth Network.
    Stage 1: Gated Bi-GRU
    Stage 2: Sawtooth TCN Refinement
    Stage 3: Sawtooth TCN Refinement
    """

    def __init__(self):
        super(GI_HCSN, self).__init__()

        # Hyperparameters
        input_dim = config.INPUT_DIM
        gru_hidden = config.HIDDEN_DIM  # 128
        num_classes = config.NUM_CLASSES
        tcn_channels = config.TCN_CHANNELS
        kernel_size = config.KERNEL_SIZE
        dilations = config.DILATIONS
        dropout = config.DROPOUT

        # --- Stage 1: Gated Encoder ---
        self.input_gate = InputGatingLayer(input_dim)

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden,
            num_layers=config.GRU_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if config.GRU_LAYERS > 1 else 0,
        )

        # Projection from GRU (2*hidden) to Classes
        self.stage1_fc = nn.Linear(gru_hidden * 2, num_classes)

        # --- Stage 2: Refinement ---
        self.stage2 = SawtoothTCNStage(
            num_classes, tcn_channels, kernel_size, dilations, dropout
        )

        # --- Stage 3: Refinement ---
        self.stage3 = SawtoothTCNStage(
            num_classes, tcn_channels, kernel_size, dilations, dropout
        )

    def forward(self, x):
        # x: (Batch, Time, InputDim)

        # --- Stage 1 ---
        # 1. Gating
        x_gated = self.input_gate(x)

        # 2. Bi-GRU
        # self.gru returns (output, h_n). output is (Batch, Time, 2*Hidden)
        gru_out, _ = self.gru(x_gated)

        # 3. Classification
        logits_1 = self.stage1_fc(gru_out)
        probs_1 = torch.softmax(logits_1, dim=2)

        # --- Stage 2 ---
        # Input is probabilities from Stage 1
        logits_2 = self.stage2(probs_1)
        probs_2 = torch.softmax(logits_2, dim=2)

        # --- Stage 3 ---
        # Input is probabilities from Stage 2
        logits_3 = self.stage3(probs_2)
        probs_3 = torch.softmax(logits_3, dim=2)

        # Return logits for loss calculation (CrossEntropyLoss expects logits)
        # Or probabilities if using custom loss.
        # The prompt implies we predict labels, but usually training uses logits.
        # However, the refinement stages take probabilities as input.
        # We return probabilities for consistency with the description "Output: P1, P2, P3".
        # But for numerical stability in loss, it's often better to return Logits.
        # Given the "Metric" section implies we need predictions, returning Logits allows
        # flexibility (argmax on logits == argmax on probs).
        # We will return Logits to be safe for CrossEntropy, but note inputs to stages were Probs.

        return logits_1, logits_2, logits_3
