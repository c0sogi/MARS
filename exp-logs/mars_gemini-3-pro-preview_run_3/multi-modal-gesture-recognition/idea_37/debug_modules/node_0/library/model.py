import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class BiGRUEncoder(nn.Module):
    """
    Stage 1: Regularized High-Capacity Encoder.

    Processes frame-wise features using a Bi-Directional GRU to capture
    short-term temporal dependencies and kinematic context.
    """

    def __init__(self):
        super(BiGRUEncoder, self).__init__()

        self.input_dim = Config.TOTAL_INPUT_DIM
        self.hidden_dim = Config.HIDDEN_DIM  # 128 per direction
        self.num_classes = Config.NUM_CLASSES
        self.dropout_p = Config.DROPOUT_ENCODER

        # 2-layer Bi-GRU
        # Total hidden size output will be hidden_dim * 2 (256)
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_p,
        )

        # Projection to class probabilities
        # We apply dropout before the final linear layer as well
        self.dropout = nn.Dropout(self.dropout_p)
        self.classifier = nn.Linear(self.hidden_dim * 2, self.num_classes)

    def forward(self, x):
        # x shape: (Batch, Time, Features)

        # GRU Output: (Batch, Time, Hidden * 2)
        out, _ = self.gru(x)

        out = self.dropout(out)
        logits = self.classifier(out)

        return logits


class DilatedResidualLayer(nn.Module):
    """
    Single layer of the Monotonic TCN.
    Uses Gated Activation (Tanh * Sigmoid) and Residual Connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_p):
        super(DilatedResidualLayer, self).__init__()

        # Padding for non-causal convolution (centered)
        # padding = (kernel_size - 1) * dilation / 2
        self.padding = (kernel_size - 1) * dilation // 2

        # Conv1d outputting 2 * out_channels for Gated Activation
        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels * 2,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

        self.dropout = nn.Dropout(dropout_p)

        # 1x1 Conv for residual alignment if needed, or just processing
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x):
        # x shape: (Batch, Channels, Time)

        out = self.conv_dilated(x)

        # Split for Gated Activation
        # out shape: (Batch, 2*Channels, Time)
        out_tanh, out_sigmoid = torch.chunk(out, 2, dim=1)

        # Gated Activation
        out = torch.tanh(out_tanh) * torch.sigmoid(out_sigmoid)

        out = self.dropout(out)

        # 1x1 projection
        out = self.conv_1x1(out)

        # Residual connection
        return x + out


class MonotonicTCNBlock(nn.Module):
    """
    Stage 2 & 3: Monotonic Non-Causal Refinement.

    Refines class probabilities using a stack of dilated convolutions
    with a monotonically increasing dilation schedule.
    """

    def __init__(self):
        super(MonotonicTCNBlock, self).__init__()

        self.num_classes = Config.NUM_CLASSES
        # Internal hidden dimension for TCN (standard practice is 64 for MS-TCN)
        self.tcn_hidden = 64
        self.dropout_p = Config.DROPOUT_TCN
        self.kernel_size = Config.TCN_KERNEL_SIZE
        self.dilations = Config.TCN_DILATIONS  # [1, 2, 4, 8, 16]

        # Input projection: Classes -> Hidden
        self.conv_in = nn.Conv1d(self.num_classes, self.tcn_hidden, 1)

        # Stack of dilated layers
        self.layers = nn.ModuleList()
        for dilation in self.dilations:
            self.layers.append(
                DilatedResidualLayer(
                    self.tcn_hidden,
                    self.tcn_hidden,
                    self.kernel_size,
                    dilation,
                    self.dropout_p,
                )
            )

        # Output projection: Hidden -> Classes
        self.conv_out = nn.Conv1d(self.tcn_hidden, self.num_classes, 1)

    def forward(self, x):
        # x shape: (Batch, Time, Classes) -> Transpose for Conv1d
        x = x.transpose(1, 2)  # (Batch, Classes, Time)

        out = self.conv_in(x)

        for layer in self.layers:
            out = layer(out)

        out = self.conv_out(out)

        # Transpose back: (Batch, Time, Classes)
        out = out.transpose(1, 2)

        return out


class RHCKN(nn.Module):
    """
    Regularized High-Capacity Kinematic Network (RHC-KN).

    A Three-Stage Cascaded Network:
    1. Bi-GRU Encoder (Raw Features -> P1)
    2. TCN Refinement 1 (P1 -> P2)
    3. TCN Refinement 2 (P2 -> P3)
    """

    def __init__(self):
        super(RHCKN, self).__init__()

        # Stage 1
        self.stage1_encoder = BiGRUEncoder()

        # Stage 2 (Independent weights)
        self.stage2_refinement = MonotonicTCNBlock()

        # Stage 3 (Independent weights)
        self.stage3_refinement = MonotonicTCNBlock()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Time, Features).

        Returns:
            dict: Dictionary containing log-probabilities for each stage.
                  {'stage1': ..., 'stage2': ..., 'stage3': ...}
                  Outputs are LogSoftmax'd for numerical stability with NLLLoss/KLDiv.
        """

        # --- Stage 1 ---
        # Logits: (Batch, Time, Classes)
        s1_logits = self.stage1_encoder(x)
        s1_probs = F.softmax(s1_logits, dim=2)  # Probabilities for next stage

        # --- Stage 2 ---
        # Input: Strictly probabilities from Stage 1
        s2_logits = self.stage2_refinement(s1_probs)
        s2_probs = F.softmax(s2_logits, dim=2)

        # --- Stage 3 ---
        # Input: Strictly probabilities from Stage 2
        s3_logits = self.stage3_refinement(s2_probs)

        # Return log_softmax for all stages for loss calculation
        return {
            "stage1": F.log_softmax(s1_logits, dim=2),
            "stage2": F.log_softmax(s2_logits, dim=2),
            "stage3": F.log_softmax(s3_logits, dim=2),
        }
