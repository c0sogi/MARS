import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedResidualLayer(nn.Module):
    """
    A single layer of the MSTCN, implementing Gated Dilated Temporal Convolution.
    Structure: Dilated Conv -> Split (Filter/Gate) -> Activation -> 1x1 Conv -> Residual.
    """

    def __init__(self, dilation, in_channels, out_channels, dropout=Config.DROPOUT_TCN):
        super(DilatedResidualLayer, self).__init__()

        # Kernel size is fixed at 3 based on description/standard MSTCN
        kernel_size = Config.TCN_KERNEL_SIZE

        # Centered padding: (k-1) * d / 2
        # For k=3, padding = d
        padding = dilation

        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels * 2,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.conv_1x1 = nn.Conv1d(out_channels, in_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv_dilated(x)

        # Split into filter and gate
        filter_out, gate_out = torch.chunk(out, 2, dim=1)

        # Gated activation: tanh * sigmoid
        out = torch.tanh(filter_out) * torch.sigmoid(gate_out)

        out = self.dropout(out)
        out = self.conv_1x1(out)

        # Residual connection
        return x + out


class MSTCNStage(nn.Module):
    """
    A single stage of the refinement network.
    Stack of DilatedResidualLayers followed by a projection to class logits.
    """

    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(MSTCNStage, self).__init__()

        self.layers = nn.ModuleList(
            [
                DilatedResidualLayer(
                    dilation=2**i, in_channels=num_f_maps, out_channels=num_f_maps
                )
                for i in range(num_layers)
            ]
        )

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class KinematicBiGRU(nn.Module):
    """
    Stage 1: Physically-Aligned Kinematic Encoder.
    Bi-Directional GRU processing Early Fusion features.
    """

    def __init__(
        self, input_dim, hidden_dim, num_classes, dropout=Config.DROPOUT_ENCODER
    ):
        super(KinematicBiGRU, self).__init__()

        # Hidden dim is total, so per direction is hidden_dim // 2
        # Config.HIDDEN_DIM is 192, so 96 per direction.
        self.gru = nn.GRU(
            input_dim,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (Batch, Time, InputDim)

        # GRU Output: (Batch, Time, HiddenDim)
        out, _ = self.gru(x)

        out = self.dropout(out)

        # Project to classes -> Logits
        logits = self.fc(out)

        return logits


class PAM_CN(nn.Module):
    """
    Physically-Aligned Moderate-Capacity Network.
    Three-Stage Cascaded Network:
    1. KinematicBiGRU (Encoder)
    2. MSTCNStage (Refinement 1)
    3. MSTCNStage (Refinement 2)
    """

    def __init__(self):
        super(PAM_CN, self).__init__()

        # --- Stage 1: Encoder ---
        self.stage1 = KinematicBiGRU(
            input_dim=Config.INPUT_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            num_classes=Config.NUM_CLASSES,
        )

        # --- Stage 2: Refinement ---
        # Input to Stage 2 is class probabilities (NUM_CLASSES)
        # We project probabilities to TCN_HIDDEN_DIM first?
        # Or does the TCN operate on TCN_HIDDEN_DIM?
        # Usually MSTCN maps input_dim -> hidden_dim -> layers -> output_dim.
        # But here, we define the stage to take 'num_f_maps' channels.
        # We need an adapter 1x1 conv to map from classes to hidden dim.

        self.stage2_input_conv = nn.Conv1d(Config.NUM_CLASSES, Config.TCN_HIDDEN_DIM, 1)
        self.stage2 = MSTCNStage(
            num_layers=len(Config.DILATIONS),
            num_f_maps=Config.TCN_HIDDEN_DIM,
            dim=Config.TCN_HIDDEN_DIM,  # dim param not strictly used in loop logic above but good for ref
            num_classes=Config.NUM_CLASSES,
        )

        # --- Stage 3: Refinement ---
        # Independent weights, same structure
        self.stage3_input_conv = nn.Conv1d(Config.NUM_CLASSES, Config.TCN_HIDDEN_DIM, 1)
        self.stage3 = MSTCNStage(
            num_layers=len(Config.DILATIONS),
            num_f_maps=Config.TCN_HIDDEN_DIM,
            dim=Config.TCN_HIDDEN_DIM,
            num_classes=Config.NUM_CLASSES,
        )

    def forward(self, x):
        """
        Args:
            x: (Batch, Time, InputDim) - Early Fusion Features

        Returns:
            (p1, p2, p3): Tuple of logits for each stage.
                          Shape (Batch, Time, NumClasses)
        """
        # --- Stage 1 ---
        # Output: (Batch, Time, Classes)
        logits1 = self.stage1(x)

        # Prepare for Stage 2
        # Apply Softmax to get probabilities (Information Bottleneck)
        probs1 = F.softmax(logits1, dim=2)

        # Transpose for CNN: (Batch, Classes, Time)
        probs1_t = probs1.transpose(1, 2)

        # --- Stage 2 ---
        # Map to hidden dim
        stage2_in = self.stage2_input_conv(probs1_t)
        # Apply TCN
        logits2_t = self.stage2(stage2_in)
        # Transpose back: (Batch, Time, Classes)
        logits2 = logits2_t.transpose(1, 2)

        # Prepare for Stage 3
        probs2 = F.softmax(logits2, dim=2)
        probs2_t = probs2.transpose(1, 2)

        # --- Stage 3 ---
        stage3_in = self.stage3_input_conv(probs2_t)
        logits3_t = self.stage3(stage3_in)
        logits3 = logits3_t.transpose(1, 2)

        return logits1, logits2, logits3
