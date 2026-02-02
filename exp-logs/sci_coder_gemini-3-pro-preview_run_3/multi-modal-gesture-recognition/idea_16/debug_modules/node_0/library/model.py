import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class BiGRUEncoder(nn.Module):
    """
    Stage 1: Kinematic Sequence Encoder.
    Processes fused features using a Bi-directional GRU to generate initial class logits.
    """

    def __init__(self):
        super(BiGRUEncoder, self).__init__()
        self.gru = nn.GRU(
            input_size=Config.INPUT_DIM,
            hidden_size=Config.GRU_HIDDEN_DIM,
            num_layers=Config.GRU_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
        )
        # Map 2 * hidden_dim -> num_classes
        self.fc = nn.Linear(Config.GRU_HIDDEN_DIM * 2, Config.NUM_CLASSES)

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        self.gru.flatten_parameters()
        out, _ = self.gru(x)
        logits = self.fc(out)
        return logits


class GatedDilatedLayer(nn.Module):
    """
    A single layer of the TCN Refinement module.
    Uses Gated Activation Units (WaveNet style) and Residual Connections.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(GatedDilatedLayer, self).__init__()

        # Padding for 'same' output length with dilation
        # For kernel_size=3, padding = dilation
        self.padding = dilation

        # Dilated Convolution: Maps input to 2 * out_channels (for Filter + Gate)
        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels * 2,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

        # 1x1 Convolution for output projection
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Channels, Time)

        # 1. Dilated Conv
        out = self.conv_dilated(x)

        # 2. Gated Activation
        # Split into Filter and Gate
        filter_conv, gate_conv = out.chunk(2, dim=1)

        # Tanh * Sigmoid
        activation = torch.tanh(filter_conv) * torch.sigmoid(gate_conv)

        # 3. Dropout
        activation = self.dropout(activation)

        # 4. 1x1 Projection
        out = self.conv_1x1(activation)

        # 5. Residual Connection
        return x + out


class RefinementStage(nn.Module):
    """
    Stage 2 & 3: Deep-Field Gated Refinement Module.
    Takes class probabilities as input and refines them using a stack of dilated layers.
    """

    def __init__(self, num_classes, hidden_dim, num_layers, kernel_size, dropout):
        super(RefinementStage, self).__init__()

        # Input projection: Probabilities (num_classes) -> Hidden Dim
        self.conv_in = nn.Conv1d(num_classes, hidden_dim, 1)

        # Stack of Gated Dilated Layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Dilation schedule: 1, 2, 4, 8, 16, 32...
            dilation = 2**i
            self.layers.append(
                GatedDilatedLayer(
                    hidden_dim, hidden_dim, kernel_size, dilation, dropout
                )
            )

        # Output projection: Hidden Dim -> Logits (num_classes)
        self.conv_out = nn.Conv1d(hidden_dim, num_classes, 1)

    def forward(self, probs):
        # probs: (Batch, Time, NumClasses)

        # Transpose for Conv1d: (Batch, NumClasses, Time)
        x = probs.transpose(1, 2)

        # Project to hidden dimension
        x = self.conv_in(x)

        # Apply dilated layers
        for layer in self.layers:
            x = layer(x)

        # Project back to class space
        out = self.conv_out(x)

        # Transpose back: (Batch, Time, NumClasses)
        logits = out.transpose(1, 2)

        return logits


class RDKRN(nn.Module):
    """
    Robust Deep-Field Kinematic Refinement Network.
    Three-Stage Cascaded Network:
    1. Bi-GRU Encoder
    2. Refinement Stage 1 (TCN)
    3. Refinement Stage 2 (TCN)
    """

    def __init__(self):
        super(RDKRN, self).__init__()

        # Stage 1
        self.stage1 = BiGRUEncoder()

        # Stage 2
        self.stage2 = RefinementStage(
            num_classes=Config.NUM_CLASSES,
            hidden_dim=Config.TCN_FEATURE_DIM,
            num_layers=Config.TCN_NUM_LAYERS,  # 6 layers -> max dilation 32
            kernel_size=Config.TCN_KERNEL_SIZE,
            dropout=Config.TCN_DROPOUT,
        )

        # Stage 3
        self.stage3 = RefinementStage(
            num_classes=Config.NUM_CLASSES,
            hidden_dim=Config.TCN_FEATURE_DIM,
            num_layers=Config.TCN_NUM_LAYERS,
            kernel_size=Config.TCN_KERNEL_SIZE,
            dropout=Config.TCN_DROPOUT,
        )

    def forward(self, x):
        # x: (Batch, Time, InputDim)

        # --- Stage 1 ---
        # Output: Logits
        l1 = self.stage1(x)

        # Bottleneck: Convert to Probabilities for next stage
        # Detach? No, we want gradients to flow back through probabilities if needed,
        # but standard MS-TCN trains end-to-end.
        p1 = F.softmax(l1, dim=2)

        # --- Stage 2 ---
        # Input: P1, Output: Logits
        l2 = self.stage2(p1)
        p2 = F.softmax(l2, dim=2)

        # --- Stage 3 ---
        # Input: P2, Output: Logits
        l3 = self.stage3(p2)

        # Return all logits for Deep Supervision Loss
        return [l1, l2, l3]
