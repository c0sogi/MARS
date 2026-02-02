import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class DilatedResidualLayer(nn.Module):
    """
    A single Gated Dilated Temporal Convolutional Layer with Residual Connection.
    Implements the operation: Output = Activation(Conv(Input)) * 1x1Conv + Input
    where Activation is Tanh(Filter) * Sigmoid(Gate).
    """

    def __init__(self, dilation, in_channels, out_channels, kernel_size, dropout):
        super(DilatedResidualLayer, self).__init__()

        # Padding to maintain temporal dimension: (kernel_size - 1) * dilation
        # We use 'same' padding logic via explicit padding calculation
        self.padding = (kernel_size - 1) * dilation

        # Dilated Convolution
        # Output channels * 2 because we split into Filter and Gate
        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels * 2,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

        # 1x1 Convolution for residual aggregation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Channels, Time)

        out = self.conv_dilated(x)

        # Causal padding adjustment if necessary, but for offline video
        # (like this task), we usually use centered padding.
        # PyTorch Conv1d with padding adds zeros to both sides.
        # However, standard TCN implementations often crop the end if padding is causal.
        # Here we assume centered padding (non-causal) is acceptable for full video recognition.
        # If the padding creates extra frames, we crop them to match input length.

        # With padding=dilation*(k-1) and stride=1, the output length might be larger
        # if not handled carefully.
        # Actually, for centered padding: padding = (k-1)*d / 2.
        # But dilation makes it tricky.
        # Standard MS-TCN approach: Remove extra padding from the end (Chomp)
        # or use correct padding.
        # Let's use the Chomp strategy to be safe if padding adds too much,
        # or rely on PyTorch's padding.
        # Given kernel=3, dilation=d, padding=d results in length L + 2d - 2d = L (centered).
        # Wait, L_out = L_in + 2*padding - dilation*(kernel-1) - 1 + 1
        # L_out = L_in + 2*d - d*(2) = L_in.
        # So padding=dilation is correct for kernel=3 to keep size same.

        # Split into Filter and Gate
        out_filter, out_gate = torch.chunk(out, 2, dim=1)

        # Gated Activation
        out = torch.tanh(out_filter) * torch.sigmoid(out_gate)

        # Spatial convolution (1x1)
        out = self.conv_1x1(out)

        out = self.dropout(out)

        # Residual connection
        return x + out


class SawtoothTCN(nn.Module):
    """
    Hierarchical Refinement Module using a stack of Dilated Residual Layers
    configured with a Sawtooth Dilation Schedule.
    """

    def __init__(self, num_classes, hidden_dim, kernel_size, dropout):
        super(SawtoothTCN, self).__init__()

        # Input projection: Probabilities (num_classes) -> Hidden Dim
        self.conv_in = nn.Conv1d(num_classes, hidden_dim, 1)

        # Stack of dilated layers
        self.layers = nn.ModuleList()
        dilations = config.SAWTOOTH_DILATIONS

        for d in dilations:
            self.layers.append(
                DilatedResidualLayer(
                    dilation=d,
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )

        # Output projection: Hidden Dim -> Logits (num_classes)
        self.conv_out = nn.Conv1d(hidden_dim, num_classes, 1)

    def forward(self, x):
        # x: (Batch, NumClasses, Time) - Input probabilities from previous stage

        out = self.conv_in(x)

        for layer in self.layers:
            out = layer(out)

        out = self.conv_out(out)

        return out


class BiGRUEncoder(nn.Module):
    """
    Stage 1: High-Capacity Kinematic Sequence Encoder.
    Uses a Bi-Directional GRU to extract temporal features from raw inputs.
    """

    def __init__(self, input_dim, hidden_size, num_layers, num_classes, dropout):
        super(BiGRUEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Projection to classes
        # Input to Linear is hidden_size * 2 (bidirectional)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (Batch, Time, InputDim)

        # RNN Processing
        # out: (Batch, Time, Hidden*2)
        self.gru.flatten_parameters()
        out, _ = self.gru(x)

        # Project to classes
        # out: (Batch, Time, NumClasses)
        out = self.fc(out)

        # Transpose for TCN/Loss compatibility
        # Output: (Batch, NumClasses, Time)
        out = out.permute(0, 2, 1)

        return out


class RHKRN(nn.Module):
    """
    Robust Hierarchical Kinematic Refinement Network.
    A Three-Stage Cascaded Network:
    1. Bi-GRU Encoder (Features -> Logits)
    2. Sawtooth TCN Refinement (Probs -> Logits)
    3. Sawtooth TCN Refinement (Probs -> Logits)
    """

    def __init__(self):
        super(RHKRN, self).__init__()

        # Stage 1
        self.stage1 = BiGRUEncoder(
            input_dim=config.INPUT_DIM,
            hidden_size=config.GRU_HIDDEN_SIZE,
            num_layers=config.GRU_LAYERS,
            num_classes=config.NUM_CLASSES,
            dropout=config.GRU_DROPOUT,
        )

        # Stage 2
        self.stage2 = SawtoothTCN(
            num_classes=config.NUM_CLASSES,
            hidden_dim=config.MSTCN_FEATURES,
            kernel_size=config.MSTCN_KERNEL_SIZE,
            dropout=config.MSTCN_DROPOUT,
        )

        # Stage 3
        self.stage3 = SawtoothTCN(
            num_classes=config.NUM_CLASSES,
            hidden_dim=config.MSTCN_FEATURES,
            kernel_size=config.MSTCN_KERNEL_SIZE,
            dropout=config.MSTCN_DROPOUT,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Time, InputDim)

        Returns:
            list: [Stage1_Logits, Stage2_Logits, Stage3_Logits]
                  Each tensor has shape (Batch, NumClasses, Time)
        """
        outputs = []

        # --- Stage 1 ---
        # Input: Features
        # Output: Logits (B, C, T)
        s1_logits = self.stage1(x)
        outputs.append(s1_logits)

        # --- Stage 2 ---
        # Input: Probabilities from Stage 1 (Softmax)
        # We detach? No, we want gradients to flow back to Stage 1.
        s1_probs = F.softmax(s1_logits, dim=1)
        s2_logits = self.stage2(s1_probs)
        outputs.append(s2_logits)

        # --- Stage 3 ---
        # Input: Probabilities from Stage 2
        s2_probs = F.softmax(s2_logits, dim=1)
        s3_logits = self.stage3(s2_probs)
        outputs.append(s3_logits)

        return outputs
