import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class DecoupledGatedInput(nn.Module):
    """
    Handles magnitude disparity between Skeleton (~1000) and Audio (~100) features.
    Uses LayerNorm to generate a gate, but applies it to the raw input to preserve
    physical scale hierarchies (Position >> Velocity >> Acceleration).
    """

    def __init__(self, input_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.gate_fc = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Batch, Time, InputDim)

        # Path A: Gate Generation (Normalized to prevent saturation)
        x_norm = self.norm(x)
        gate = self.sigmoid(self.gate_fc(x_norm))

        # Path B: Signal Retention (Raw) -> Fusion
        # We multiply raw input by the gate. This allows the model to suppress
        # noise (gate ~ 0) while keeping valid signal at its original physical scale.
        return x * gate


class DilatedResidualLayer(nn.Module):
    """
    Single layer for MSTCN with dilation and residual connection.
    Uses Gated Activation (tanh * sigmoid) for better temporal selection.
    Cite solution_lesson_node_00141.
    """

    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        # Calculate padding for 'same' convolution with dilation
        self.padding = (dilation * (kernel_size - 1)) // 2

        # Output channels * 2 for Gating (Filter + Gate)
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels * 2,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Channels, Time)
        out = self.conv(x)

        # Split into Filter and Gate
        filter, gate = torch.chunk(out, 2, dim=1)

        # Gated Activation
        out = torch.tanh(filter) * torch.sigmoid(gate)
        out = self.dropout(out)

        return x + out


class SingleStageTCN(nn.Module):
    """
    Refinement stage using a stack of dilated convolutions.
    Input: Class Probabilities (from previous stage).
    Output: Refined Logits.
    """

    def __init__(self, num_classes, hidden_channels, kernel_size, dilations, dropout):
        super().__init__()

        # Map probabilities to hidden channels
        self.conv_in = nn.Conv1d(num_classes, hidden_channels, 1)

        # Stack of dilated residual layers
        self.layers = nn.ModuleList()
        for d in dilations:
            self.layers.append(
                DilatedResidualLayer(hidden_channels, kernel_size, d, dropout)
            )

        # Map back to classes
        self.conv_out = nn.Conv1d(hidden_channels, num_classes, 1)

    def forward(self, x):
        # x: (Batch, Time, NumClasses)

        # Transpose for Conv1d: (Batch, NumClasses, Time)
        x = x.transpose(1, 2)

        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)

        # Transpose back: (Batch, Time, NumClasses)
        return out.transpose(1, 2)


class Stage1_BiGRU(nn.Module):
    """
    Initial sequence modeling using Bi-Directional GRU.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=config.GRU_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if config.GRU_NUM_LAYERS > 1 else 0,
        )
        # Bidirectional outputs concatenated -> hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        self.gru.flatten_parameters()
        out, _ = self.gru(x)
        logits = self.fc(out)
        return logits


class SKD_GN(nn.Module):
    """
    Structural-Kinematic Decoupled-Gated Network.
    Three-stage cascaded architecture:
    1. Gated Bi-GRU Encoder
    2. TCN Refinement 1
    3. TCN Refinement 2
    """

    def __init__(self):
        super().__init__()

        # --- Feature Dimensions ---
        skel_dim = config.get_skeleton_input_dim()
        audio_dim = config.AUDIO_N_MFCC
        total_input_dim = skel_dim + audio_dim

        # --- Stage 1: Encoder ---
        self.gated_input = DecoupledGatedInput(total_input_dim)
        self.stage1 = Stage1_BiGRU(
            total_input_dim,
            config.GRU_HIDDEN_SIZE,
            config.NUM_CLASSES,
            config.GRU_DROPOUT,
        )

        # --- Stage 2: Refinement ---
        self.stage2 = SingleStageTCN(
            config.NUM_CLASSES,
            config.TCN_CHANNELS,
            config.TCN_KERNEL_SIZE,
            config.TCN_DILATIONS,
            config.TCN_DROPOUT,
        )

        # --- Stage 3: Refinement ---
        self.stage3 = SingleStageTCN(
            config.NUM_CLASSES,
            config.TCN_CHANNELS,
            config.TCN_KERNEL_SIZE,
            config.TCN_DILATIONS,
            config.TCN_DROPOUT,
        )

    def forward(self, skeleton, audio):
        """
        Args:
            skeleton: (Batch, Time, SkelDim)
            audio: (Batch, Time, AudioDim)
        Returns:
            dict: {'p1': logits, 'p2': logits, 'p3': logits}
        """
        # Early Fusion
        x = torch.cat([skeleton, audio], dim=2)

        # Decoupled Gating
        x = self.gated_input(x)

        # Stage 1 Prediction (Logits)
        p1 = self.stage1(x)

        # Stage 2 Prediction (Refinement of P1 Softmax)
        # We pass probabilities, not logits, to the refinement stages
        p1_probs = F.softmax(p1, dim=2)
        p2 = self.stage2(p1_probs)

        # Stage 3 Prediction (Refinement of P2 Softmax)
        p2_probs = F.softmax(p2, dim=2)
        p3 = self.stage3(p2_probs)

        return {"p1": p1, "p2": p2, "p3": p3}
