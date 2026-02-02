import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class InputGating(nn.Module):
    """
    Feature-Wise Input Gating: X_tilde = X * sigmoid(W * X + b)
    Dynamically suppresses noisy sensor channels before the recurrent backbone.
    """

    def __init__(self, input_dim):
        super(InputGating, self).__init__()
        self.gate = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (Batch, Time, Features)
        g = self.sigmoid(self.gate(x))
        return x * g


class DilatedResidualLayer(nn.Module):
    """
    Gated Dilated Temporal Convolutional Layer (WaveNet style).
    Uses centered padding for non-causal convolution.
    Structure:
        Input -> DilatedConv -> Split(Filter, Gate) -> Tanh * Sigmoid -> 1x1Conv -> Residual + Input
    """

    def __init__(self, channels, kernel_size, dilation, dropout):
        super(DilatedResidualLayer, self).__init__()

        self.conv_dilated = nn.Conv1d(
            channels,
            2 * channels,  # Output 2x channels for gating (Filter + Gate)
            kernel_size=kernel_size,
            padding=dilation,  # Centered padding: P = dilation for K=3
            dilation=dilation,
        )

        self.conv_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        out = self.conv_dilated(x)

        # Split for Gated Activation
        filter_out, gate_out = out.chunk(2, dim=1)

        # Tanh * Sigmoid
        out = torch.tanh(filter_out) * torch.sigmoid(gate_out)

        # 1x1 Conv projection
        out = self.conv_1x1(out)
        out = self.dropout(out)

        # Residual connection
        return x + out


class MSTCNStage(nn.Module):
    """
    Monotonic Non-Causal Gated Refinement Stage.
    Stacks DilatedResidualLayers with increasing dilation.
    """

    def __init__(self, num_layers, num_f_maps, dim, num_classes, dilations):
        super(MSTCNStage, self).__init__()

        self.layers = nn.ModuleList()

        # Input projection: Probabilities -> Hidden Dim
        self.conv_1x1_in = nn.Conv1d(num_classes, num_f_maps, kernel_size=1)

        # Stack dilated layers
        for i in range(num_layers):
            dilation = dilations[i] if i < len(dilations) else dilations[-1]
            self.layers.append(
                DilatedResidualLayer(
                    channels=num_f_maps,
                    kernel_size=config.KERNEL_SIZE,
                    dilation=dilation,
                    dropout=config.DROPOUT,
                )
            )

        # Output projection: Hidden Dim -> Logits
        self.conv_1x1_out = nn.Conv1d(num_f_maps, num_classes, kernel_size=1)

    def forward(self, x):
        # x shape: (Batch, Num_Classes, Time) - these are Softmax probabilities
        out = self.conv_1x1_in(x)

        for layer in self.layers:
            out = layer(out)

        out = self.conv_1x1_out(out)
        return out


class KinematicEncoder(nn.Module):
    """
    Stage 1: Gated High-Capacity Kinematic Encoder.
    InputGating -> Bi-GRU -> FC -> Logits
    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(KinematicEncoder, self).__init__()

        self.input_gating = InputGating(input_dim)

        # Bi-directional GRU
        # hidden_dim in config is total capacity.
        # For bidirectional, we use hidden_dim // 2 per direction if we want output to match hidden_dim,
        # or we just use hidden_dim per direction if we want high capacity.
        # Description says: "128 units per direction (256 total)".
        # config.HIDDEN_DIM is 256.
        self.gru = nn.GRU(
            input_dim,
            hidden_dim // 2,
            num_layers=config.ENCODER_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=config.DROPOUT if config.ENCODER_LAYERS > 1 else 0,
        )

        self.dropout = nn.Dropout(config.DROPOUT)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (Batch, Time, Features)

        # 1. Feature-wise Gating
        x = self.input_gating(x)

        # 2. Recurrent Backbone
        # GRU output: (Batch, Time, Hidden_Dim * Num_Directions)
        out, _ = self.gru(x)

        out = self.dropout(out)

        # 3. Projection
        logits = self.fc(out)

        # Return logits: (Batch, Time, Classes)
        return logits


class HSGKN(nn.Module):
    """
    Hierarchically-Scaled Gated-Kinematic Network.
    Three-Stage Cascaded Network:
    1. Gated Bi-GRU Encoder
    2. MS-TCN Refinement Stage 1
    3. MS-TCN Refinement Stage 2
    """

    def __init__(self):
        super(HSGKN, self).__init__()

        # Hyperparameters
        self.input_dim = config.INPUT_DIM
        self.hidden_dim = config.HIDDEN_DIM
        self.num_classes = config.NUM_CLASSES
        self.dilations = config.STAGE_DILATIONS
        self.num_layers = len(self.dilations)

        # Stage 1: Encoder
        self.stage1 = KinematicEncoder(
            self.input_dim, self.hidden_dim, self.num_classes
        )

        # Stage 2: Refinement
        # Note: TCN hidden dim is set to self.hidden_dim (256) for high capacity
        self.stage2 = MSTCNStage(
            num_layers=self.num_layers,
            num_f_maps=self.hidden_dim,
            dim=self.hidden_dim,
            num_classes=self.num_classes,
            dilations=self.dilations,
        )

        # Stage 3: Independent Refinement
        self.stage3 = MSTCNStage(
            num_layers=self.num_layers,
            num_f_maps=self.hidden_dim,
            dim=self.hidden_dim,
            num_classes=self.num_classes,
            dilations=self.dilations,
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (Batch, Time, Features)

        Returns:
            Dictionary containing outputs from all stages for Deep Supervision.
            Keys: 'stage1', 'stage2', 'stage3'
            Values: Logits tensor (Batch, Classes, Time) - Note the transpose for TCN compatibility
        """
        # --- Stage 1 ---
        # Encoder Output: (Batch, Time, Classes)
        s1_logits = self.stage1(x)

        # Prepare for TCN: Transpose to (Batch, Classes, Time)
        s1_logits_t = s1_logits.permute(0, 2, 1)

        # Apply Softmax to get probabilities for next stage
        s1_probs = F.softmax(s1_logits_t, dim=1)

        # --- Stage 2 ---
        # Input: Probabilities from Stage 1
        s2_logits_t = self.stage2(s1_probs)
        s2_probs = F.softmax(s2_logits_t, dim=1)

        # --- Stage 3 ---
        # Input: Probabilities from Stage 2
        s3_logits_t = self.stage3(s2_probs)

        return {"stage1": s1_logits_t, "stage2": s2_logits_t, "stage3": s3_logits_t}
