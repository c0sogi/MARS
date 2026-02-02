import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class GatedDilatedConvLayer(nn.Module):
    """
    A Dilated Convolutional Layer with Gated Activation (WaveNet style).
    Output = Tanh(Filter * x) * Sigmoid(Gate * x)
    Includes a residual connection and dropout.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(GatedDilatedConvLayer, self).__init__()

        # Convolution producing 2 * out_channels (for filter and gate)
        self.conv = nn.Conv1d(
            in_channels,
            2 * out_channels,
            kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
        )

        self.dropout = nn.Dropout(dropout)
        self.conv_out = nn.Conv1d(out_channels, out_channels, 1)

        # Projection for residual connection if channel dimensions change
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        # x: (Batch, Channels, Time)
        residual = x if self.downsample is None else self.downsample(x)

        out = self.conv(x)

        # Split into filter and gate
        filter_gate, gate_gate = out.chunk(2, dim=1)

        # Gated activation
        out = torch.tanh(filter_gate) * torch.sigmoid(gate_gate)

        out = self.dropout(out)
        out = self.conv_out(out)

        return out + residual


class RefinementStage(nn.Module):
    """
    A stack of Gated Dilated Convolutional Layers designed to refine class probabilities.
    Input: Class Probabilities (Batch, NumClasses, Time)
    Output: Refined Class Logits (Batch, NumClasses, Time)
    """

    def __init__(self, num_classes, hidden_channels, kernel_size, dropout):
        super(RefinementStage, self).__init__()

        self.layers = nn.ModuleList()
        # Exponentially increasing dilation factors
        dilations = [1, 2, 4, 8, 16]

        # Ensure hidden_channels list matches dilations length
        assert len(hidden_channels) == len(
            dilations
        ), "Hidden channels list must match number of dilation layers"

        # First layer: Map from NumClasses to Hidden Dimension
        self.layers.append(
            GatedDilatedConvLayer(
                num_classes, hidden_channels[0], kernel_size, dilations[0], dropout
            )
        )

        # Subsequent layers: Map Hidden to Hidden
        for i in range(1, len(dilations)):
            self.layers.append(
                GatedDilatedConvLayer(
                    hidden_channels[i - 1],
                    hidden_channels[i],
                    kernel_size,
                    dilations[i],
                    dropout,
                )
            )

        # Final projection: Map back to NumClasses
        self.final_conv = nn.Conv1d(hidden_channels[-1], num_classes, 1)

    def forward(self, x):
        # x: (Batch, NumClasses, Time) - Input probabilities
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.final_conv(out)
        return out


class SequenceEncoder(nn.Module):
    """
    Stage 1: Bi-GRU Encoder.
    Processes raw features and outputs initial frame-wise predictions.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super(SequenceEncoder, self).__init__()

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        # Output dimension is hidden_dim * 2 because of bidirectional GRU
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        out, _ = self.gru(x)
        out = self.dropout(out)
        out = self.fc(out)

        # Output: (Batch, Time, NumClasses)
        # Permute to (Batch, NumClasses, Time) for compatibility with TCN layers
        return out.permute(0, 2, 1)


class KC_IRN(nn.Module):
    """
    Kinematically-Consistent Iterative Refinement Network.
    Stage 1: Bi-GRU Encoder
    Stage 2: Refinement TCN (Gated)
    Stage 3: Refinement TCN (Gated)
    """

    def __init__(self):
        super(KC_IRN, self).__init__()

        # Stage 1: Encoder
        self.stage1 = SequenceEncoder(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.GRU_HIDDEN_DIM,
            num_layers=config.GRU_LAYERS,
            num_classes=config.NUM_CLASSES,
            dropout=config.GRU_DROPOUT,
        )

        # Stage 2: First Refinement
        self.stage2 = RefinementStage(
            num_classes=config.NUM_CLASSES,
            hidden_channels=config.TCN_NUM_CHANNELS,
            kernel_size=config.TCN_KERNEL_SIZE,
            dropout=config.TCN_DROPOUT,
        )

        # Stage 3: Second Refinement
        self.stage3 = RefinementStage(
            num_classes=config.NUM_CLASSES,
            hidden_channels=config.TCN_NUM_CHANNELS,
            kernel_size=config.TCN_KERNEL_SIZE,
            dropout=config.TCN_DROPOUT,
        )

    def forward(self, x):
        """
        Args:
            x: Input features (Batch, Time, InputDim)
        Returns:
            list: [log_probs_stage1, log_probs_stage2, log_probs_stage3]
                  Each tensor has shape (Batch, NumClasses, Time)
        """
        # --- Stage 1 ---
        out1_logits = self.stage1(x)
        out1_log_probs = F.log_softmax(out1_logits, dim=1)

        # Convert to probabilities [0, 1] to pass to next stage (Information Bottleneck)
        out1_probs = torch.exp(out1_log_probs)

        # --- Stage 2 ---
        out2_logits = self.stage2(out1_probs)
        out2_log_probs = F.log_softmax(out2_logits, dim=1)

        out2_probs = torch.exp(out2_log_probs)

        # --- Stage 3 ---
        out3_logits = self.stage3(out2_probs)
        out3_log_probs = F.log_softmax(out3_logits, dim=1)

        return [out1_log_probs, out2_log_probs, out3_log_probs]
