import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class GatedInputLayer(nn.Module):
    """
    Applies feature-wise gating to suppress noise in the input signal.
    Formula: x_tilde = x * sigmoid(W * x + b)
    """

    def __init__(self, input_dim):
        super(GatedInputLayer, self).__init__()
        self.gate_fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        gate = torch.sigmoid(self.gate_fc(x))
        return x * gate


class DilatedGatedBlock(nn.Module):
    """
    WaveNet-style Gated Dilated Convolutional Block.
    Uses centered padding for non-causal temporal modeling.
    """

    def __init__(self, channels, kernel_size, dilation, dropout):
        super(DilatedGatedBlock, self).__init__()

        # Dilated Convolution: Input -> 2 * Hidden (for Filter + Gate)
        # Padding = dilation ensures centered convolution for kernel_size=3
        self.conv_dilated = nn.Conv1d(
            channels, channels * 2, kernel_size, padding=dilation, dilation=dilation
        )

        # 1x1 Convolution for projection
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Channels, Time)
        residual = x

        out = self.conv_dilated(x)

        # Split into Filter and Gate
        filter_out, gate_out = out.chunk(2, dim=1)

        # Gated Activation: tanh(Filter) * sigmoid(Gate)
        out = torch.tanh(filter_out) * torch.sigmoid(gate_out)

        out = self.conv_1x1(out)
        out = self.dropout(out)

        return out + residual


class SingleStageTCN(nn.Module):
    """
    A single refinement stage consisting of stacked DilatedGatedBlocks.
    """

    def __init__(
        self, num_layers, num_f_maps, input_dim, output_dim, dilation_schedule
    ):
        super(SingleStageTCN, self).__init__()

        self.conv_in = nn.Conv1d(input_dim, num_f_maps, 1)

        self.layers = nn.ModuleList(
            [
                DilatedGatedBlock(
                    num_f_maps,
                    Config.MSTCN_KERNEL_SIZE,
                    dilation=dilation_schedule[i % len(dilation_schedule)],
                    dropout=Config.DROPOUT,
                )
                for i in range(num_layers)
            ]
        )

        self.conv_out = nn.Conv1d(num_f_maps, output_dim, 1)

    def forward(self, x):
        # x: (Batch, Time, Dim) -> Permute to (Batch, Dim, Time) for Conv1d
        x = x.permute(0, 2, 1)

        out = self.conv_in(x)

        for layer in self.layers:
            out = layer(out)

        out = self.conv_out(out)

        # Permute back to (Batch, Time, Dim)
        return out.permute(0, 2, 1)


class HNGKN(nn.Module):
    """
    Hierarchically-Normalized Gated-Kinematic Network (HNG-KN).
    Stage 1: Gated Bi-GRU Encoder
    Stage 2: Monotonic MS-TCN Refinement
    Stage 3: Independent Iterative Refinement
    """

    def __init__(self):
        super(HNGKN, self).__init__()

        # --- Stage 1: Gated High-Capacity Kinematic Encoder ---
        self.input_gate = GatedInputLayer(Config.INPUT_DIM)

        # Bi-GRU: Hidden dim 128 per direction -> 256 total output
        self.encoder = nn.GRU(
            Config.INPUT_DIM,
            Config.ENCODER_HIDDEN_DIM // 2,
            num_layers=Config.ENCODER_LAYERS,
            batch_first=True,
            bidirectional=True,
        )

        self.stage1_fc = nn.Linear(Config.ENCODER_HIDDEN_DIM, Config.NUM_CLASSES)

        # --- Stage 2: Monotonic Non-Causal Gated Refinement ---
        # Input: Probabilities from Stage 1 (Dim: NUM_CLASSES)
        self.stage2 = SingleStageTCN(
            num_layers=Config.MSTCN_LAYERS,
            num_f_maps=Config.MSTCN_HIDDEN_DIM,
            input_dim=Config.NUM_CLASSES,
            output_dim=Config.NUM_CLASSES,
            dilation_schedule=Config.MSTCN_DILATION_SCHEDULE,
        )

        # --- Stage 3: Independent Iterative Refinement ---
        # Input: Probabilities from Stage 2 (Dim: NUM_CLASSES)
        self.stage3 = SingleStageTCN(
            num_layers=Config.MSTCN_LAYERS,
            num_f_maps=Config.MSTCN_HIDDEN_DIM,
            input_dim=Config.NUM_CLASSES,
            output_dim=Config.NUM_CLASSES,
            dilation_schedule=Config.MSTCN_DILATION_SCHEDULE,
        )

    def forward(self, x):
        """
        Args:
            x: (Batch, Time, InputDim)
        Returns:
            logits1, logits2, logits3: Logits from each stage for Deep Supervision.
        """

        # --- Stage 1 ---
        x_gated = self.input_gate(x)

        # Encoder output: (Batch, Time, HiddenDim)
        enc_out, _ = self.encoder(x_gated)

        logits1 = self.stage1_fc(enc_out)

        # Apply Softmax to get probabilities for next stage
        # We detach to stop gradients flowing back through probabilities if desired,
        # but usually in MS-TCN gradients flow through. We keep flow.
        probs1 = F.softmax(logits1, dim=2)

        # --- Stage 2 ---
        logits2 = self.stage2(probs1)
        probs2 = F.softmax(logits2, dim=2)

        # --- Stage 3 ---
        logits3 = self.stage3(probs2)

        return logits1, logits2, logits3
