import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SimplifiedGatedBlock(nn.Module):
    """
    Simplified Gated Activation Block for MS-TCN.
    Structure: Z = tanh(W_f * X) * sigmoid(W_g * X)
    Residual: Y = X + Z

    Crucially, this block removes the 1x1 output projection layer in the residual path
    to facilitate gradient flow and reduce complexity.
    """

    def __init__(self, channels, kernel_size, dilation, dropout):
        super(SimplifiedGatedBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=2 * channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Channels, Time)
        out = self.conv(x)

        # Split into filter and gate
        filter_out, gate_out = out.chunk(2, dim=1)

        # Gated Activation
        z = torch.tanh(filter_out) * torch.sigmoid(gate_out)
        z = self.dropout(z)

        # Residual Connection (Identity mapping, no projection)
        return x + z


class BiLSTMEncoder(nn.Module):
    """
    Stage 1: Geometric Recurrent Encoder.
    Processes raw features using Bi-LSTM and outputs initial probabilities.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(BiLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Output projection dimension is hidden_dim * 2 (bidirectional)
        proj_dim = hidden_dim * 2

        # Heads
        self.cls_head = nn.Linear(proj_dim, num_classes)
        self.bnd_head = nn.Linear(proj_dim, 1)
        self.fg_head = nn.Linear(proj_dim, 1)

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        lstm_out, _ = self.lstm(x)  # (Batch, Time, Hidden*2)

        # Projections
        logits_cls = self.cls_head(lstm_out)
        logits_bnd = self.bnd_head(lstm_out)
        logits_fg = self.fg_head(lstm_out)

        # Activations
        probs_cls = F.softmax(logits_cls, dim=2)
        probs_bnd = torch.sigmoid(logits_bnd)
        probs_fg = torch.sigmoid(logits_fg)

        # Concatenate: (Batch, Time, NumClasses + 1 + 1)
        return torch.cat([probs_cls, probs_bnd, probs_fg], dim=2)


class GatedMSTCN(nn.Module):
    """
    Stage 2 & 3: Hierarchical Gated Refinement.
    Refines probabilities using dilated convolutions.
    """

    def __init__(
        self, input_dim, hidden_dim, num_layers, kernel_size, dropout, num_classes
    ):
        super(GatedMSTCN, self).__init__()

        # Input Projection: (NumClasses + 2) -> HiddenDim
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        # Stack of Gated Blocks
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2**i
            self.layers.append(
                SimplifiedGatedBlock(hidden_dim, kernel_size, dilation, dropout)
            )

        # Output Projection: HiddenDim -> (NumClasses + 2)
        self.output_proj = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)

        self.num_classes = num_classes

    def forward(self, x, mask):
        # x: (Batch, Time, InputChannels)
        # mask: (Batch, Time)

        # Permute for Conv1d: (Batch, InputChannels, Time)
        x = x.permute(0, 2, 1)

        # Input Projection
        out = self.input_proj(x)

        # Apply Blocks
        for layer in self.layers:
            out = layer(out)

        # Output Projection
        out = self.output_proj(out)

        # Permute back: (Batch, Time, InputChannels)
        out = out.permute(0, 2, 1)

        # Split and Apply Activations
        # InputChannels = NumClasses + 1 (Bnd) + 1 (Fg)
        logits_cls = out[:, :, : self.num_classes]
        logits_bnd = out[:, :, self.num_classes : self.num_classes + 1]
        logits_fg = out[:, :, self.num_classes + 1 :]

        probs_cls = F.softmax(logits_cls, dim=2)
        probs_bnd = torch.sigmoid(logits_bnd)
        probs_fg = torch.sigmoid(logits_fg)

        # Concatenate
        result = torch.cat([probs_cls, probs_bnd, probs_fg], dim=2)

        # Apply Mask (Zero out padding)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            result = result * mask_expanded

        return result


class GHG_CRCN(nn.Module):
    """
    Geometric Hierarchical Gated-Cascaded Recurrent-Convolutional Network.

    Stage 1: Bi-LSTM Encoder (Geometry -> Probs)
    Stage 2: Gated MS-TCN (Probs -> Refined Probs)
    Stage 3: Gated MS-TCN (Refined Probs -> Final Probs)
    """

    def __init__(self):
        super(GHG_CRCN, self).__init__()

        # Dimensions
        self.input_dim = Config.INPUT_DIM
        self.num_classes = Config.NUM_CLASSES

        # The intermediate representation has (NumClasses + 1 Bnd + 1 Fg) channels
        self.stage_io_dim = self.num_classes + 1 + 1

        # Stage 1
        self.stage1 = BiLSTMEncoder(
            input_dim=self.input_dim,
            hidden_dim=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_LAYERS,
            num_classes=self.num_classes,
        )

        # Stage 2
        self.stage2 = GatedMSTCN(
            input_dim=self.stage_io_dim,
            hidden_dim=Config.TCN_CHANNELS,
            num_layers=Config.TCN_LAYERS,
            kernel_size=Config.TCN_KERNEL_SIZE,
            dropout=Config.DROPOUT,
            num_classes=self.num_classes,
        )

        # Stage 3
        self.stage3 = GatedMSTCN(
            input_dim=self.stage_io_dim,
            hidden_dim=Config.TCN_CHANNELS,
            num_layers=Config.TCN_LAYERS,
            kernel_size=Config.TCN_KERNEL_SIZE,
            dropout=Config.DROPOUT,
            num_classes=self.num_classes,
        )

    def forward(self, x, mask):
        """
        Args:
            x: (Batch, Time, InputDim)
            mask: (Batch, Time) Boolean mask where True is valid, False is padding.
        Returns:
            out1, out2, out3: Outputs from each stage for Deep Supervision.
                              Shape: (Batch, Time, NumClasses + 2)
        """
        # --- Stage 1 ---
        out1 = self.stage1(x)

        # Inter-Stage Masking
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            out1 = out1 * mask_expanded

        # --- Stage 2 ---
        out2 = self.stage2(out1, mask)

        # --- Stage 3 ---
        out3 = self.stage3(out2, mask)

        return out1, out2, out3
