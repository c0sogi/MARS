import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    INPUT_DIM,
    HIDDEN_DIM,
    NUM_LAYERS_LSTM,
    NUM_LAYERS_TCN,
    KERNEL_SIZE_TCN,
    DROPOUT,
    NUM_CLASSES,
)


class GatedActivationUnit(nn.Module):
    """
    Gated Activation Unit: tanh(W_f * x) * sigmoid(W_g * x)
    Used in the TCN refinement stages.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(GatedActivationUnit, self).__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.filter_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=padding
        )
        self.gate_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=padding
        )

    def forward(self, x):
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        return filter_out * gate_out


class DilatedResidualLayer(nn.Module):
    """
    A single dilated residual block for the TCN.
    Consists of a Gated Activation Unit, a 1x1 convolution, dropout, and a residual connection.
    """

    def __init__(self, channels, kernel_size, dilation, dropout):
        super(DilatedResidualLayer, self).__init__()
        self.gated_activation = GatedActivationUnit(
            channels, channels, kernel_size, dilation
        )
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.gated_activation(x)
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class SingleStageTCN(nn.Module):
    """
    A single stage of the TCN refinement module.
    Takes class/boundary probabilities as input, projects to hidden dim,
    processes via dilated blocks, and projects back to probabilities.
    """

    def __init__(
        self, input_channels, num_classes, num_layers, kernel_size, hidden_dim, dropout
    ):
        super(SingleStageTCN, self).__init__()

        # Input projection: (NumClasses + 1) -> HiddenDim
        self.conv_in = nn.Conv1d(input_channels, hidden_dim, 1)

        # Stack of dilated residual layers
        layers = []
        for i in range(num_layers):
            dilation = 2**i
            layers.append(
                DilatedResidualLayer(hidden_dim, kernel_size, dilation, dropout)
            )
        self.layers = nn.ModuleList(layers)

        # Output projection: HiddenDim -> (NumClasses + 1)
        self.conv_out = nn.Conv1d(hidden_dim, num_classes + 1, 1)

        self.num_classes = num_classes

    def forward(self, x, mask):
        """
        Args:
            x: Input tensor (B, InputChannels, T)
            mask: Binary mask (B, 1, T)
        """
        out = self.conv_in(x)

        for layer in self.layers:
            out = layer(out)
            # Apply mask inside the block if desired, but usually applied between stages.
            # However, applying it here ensures padding doesn't leak into valid regions via dilation.
            out = out * mask

        out = self.conv_out(out)

        # Split into Class and Boundary heads
        # out shape: (B, C+1, T)
        cls_logits = out[:, : self.num_classes, :]
        bnd_logits = out[:, self.num_classes :, :]

        # Apply activations
        cls_probs = F.softmax(cls_logits, dim=1)
        bnd_probs = torch.sigmoid(bnd_logits)

        return torch.cat([cls_probs, bnd_probs], dim=1)


class GeometricRecurrentEncoder(nn.Module):
    """
    Stage 1: Bi-LSTM Encoder processing geometric and audio features.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super(GeometricRecurrentEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Projection heads
        # Input to linear is hidden_dim * 2 (bidirectional)
        self.cls_head = nn.Linear(hidden_dim * 2, num_classes)
        self.bnd_head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x: (B, T, InputDim)

        # LSTM output: (B, T, HiddenDim*2)
        lstm_out, _ = self.lstm(x)

        # Project
        cls_logits = self.cls_head(lstm_out)  # (B, T, NumClasses)
        bnd_logits = self.bnd_head(lstm_out)  # (B, T, 1)

        # Activations
        cls_probs = F.softmax(cls_logits, dim=2)
        bnd_probs = torch.sigmoid(bnd_logits)

        # Concatenate: (B, T, NumClasses + 1)
        return torch.cat([cls_probs, bnd_probs], dim=2)


class SBG_CRCN(nn.Module):
    """
    Soft-Boundary Geometric Gated-Cascaded Recurrent-Convolutional Network.
    Stage 1: Geometric Recurrent Encoder
    Stage 2: Gated MS-TCN Refinement
    Stage 3: Gated MS-TCN Sharpening
    """

    def __init__(self):
        super(SBG_CRCN, self).__init__()

        # Stage 1
        self.stage1 = GeometricRecurrentEncoder(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS_LSTM,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT,
        )

        # Stage 2
        # Input channels = NumClasses (21) + Boundary (1) = 22
        self.stage2 = SingleStageTCN(
            input_channels=NUM_CLASSES + 1,
            num_classes=NUM_CLASSES,
            num_layers=NUM_LAYERS_TCN,
            kernel_size=KERNEL_SIZE_TCN,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
        )

        # Stage 3
        self.stage3 = SingleStageTCN(
            input_channels=NUM_CLASSES + 1,
            num_classes=NUM_CLASSES,
            num_layers=NUM_LAYERS_TCN,
            kernel_size=KERNEL_SIZE_TCN,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
        )

    def forward(self, x, mask):
        """
        Args:
            x: Input features (B, T, InputDim)
            mask: Sequence mask (B, T)

        Returns:
            dict: Outputs from all stages for Deep Supervision.
                  Each output is a dict with 'class_probs' and 'boundary_probs'.
        """
        outputs = {}

        # Prepare Mask
        # LSTM keeps (B, T, C), TCN needs (B, C, T)
        # Mask for LSTM: (B, T, 1)
        mask_lstm = mask.unsqueeze(2).float()
        # Mask for TCN: (B, 1, T)
        mask_tcn = mask.unsqueeze(1).float()

        # --- Stage 1 ---
        out1 = self.stage1(x)  # (B, T, C+1)
        out1 = out1 * mask_lstm  # Inter-stage masking

        outputs["stage1"] = {
            "class_probs": out1[:, :, :NUM_CLASSES],
            "boundary_probs": out1[:, :, NUM_CLASSES:],
        }

        # Prepare for TCN: Permute to (B, C+1, T)
        out1_tcn = out1.permute(0, 2, 1)

        # --- Stage 2 ---
        out2_tcn = self.stage2(out1_tcn, mask_tcn)  # (B, C+1, T)
        out2_tcn = out2_tcn * mask_tcn

        # Permute back for storage/consistency: (B, T, C+1)
        out2 = out2_tcn.permute(0, 2, 1)

        outputs["stage2"] = {
            "class_probs": out2[:, :, :NUM_CLASSES],
            "boundary_probs": out2[:, :, NUM_CLASSES:],
        }

        # --- Stage 3 ---
        out3_tcn = self.stage3(out2_tcn, mask_tcn)
        out3_tcn = out3_tcn * mask_tcn

        out3 = out3_tcn.permute(0, 2, 1)

        outputs["stage3"] = {
            "class_probs": out3[:, :, :NUM_CLASSES],
            "boundary_probs": out3[:, :, NUM_CLASSES:],
        }

        return outputs
