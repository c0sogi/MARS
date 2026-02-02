import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class BiGRUEncoder(nn.Module):
    """
    Stage 1: Wide-Capacity Kinematic Encoder.
    Processes early-fusion features using a Bi-GRU and outputs initial Log-Probabilities.
    """

    def __init__(self):
        super(BiGRUEncoder, self).__init__()
        self.input_dim = Config.INPUT_DIM
        self.hidden_dim = Config.ENCODER_HIDDEN_DIM  # 128
        self.num_layers = Config.ENCODER_NUM_LAYERS
        self.num_classes = Config.NUM_CLASSES
        self.dropout_p = Config.ENCODER_DROPOUT

        # Bi-Directional GRU
        # Output dim will be hidden_dim * 2 (256)
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_p if self.num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(self.dropout_p)
        self.fc = nn.Linear(self.hidden_dim * 2, self.num_classes)

    def forward(self, x):
        # x shape: (Batch, Time, InputDim)

        # GRU Forward
        # out shape: (Batch, Time, HiddenDim * 2)
        out, _ = self.gru(x)

        out = self.dropout(out)

        # Projection to classes
        # logits shape: (Batch, Time, NumClasses)
        logits = self.fc(out)

        # Log-Softmax for stable additive refinement later
        # log_probs shape: (Batch, Time, NumClasses)
        log_probs = F.log_softmax(logits, dim=2)

        return log_probs


class GatedDilatedBlock(nn.Module):
    """
    Building block for the Refinement Stage.
    Dilated Conv -> Gated Activation (Tanh * Sigmoid) -> 1x1 Conv -> Residual
    """

    def __init__(self, channels, kernel_size, dilation, dropout):
        super(GatedDilatedBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation // 2

        # Filter convolution
        self.filter_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

        # Gate convolution
        self.gate_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

        self.dropout = nn.Dropout(dropout)
        self.conv_1x1 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        # x shape: (Batch, Channels, Time)

        # Compute filter and gate
        filter_out = self.filter_conv(x)
        gate_out = self.gate_conv(x)

        # Gated Activation
        activation = torch.tanh(filter_out) * torch.sigmoid(gate_out)

        # Dropout
        activation = self.dropout(activation)

        # 1x1 Projection
        out = self.conv_1x1(activation)

        # Residual Connection
        return x + out


class ResidualRefinementStage(nn.Module):
    """
    Stages 2 & 3: Residual Sawtooth Refinement.
    Learns an additive correction to the input log-probabilities.
    L_out = L_in + TCN(L_in)
    """

    def __init__(self):
        super(ResidualRefinementStage, self).__init__()
        self.num_classes = Config.NUM_CLASSES
        self.tcn_channels = Config.TCN_CHANNELS
        self.kernel_size = Config.TCN_KERNEL_SIZE
        self.dropout = Config.TCN_DROPOUT
        self.dilation_schedule = Config.DILATION_SCHEDULE

        # Adapter: Project classes to hidden channels
        self.conv_in = nn.Conv1d(self.num_classes, self.tcn_channels, kernel_size=1)

        # Stack of Gated Dilated Blocks
        self.layers = nn.ModuleList()
        for dilation in self.dilation_schedule:
            self.layers.append(
                GatedDilatedBlock(
                    channels=self.tcn_channels,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    dropout=self.dropout,
                )
            )

        # Adapter: Project hidden channels back to classes (correction)
        self.conv_out = nn.Conv1d(self.tcn_channels, self.num_classes, kernel_size=1)

    def forward(self, log_probs_in):
        # log_probs_in shape: (Batch, Time, NumClasses)
        # Permute for CNN: (Batch, NumClasses, Time)
        x = log_probs_in.permute(0, 2, 1)

        # Initial projection
        features = self.conv_in(x)

        # Pass through TCN layers
        for layer in self.layers:
            features = layer(features)

        # Compute correction
        correction = self.conv_out(features)

        # Permute back: (Batch, Time, NumClasses)
        correction = correction.permute(0, 2, 1)

        # Additive Residual Correction
        log_probs_out = log_probs_in + correction

        return log_probs_out


class RLK_RN(nn.Module):
    """
    Residual Log-Kinematic Refinement Network.
    Stage 1: BiGRU Encoder
    Stage 2: Refinement 1
    Stage 3: Refinement 2
    """

    def __init__(self):
        super(RLK_RN, self).__init__()

        # Stage 1
        self.stage1_encoder = BiGRUEncoder()

        # Stage 2
        self.stage2_refinement = ResidualRefinementStage()

        # Stage 3
        self.stage3_refinement = ResidualRefinementStage()

    def forward(self, x):
        """
        Args:
            x: Input features (Batch, Time, InputDim)

        Returns:
            list: [stage1_log_probs, stage2_log_probs, stage3_log_probs]
        """
        # Stage 1
        l1 = self.stage1_encoder(x)

        # Stage 2 (Refines L1)
        l2 = self.stage2_refinement(l1)

        # Stage 3 (Refines L2)
        l3 = self.stage3_refinement(l2)

        return [l1, l2, l3]
