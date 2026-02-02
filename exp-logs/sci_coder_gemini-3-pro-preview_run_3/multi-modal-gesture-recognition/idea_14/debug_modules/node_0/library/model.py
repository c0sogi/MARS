import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for 1D Temporal Data.
    Recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.
    """

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (Batch, Channel, Time)
        b, c, _ = x.size()
        # Squeeze: Global Average Pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: Bottleneck MLP
        y = self.fc(y).view(b, c, 1)
        # Scale
        return x * y.expand_as(x)


class GatedRefinementUnit(nn.Module):
    """
    Gated Dilated Temporal Convolutional Unit with Squeeze-and-Excitation.
    Implements the dilated convolution with gated activation (Tanh * Sigmoid).
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.0):
        super(GatedRefinementUnit, self).__init__()

        # Calculate padding to maintain temporal dimension (assuming 'same' padding logic)
        # For kernel=3, dilation=d: padding = d
        self.padding = dilation * (kernel_size - 1) // 2

        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels * 2,  # *2 for Gating (Feature + Gate)
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

        self.dropout = nn.Dropout(dropout)
        self.se_block = SEBlock(out_channels)

        # Residual connection adapter
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        # x: (Batch, Channel, Time)
        residual = x

        out = self.conv_dilated(x)

        # Split for Gated Activation
        # out: (Batch, 2*out_channels, Time)
        feature, gate = torch.chunk(out, 2, dim=1)

        # Gated Activation: tanh(feature) * sigmoid(gate)
        out = torch.tanh(feature) * torch.sigmoid(gate)

        out = self.dropout(out)

        # Attentive Refinement (SE-Block)
        out = self.se_block(out)

        # Residual Connection
        if self.downsample is not None:
            residual = self.downsample(residual)

        return out + residual


class RefinementStage(nn.Module):
    """
    A single stage of the refinement network.
    Consists of stacked GatedRefinementUnits with increasing dilation.
    """

    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(RefinementStage, self).__init__()

        self.layers = nn.ModuleList()

        # First layer: adapt input dimension (e.g., num_classes) to feature map dimension
        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        # Stack dilated layers
        for i in range(num_layers):
            dilation = 2**i
            self.layers.append(
                GatedRefinementUnit(
                    num_f_maps,
                    num_f_maps,
                    kernel_size=Config.TCN_KERNEL_SIZE,
                    dilation=dilation,
                    dropout=Config.DROPOUT,
                )
            )

        # Output head: map back to num_classes
        self.conv_1x1_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        # x: (Batch, InputDim, Time)
        out = self.conv_1x1_in(x)

        for layer in self.layers:
            out = layer(out)

        out = self.conv_1x1_out(out)
        return out


class RSKARN(nn.Module):
    """
    Robust Spatial-Kinematic Attentive Refinement Network.
    Stage 1: Bi-GRU Encoder (Spatial-Kinematic Sequence Encoder)
    Stage 2: Attentive Gated Refinement Module (Refines Stage 1 probs)
    Stage 3: Iterative Attentive Refinement (Refines Stage 2 probs)
    """

    def __init__(self):
        super(RSKARN, self).__init__()

        # ==========================================
        # Stage 1: Spatial-Kinematic Sequence Encoder
        # ==========================================
        # Input: Early Fusion Vector
        self.gru = nn.GRU(
            input_size=Config.INPUT_DIM,
            hidden_size=Config.GRU_HIDDEN_DIM,
            num_layers=Config.GRU_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT if Config.GRU_NUM_LAYERS > 1 else 0,
        )

        # Map GRU output (Hidden*2) to Classes
        self.stage1_fc = nn.Linear(Config.GRU_HIDDEN_DIM * 2, Config.NUM_CLASSES)

        # ==========================================
        # Stage 2: Attentive Gated Refinement
        # ==========================================
        # Input: Probabilities from Stage 1 (Dim = Num Classes)
        self.stage2 = RefinementStage(
            num_layers=Config.TCN_NUM_LAYERS,
            num_f_maps=Config.TCN_NUM_CHANNELS,
            dim=Config.NUM_CLASSES,
            num_classes=Config.NUM_CLASSES,
        )

        # ==========================================
        # Stage 3: Iterative Attentive Refinement
        # ==========================================
        # Input: Probabilities from Stage 2 (Dim = Num Classes)
        self.stage3 = RefinementStage(
            num_layers=Config.TCN_NUM_LAYERS,
            num_f_maps=Config.TCN_NUM_CHANNELS,
            dim=Config.NUM_CLASSES,
            num_classes=Config.NUM_CLASSES,
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Time, InputDim)

        Returns:
            stage1_logits: (Batch, Classes, Time)
            stage2_logits: (Batch, Classes, Time)
            stage3_logits: (Batch, Classes, Time)
        """
        # ==========================================
        # Stage 1 Forward
        # ==========================================
        # x: (B, T, D)
        gru_out, _ = self.gru(x)  # (B, T, Hidden*2)

        # Project to classes
        s1_logits = self.stage1_fc(gru_out)  # (B, T, C)

        # Transpose for TCN: (B, C, T)
        s1_logits_t = s1_logits.transpose(1, 2)

        # Apply Softmax to create probability bottleneck
        # We detach to stop gradients flowing back from S2 to S1 if desired,
        # but usually in cascaded refinement we allow end-to-end training.
        # The prompt implies a probability bottleneck, not necessarily a gradient stop.
        s1_probs = F.softmax(s1_logits_t, dim=1)

        # ==========================================
        # Stage 2 Forward
        # ==========================================
        # Input: s1_probs (B, C, T)
        s2_logits_t = self.stage2(s1_probs)  # (B, C, T)

        # Apply Softmax for next stage
        s2_probs = F.softmax(s2_logits_t, dim=1)

        # ==========================================
        # Stage 3 Forward
        # ==========================================
        # Input: s2_probs (B, C, T)
        s3_logits_t = self.stage3(s2_probs)  # (B, C, T)

        # Return logits in (Batch, Classes, Time) format for Loss calculation
        return s1_logits_t, s2_logits_t, s3_logits_t
