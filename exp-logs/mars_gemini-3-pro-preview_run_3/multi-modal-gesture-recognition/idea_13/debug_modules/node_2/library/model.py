import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel-wise attention.
    Recalibrates channel weights based on global context.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        reduced_channels = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T)
        b, c, t = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class GatedTCNBlock(nn.Module):
    """
    Gated Dilated Temporal Convolutional Block with SE Attention.
    Structure: Dilated Conv -> Split(Filter, Gate) -> Gating -> SE -> Dropout -> 1x1 Conv -> Residual
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(GatedTCNBlock, self).__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels * 2,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )

        self.se = SEBlock(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv_1x1 = nn.Conv1d(out_channels, in_channels, 1)

    def forward(self, x):
        # x: (B, C, T)
        residual = x

        # Dilated Conv
        out = self.conv_dilated(x)

        # Gating mechanism
        filter_out, gate_out = out.chunk(2, dim=1)
        out = torch.tanh(filter_out) * torch.sigmoid(gate_out)

        # Squeeze-and-Excitation
        out = self.se(out)

        # Dropout
        out = self.dropout(out)

        # Projection back to residual dimension
        out = self.conv_1x1(out)

        return out + residual


class RefinementModule(nn.Module):
    """
    Stage 2 & 3: Attentive Gated Refinement Module.
    Takes class probabilities as input, refines them using TCN blocks.
    """

    def __init__(
        self, num_classes, num_channels, num_layers=4, kernel_size=3, dropout=0.2
    ):
        super(RefinementModule, self).__init__()

        # Input projection: Probabilities -> Hidden Channels
        self.conv_in = nn.Conv1d(num_classes, num_channels, 1)

        # Stack of Gated TCN Blocks with increasing dilation
        layers = []
        for i in range(num_layers):
            dilation = 2**i
            layers.append(
                GatedTCNBlock(
                    num_channels, num_channels, kernel_size, dilation, dropout
                )
            )
        self.tcn_blocks = nn.Sequential(*layers)

        # Output projection: Hidden Channels -> Probabilities
        self.conv_out = nn.Conv1d(num_channels, num_classes, 1)

    def forward(self, x):
        # x: (B, NumClasses, T) - Probabilities from previous stage
        out = self.conv_in(x)
        out = self.tcn_blocks(out)
        out = self.conv_out(out)
        return out


class SequenceEncoder(nn.Module):
    """
    Stage 1: Spatial-Kinematic Sequence Encoder.
    Bi-GRU backbone to capture local temporal context from features.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(SequenceEncoder, self).__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (B, T, InputDim)
        self.gru.flatten_parameters()
        out, _ = self.gru(x)
        # out: (B, T, Hidden*2)
        logits = self.fc(out)
        # logits: (B, T, NumClasses)
        return logits


class SK_ARN(nn.Module):
    """
    Spatial-Kinematic Attentive Refinement Network.
    Three-Stage Cascaded Network:
    1. Bi-GRU Encoder
    2. Refinement Module
    3. Refinement Module
    """

    def __init__(self):
        super(SK_ARN, self).__init__()

        # Hyperparameters from Config
        self.num_classes = Config.NUM_CLASSES

        # Stage 1
        self.stage1 = SequenceEncoder(
            input_dim=Config.INPUT_DIM,
            hidden_dim=Config.GRU_HIDDEN_SIZE,
            num_layers=Config.GRU_NUM_LAYERS,
            num_classes=self.num_classes,
        )

        # Stage 2
        self.stage2 = RefinementModule(
            num_classes=self.num_classes,
            num_channels=Config.TCN_NUM_CHANNELS,
            num_layers=4,  # Dilation 1, 2, 4, 8
            kernel_size=Config.TCN_KERNEL_SIZE,
            dropout=Config.TCN_DROPOUT,
        )

        # Stage 3
        self.stage3 = RefinementModule(
            num_classes=self.num_classes,
            num_channels=Config.TCN_NUM_CHANNELS,
            num_layers=4,
            kernel_size=Config.TCN_KERNEL_SIZE,
            dropout=Config.TCN_DROPOUT,
        )

    def forward(self, x):
        # x: (B, T, InputDim)

        # --- Stage 1 ---
        # Output logits: (B, T, C)
        s1_logits = self.stage1(x)
        # Convert to probabilities for next stage
        s1_probs = F.softmax(s1_logits, dim=2)

        # Transpose for TCN: (B, C, T)
        s1_probs_t = s1_probs.transpose(1, 2)

        # --- Stage 2 ---
        s2_logits_t = self.stage2(s1_probs_t)
        s2_probs_t = F.softmax(s2_logits_t, dim=1)

        # --- Stage 3 ---
        s3_logits_t = self.stage3(s2_probs_t)

        # Transpose back to (B, T, C) for output consistency
        s2_logits = s2_logits_t.transpose(1, 2)
        s3_logits = s3_logits_t.transpose(1, 2)

        return {"stage1": s1_logits, "stage2": s2_logits, "stage3": s3_logits}


class TruncatedMSELoss(nn.Module):
    """
    Log-Space Smoothing Loss.
    Penalizes rapid changes in log-probabilities between adjacent frames.
    """

    def __init__(self, threshold=4.0):
        super(TruncatedMSELoss, self).__init__()
        self.threshold = threshold
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, logits):
        # logits: (B, T, C)
        # Use log_softmax to get log-probabilities
        log_probs = F.log_softmax(logits, dim=2)

        # Calculate diff: log_probs[t] - log_probs[t-1]
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Clamp gradients/values to avoid exploding loss on boundaries
        diff = torch.clamp(diff, min=-self.threshold, max=self.threshold)

        loss = torch.mean(diff**2)
        return loss


class CascadedLoss(nn.Module):
    """
    Combined loss for all three stages.
    L_total = L_CE(S1) + L_CE(S2) + L_smooth(S2) + L_CE(S3) + L_smooth(S3)
    """

    def __init__(self):
        super(CascadedLoss, self).__init__()

        # Class weights: Background (0) gets 0.2, others 1.0
        weights = torch.ones(Config.NUM_CLASSES)
        weights[Config.BACKGROUND_CLASS_ID] = Config.BACKGROUND_WEIGHT

        self.ce_loss = nn.CrossEntropyLoss(weight=weights)
        self.smooth_loss = TruncatedMSELoss()

        # Weights for smoothing component
        self.smooth_weight = 0.15

    def forward(self, outputs, targets):
        # outputs: dict with stage1, stage2, stage3 logits
        # targets: (B, T)

        # Flatten targets for CE: (B*T)
        targets_flat = targets.view(-1)

        # Stage 1 Loss
        s1_logits = outputs["stage1"].reshape(-1, Config.NUM_CLASSES)
        loss_s1 = self.ce_loss(s1_logits, targets_flat)

        # Stage 2 Loss
        s2_logits_flat = outputs["stage2"].reshape(-1, Config.NUM_CLASSES)
        loss_s2_ce = self.ce_loss(s2_logits_flat, targets_flat)
        loss_s2_smooth = self.smooth_loss(outputs["stage2"])
        loss_s2 = loss_s2_ce + self.smooth_weight * loss_s2_smooth

        # Stage 3 Loss
        s3_logits_flat = outputs["stage3"].reshape(-1, Config.NUM_CLASSES)
        loss_s3_ce = self.ce_loss(s3_logits_flat, targets_flat)
        loss_s3_smooth = self.smooth_loss(outputs["stage3"])
        loss_s3 = loss_s3_ce + self.smooth_weight * loss_s3_smooth

        return loss_s1 + loss_s2 + loss_s3
