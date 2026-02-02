import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class GatedDilatedConv1d(nn.Module):
    """
    Gated Dilated Temporal Convolutional Layer.
    Implements: Output = Tanh(W_f * x) * Sigmoid(W_g * x)
    Uses standard convolutions with 'same' padding via dilation adjustment.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.0):
        super(GatedDilatedConv1d, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation

        # Padding to maintain sequence length: p = d * (k - 1) / 2
        # We assume kernel_size is odd (config.KERNEL_SIZE is 3)
        self.padding = (self.dilation * (self.kernel_size - 1)) // 2

        # Filter convolution (for Tanh)
        self.conv_f = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

        # Gate convolution (for Sigmoid)
        self.conv_g = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Channels, Time)
        out_f = self.conv_f(x)
        out_g = self.conv_g(x)

        out = torch.tanh(out_f) * torch.sigmoid(out_g)
        out = self.dropout(out)

        return out


class SawtoothTCN(nn.Module):
    """
    Refinement Stage using a Sawtooth Dilation Schedule.
    Input: Class Probabilities (Batch, Time, NumClasses)
    Output: Class Logits (Batch, Time, NumClasses)
    """

    def __init__(self, num_classes, hidden_dim, dilations, kernel_size, dropout):
        super(SawtoothTCN, self).__init__()

        self.layers = nn.ModuleList()

        # 1. Projection: Classes -> Hidden
        self.conv_in = nn.Conv1d(num_classes, hidden_dim, 1)

        # 2. Stack of Gated Dilated Convs
        for d in dilations:
            self.layers.append(
                GatedDilatedConv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
            )

        # 3. Projection: Hidden -> Classes
        self.conv_out = nn.Conv1d(hidden_dim, num_classes, 1)

    def forward(self, x):
        # x: (Batch, Time, NumClasses) -> Transpose to (Batch, NumClasses, Time)
        x = x.permute(0, 2, 1)

        out = self.conv_in(x)

        # Residual connections for deep TCN
        for layer in self.layers:
            out = out + layer(out)

        out = self.conv_out(out)

        # Transpose back: (Batch, NumClasses, Time) -> (Batch, Time, NumClasses)
        out = out.permute(0, 2, 1)

        return out


class WideBiGRU(nn.Module):
    """
    Stage 1: Wide-Capacity Kinematic Encoder.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1):
        super(WideBiGRU, self).__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Output dimension is hidden_dim * 2 (bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (Batch, Time, Features)

        # GRU Output: (Batch, Time, Hidden*2)
        out, _ = self.gru(x)

        # Project to classes
        logits = self.fc(out)

        return logits


class WESKN(nn.Module):
    """
    Wide-Encoder Sawtooth Kinematic Network (Idea 23).
    Three-Stage Cascade: BiGRU -> TCN -> TCN.
    """

    def __init__(self):
        super(WESKN, self).__init__()

        # Hyperparameters from config
        self.num_classes = config.NUM_CLASSES

        # Stage 1: Encoder
        self.stage1 = WideBiGRU(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_classes=self.num_classes,
            num_layers=config.ENCODER_LAYERS,
        )

        # Stage 2: Refinement (Sawtooth)
        # We use a hidden dimension of 64 for the refinement stages (standard practice in MS-TCN)
        # or we can use config.HIDDEN_DIM (128). Let's use 64 to keep parameter count reasonable
        # for the refinement part, as it operates on low-dim probabilities.
        tcn_hidden = 64

        self.stage2 = SawtoothTCN(
            num_classes=self.num_classes,
            hidden_dim=tcn_hidden,
            dilations=config.SAWTOOTH_DILATIONS,
            kernel_size=config.KERNEL_SIZE,
            dropout=config.DROPOUT,
        )

        # Stage 3: Independent Refinement (Sawtooth)
        self.stage3 = SawtoothTCN(
            num_classes=self.num_classes,
            hidden_dim=tcn_hidden,
            dilations=config.SAWTOOTH_DILATIONS,
            kernel_size=config.KERNEL_SIZE,
            dropout=config.DROPOUT,
        )

    def forward(self, x):
        # x: (Batch, Time, Features)

        # --- Stage 1 ---
        logits_1 = self.stage1(x)
        probs_1 = F.softmax(logits_1, dim=2)

        # --- Stage 2 ---
        # Input is strictly probabilities from Stage 1
        logits_2 = self.stage2(probs_1)
        probs_2 = F.softmax(logits_2, dim=2)

        # --- Stage 3 ---
        # Input is strictly probabilities from Stage 2
        logits_3 = self.stage3(probs_2)

        # Return dictionary for loss calculation
        return {
            "logits_1": logits_1,
            "logits_2": logits_2,
            "logits_3": logits_3,
            "probs_1": probs_1,
            "probs_2": probs_2,
            "probs_3": F.softmax(logits_3, dim=2),
        }


class CascadedLoss(nn.Module):
    """
    Loss function for WES-KN.
    Combines Weighted Cross Entropy (Deep Supervision) and Log-Space Smoothing.
    """

    def __init__(self):
        super(CascadedLoss, self).__init__()

        # Weighted Cross Entropy
        # Move weights to device during forward or init if device known.
        # Here we store them and move in forward.
        self.ce_weights = config.CLASS_WEIGHTS

        self.smoothing_weight = config.SMOOTHING_LOSS_WEIGHT
        self.smoothing_threshold = config.SMOOTHING_THRESHOLD

    def truncated_mse_loss(self, log_probs):
        """
        Computes Truncated MSE over temporal differences in log-space.
        Loss = mean( clamp( (log_p_t - log_p_{t-1})^2, max=threshold^2 ) )
        """
        # log_probs: (Batch, Time, Classes)

        # Diff: P_t - P_{t-1}
        # Slice 1: to End, Slice 0: to End-1
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        mse = diff**2

        # Truncate
        threshold_sq = self.smoothing_threshold**2
        truncated_mse = torch.clamp(mse, max=threshold_sq)

        return torch.mean(truncated_mse)

    def forward(self, outputs, targets):
        """
        outputs: Dict from model forward pass
        targets: (Batch, Time) LongTensor
        """
        device = targets.device
        weight = self.ce_weights.to(device)

        # Flatten targets for CrossEntropy: (Batch * Time)
        targets_flat = targets.view(-1)

        # --- Cross Entropy (Deep Supervision) ---
        # Reshape logits: (Batch * Time, Classes)
        l1 = outputs["logits_1"].reshape(-1, config.NUM_CLASSES)
        l2 = outputs["logits_2"].reshape(-1, config.NUM_CLASSES)
        l3 = outputs["logits_3"].reshape(-1, config.NUM_CLASSES)

        loss_ce_1 = F.cross_entropy(l1, targets_flat, weight=weight)
        loss_ce_2 = F.cross_entropy(l2, targets_flat, weight=weight)
        loss_ce_3 = F.cross_entropy(l3, targets_flat, weight=weight)

        loss_ce = loss_ce_1 + loss_ce_2 + loss_ce_3

        # --- Smoothing Loss ---
        # Apply to Stage 2 and Stage 3 outputs
        # Use log_softmax for numerical stability
        log_probs_2 = F.log_softmax(outputs["logits_2"], dim=2)
        log_probs_3 = F.log_softmax(outputs["logits_3"], dim=2)

        loss_smooth_2 = self.truncated_mse_loss(log_probs_2)
        loss_smooth_3 = self.truncated_mse_loss(log_probs_3)

        loss_smooth = self.smoothing_weight * (loss_smooth_2 + loss_smooth_3)

        return loss_ce + loss_smooth
