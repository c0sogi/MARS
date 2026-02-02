import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add library path to access config
sys.path.append(os.path.abspath("./library"))
from config import Config


class DualHeadBiGRU(nn.Module):
    """
    Stage 1: Bi-Directional GRU Encoder with Dual Heads.
    Outputs both gesture class logits and boundary detection logits.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super(DualHeadBiGRU, self).__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)

        # Dual Heads
        # Bidirectional GRU outputs hidden_dim * 2
        self.cls_head = nn.Linear(hidden_dim * 2, num_classes)
        self.bnd_head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x: (Batch, Input_Dim, Time) -> Permute to (Batch, Time, Input_Dim) for GRU
        x = x.permute(0, 2, 1)

        features, _ = self.gru(x)
        features = self.dropout(features)

        # Heads
        cls_logits = self.cls_head(features)  # (Batch, Time, NumClasses)
        bnd_logits = self.bnd_head(features)  # (Batch, Time, 1)

        # Permute back to (Batch, Channel, Time) for consistency with CNNs/TCNs
        cls_logits = cls_logits.permute(0, 2, 1)
        bnd_logits = bnd_logits.permute(0, 2, 1)

        return cls_logits, bnd_logits


class TemporalBlock(nn.Module):
    """
    Dilated Temporal Convolutional Block with Gated Activations.
    """

    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()

        # We use a single conv to project to 2 * n_outputs, then split for Gating
        # Filter: tanh, Gate: sigmoid
        self.conv1 = nn.Conv1d(
            n_inputs,
            2 * n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, 1)
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection handling
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
        if self.downsample is not None:
            nn.init.kaiming_normal_(
                self.downsample.weight, mode="fan_out", nonlinearity="relu"
            )

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.dropout1(out)

        # Gated Activation: tanh(f) * sigmoid(g)
        filter_out, gate_out = out.chunk(2, dim=1)
        out = torch.tanh(filter_out) * torch.sigmoid(gate_out)

        out = self.conv2(out)
        out = self.dropout2(out)

        return out + res


class GatedRefinementStage(nn.Module):
    """
    Stages 2 & 3: Stack of Dilated TCNs for Iterative Refinement.
    Takes probabilities from previous stage as input.
    """

    def __init__(
        self, input_dim, num_classes, hidden_dim, num_layers, kernel_size, dropout
    ):
        super(GatedRefinementStage, self).__init__()

        # 1x1 Conv to project input probabilities to hidden dim
        self.conv_in = nn.Conv1d(input_dim, hidden_dim, 1)

        layers = []
        for i in range(num_layers):
            dilation_size = 2**i
            layers += [
                TemporalBlock(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size // 2,
                    dropout=dropout,
                )
            ]

        self.tcn = nn.Sequential(*layers)

        # Heads
        self.cls_head = nn.Conv1d(hidden_dim, num_classes, 1)
        self.bnd_head = nn.Conv1d(hidden_dim, 1, 1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.tcn(out)

        cls_logits = self.cls_head(out)
        bnd_logits = self.bnd_head(out)

        return cls_logits, bnd_logits


class BAKC_IRN(nn.Module):
    """
    Boundary-Aware Kinematically-Consistent Iterative Refinement Network.
    Combines DualHeadBiGRU encoder with multiple GatedRefinementStages.
    """

    def __init__(self):
        super(BAKC_IRN, self).__init__()

        # Config parameters
        input_dim = Config.INPUT_DIM
        hidden_dim = Config.HIDDEN_DIM
        num_classes = Config.NUM_CLASSES
        num_stages = Config.NUM_STAGES
        dropout = Config.DROPOUT
        kernel_size = Config.KERNEL_SIZE

        # Stage 1: Encoder
        self.stage1 = DualHeadBiGRU(input_dim, hidden_dim, num_classes, dropout)

        # Refinement Stages
        # Input to refinement is (NumClasses + 1) -> Probabilities
        refinement_input_dim = num_classes + 1

        # We use num_stages - 1 refinement stages (Stage 1 is the GRU)
        self.stages = nn.ModuleList(
            [
                GatedRefinementStage(
                    refinement_input_dim,
                    num_classes,
                    hidden_dim,
                    num_layers=10,  # Standard MS-TCN depth
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_stages - 1)
            ]
        )

    def forward(self, x):
        outputs = []

        # Stage 1
        cls_logits, bnd_logits = self.stage1(x)
        outputs.append({"cls": cls_logits, "bnd": bnd_logits})

        # Prepare input for next stage (Softmax/Sigmoid)
        cls_probs = F.softmax(cls_logits, dim=1)
        bnd_probs = torch.sigmoid(bnd_logits)
        x_in = torch.cat([cls_probs, bnd_probs], dim=1)

        # Refinement Stages
        for stage in self.stages:
            cls_logits, bnd_logits = stage(x_in)
            outputs.append({"cls": cls_logits, "bnd": bnd_logits})

            # Prepare input for next stage
            cls_probs = F.softmax(cls_logits, dim=1)
            bnd_probs = torch.sigmoid(bnd_logits)
            x_in = torch.cat([cls_probs, bnd_probs], dim=1)

        return outputs


class BoundaryAwareLoss(nn.Module):
    """
    Multi-Task Loss Function.
    Combines Weighted Cross Entropy, Binary Cross Entropy, and Boundary-Adaptive Smoothing.
    """

    def __init__(self):
        super(BoundaryAwareLoss, self).__init__()

        # Class weights: Down-weight background class (index 0)
        weights = torch.ones(Config.NUM_CLASSES)
        weights[0] = Config.BG_WEIGHT

        # Move weights to device dynamically in forward or assume device placement
        self.register_buffer("class_weights", weights)

        self.cls_criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.bnd_criterion = nn.BCEWithLogitsLoss()

        self.lambda_bnd = Config.LAMBDA_BND
        self.lambda_smooth = Config.LAMBDA_SMOOTH

    def forward(self, outputs, cls_targets, bnd_targets):
        """
        outputs: List of dicts {'cls': logits, 'bnd': logits} from each stage
        cls_targets: (Batch, Time) LongTensor
        bnd_targets: (Batch, Time) FloatTensor
        """
        total_loss = 0

        for out in outputs:
            cls_logits = out["cls"]  # (Batch, NumClasses, Time)
            bnd_logits = out["bnd"]  # (Batch, 1, Time)

            # 1. Classification Loss
            loss_cls = self.cls_criterion(cls_logits, cls_targets)

            # 2. Boundary Loss
            loss_bnd = self.bnd_criterion(bnd_logits.squeeze(1), bnd_targets)

            # 3. Boundary-Adaptive Smoothing Loss
            # Formula: (1/T) * sum( (1 - y_bnd) * || log_p(t) - log_p(t-1) ||^2 )

            # Get Log Probs for numerical stability
            log_probs = F.log_softmax(cls_logits, dim=1)  # (B, C, T)

            # Calculate temporal difference: log_p(t) - log_p(t-1)
            # Slice to align: diff[:, :, i] = log_probs[:, :, i+1] - log_probs[:, :, i]
            diff = log_probs[:, :, 1:] - log_probs[:, :, :-1]  # (B, C, T-1)

            # Squared Norm over Class dimension
            diff_sq = torch.sum(diff**2, dim=1)  # (B, T-1)

            # Get Boundary Weights (1 - y_bnd)
            # We align with the first frame of the pair (t)
            bnd_weight = 1.0 - bnd_targets[:, :-1]  # (B, T-1)

            # Clamp to [0, 1] for safety
            bnd_weight = torch.clamp(bnd_weight, 0, 1)

            # Compute weighted mean
            loss_smooth = torch.mean(bnd_weight * diff_sq)

            # Sum components
            total_loss += (
                loss_cls + self.lambda_bnd * loss_bnd + self.lambda_smooth * loss_smooth
            )

        return total_loss
