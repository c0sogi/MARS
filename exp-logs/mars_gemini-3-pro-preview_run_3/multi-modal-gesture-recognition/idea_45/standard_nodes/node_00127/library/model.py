import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class TemporalBlock(nn.Module):
    """
    A single block for the Gated Dilated TCN.
    Uses centered padding (padding=dilation for kernel_size=3) to maintain temporal dimension.
    Implements WaveNet-style Gated Activation: Tanh * Sigmoid.
    """

    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()

        # We need 2 * n_outputs filters (half for tanh, half for sigmoid)
        self.conv1 = nn.Conv1d(
            n_inputs,
            2 * n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, 1
        )  # 1x1 conv for residual aggregation

        # Residual connection handling: 1x1 conv if channel counts differ
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )

        self.init_weights()

    def init_weights(self):
        # Kaiming init for conv1
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.conv1.bias, 0)
        # Normal init for conv2
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        nn.init.constant_(self.conv2.bias, 0)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0, 0.01)
            nn.init.constant_(self.downsample.bias, 0)

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        # Gated Activation
        out1, out2 = out.chunk(2, dim=1)
        out = torch.tanh(out1) * torch.sigmoid(out2)

        out = self.dropout(out)
        out = self.conv2(out)

        return out + res


class SingleStageTCN(nn.Module):
    """
    A stack of TemporalBlocks forming one refinement stage.
    Follows the Split-Horizon dilation schedule.
    """

    def __init__(self, num_f_maps, num_classes):
        super(SingleStageTCN, self).__init__()
        layers = []
        dilation_schedule = config.DILATION_SCHEDULE  # [1, 2, 4, 8]
        kernel_size = config.KERNEL_SIZE  # 3

        # Stack layers based on schedule
        for d in dilation_schedule:
            # Padding = dilation ensures centered alignment for k=3
            layers.append(
                TemporalBlock(
                    num_f_maps,
                    num_f_maps,
                    kernel_size,
                    stride=1,
                    dilation=d,
                    padding=d,
                    dropout=0.2,
                )
            )

        self.network = nn.Sequential(*layers)
        # Final projection to classes
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.network(x)
        out = self.conv_out(out)
        return out


class SHPAMCN(nn.Module):
    """
    Split-Horizon Physically-Aligned Moderate-Capacity Network.
    Stage 1: Bi-GRU Encoder
    Stage 2: Gated Dilated TCN (Refinement)
    Stage 3: Gated Dilated TCN (Refinement)
    """

    def __init__(self):
        super(SHPAMCN, self).__init__()

        # ==========================================
        # Stage 1: Physically-Aligned Moderate-Capacity Encoder
        # ==========================================
        # Input: 180 (Skeleton) + 13 (Audio) = 193
        input_dim = 193
        hidden_dim = config.HIDDEN_SIZE  # 96

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=config.NUM_GRU_LAYERS,
            batch_first=True,
            bidirectional=True,
        )

        # Project GRU output (Hidden * 2) to Num Classes
        self.stage1_fc = nn.Linear(hidden_dim * 2, config.NUM_CLASSES)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)

        # ==========================================
        # Stage 2: Split-Horizon Non-Causal Refinement
        # ==========================================
        # TCN internal dimension (moderate capacity)
        tcn_dim = 64

        # Maps probabilities (NumClasses) to TCN feature space
        self.stage2_input_conv = nn.Conv1d(config.NUM_CLASSES, tcn_dim, 1)
        self.stage2_tcn = SingleStageTCN(tcn_dim, config.NUM_CLASSES)

        # ==========================================
        # Stage 3: Independent Split-Horizon Refinement
        # ==========================================
        self.stage3_input_conv = nn.Conv1d(config.NUM_CLASSES, tcn_dim, 1)
        self.stage3_tcn = SingleStageTCN(tcn_dim, config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x: Tensor (Batch, Time, Features)
        Returns:
            List of Tensors [(Batch, Time, Classes), ...] for deep supervision
        """
        # --- Stage 1 ---
        # GRU
        out, _ = self.gru(x)  # (B, T, Hidden*2)
        out = self.dropout(out)
        logits1 = self.stage1_fc(out)  # (B, T, Classes)

        # Prepare for Stage 2: Softmax -> Transpose
        probs1 = F.softmax(logits1, dim=2)  # (B, T, C)
        probs1_t = probs1.transpose(1, 2)  # (B, C, T) for Conv1d

        # --- Stage 2 ---
        out_s2 = self.stage2_input_conv(probs1_t)
        logits2_t = self.stage2_tcn(out_s2)  # (B, C, T)
        logits2 = logits2_t.transpose(1, 2)  # (B, T, C)

        # Prepare for Stage 3
        probs2 = F.softmax(logits2, dim=2)
        probs2_t = probs2.transpose(1, 2)

        # --- Stage 3 ---
        out_s3 = self.stage3_input_conv(probs2_t)
        logits3_t = self.stage3_tcn(out_s3)
        logits3 = logits3_t.transpose(1, 2)

        return [logits1, logits2, logits3]
