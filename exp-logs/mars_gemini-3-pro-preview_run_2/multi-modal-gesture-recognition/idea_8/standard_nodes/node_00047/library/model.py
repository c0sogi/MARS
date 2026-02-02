import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class DilatedResidualLayer(nn.Module):
    """
    A single dilated residual layer for the TCN.
    Consists of: Dilated Conv1d -> ReLU -> Dropout -> 1x1 Conv1d -> ReLU -> Dropout -> Residual
    """

    def __init__(self, channels, kernel_size, dilation, dropout):
        super(DilatedResidualLayer, self).__init__()

        # Calculate padding to maintain temporal dimension: (k-1) * d / 2
        # For odd kernel size k=3: padding = d
        self.conv_dilated = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation,
        )

        self.conv_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv_dilated(x)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv_1x1(out)
        out = F.relu(out)
        out = self.dropout(out)

        return x + out


class SingleStageTCN(nn.Module):
    """
    A single refinement stage using Multi-Stage TCN architecture.
    Refines probability maps from the previous stage.
    """

    def __init__(self, num_layers, num_f_maps, num_classes, kernel_size, dropout):
        super(SingleStageTCN, self).__init__()

        self.conv_in = nn.Conv1d(num_classes, num_f_maps, kernel_size=1)

        self.layers = nn.ModuleList(
            [
                DilatedResidualLayer(
                    channels=num_f_maps,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, kernel_size=1)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input probabilities [Batch, NumClasses, Time]
            mask: Binary mask [Batch, 1, Time] indicating valid frames
        """
        out = self.conv_in(x)

        for layer in self.layers:
            out = layer(out)

        out = self.conv_out(out)

        # Apply mask if provided to zero out padding regions in the logits
        if mask is not None:
            out = out * mask

        return out


class DSR_CRCN(nn.Module):
    """
    Dual-Stage Refined Cascaded Recurrent-Convolutional Network.
    Stage 0: Bi-LSTM Encoder (Generation)
    Stage 1: MS-TCN Refinement 1
    Stage 2: MS-TCN Refinement 2
    """

    def __init__(self):
        super(DSR_CRCN, self).__init__()

        # ==========================
        # Stage 0: Generation (LSTM)
        # ==========================
        self.lstm = nn.LSTM(
            input_size=config.INPUT_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=config.BIDIRECTIONAL,
        )

        lstm_out_dim = (
            config.HIDDEN_DIM * 2 if config.BIDIRECTIONAL else config.HIDDEN_DIM
        )
        self.stage0_fc = nn.Linear(lstm_out_dim, config.NUM_CLASSES)

        # ==========================
        # Stage 1: Refinement 1
        # ==========================
        self.stage1_tcn = SingleStageTCN(
            num_layers=config.NUM_LAYERS,
            num_f_maps=config.NUM_F_MAPS,
            num_classes=config.NUM_CLASSES,
            kernel_size=config.KERNEL_SIZE,
            dropout=config.DROPOUT,
        )

        # ==========================
        # Stage 2: Refinement 2
        # ==========================
        self.stage2_tcn = SingleStageTCN(
            num_layers=config.NUM_LAYERS,
            num_f_maps=config.NUM_F_MAPS,
            num_classes=config.NUM_CLASSES,
            kernel_size=config.KERNEL_SIZE,
            dropout=config.DROPOUT,
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: Input features [Batch, Time, InputDim]
            mask: Boolean mask [Batch, Time] (True for valid, False for padding)

        Returns:
            out0: Logits from Stage 0 [Batch, Time, NumClasses]
            out1: Logits from Stage 1 [Batch, Time, NumClasses]
            out2: Logits from Stage 2 [Batch, Time, NumClasses]
        """
        # Prepare mask for TCN stages: [Batch, 1, Time]
        tcn_mask = None
        if mask is not None:
            tcn_mask = mask.unsqueeze(1).float()

        # ----------------------
        # Stage 0: Generation
        # ----------------------
        # LSTM Output: [Batch, Time, Hidden*2]
        lstm_out, _ = self.lstm(x)

        # Linear Projection: [Batch, Time, NumClasses]
        logits0 = self.stage0_fc(lstm_out)

        # Convert to probabilities for next stage: [Batch, Time, NumClasses]
        probs0 = F.softmax(logits0, dim=2)

        # Transpose for TCN: [Batch, NumClasses, Time]
        probs0_t = probs0.transpose(1, 2)

        # Apply mask to probabilities to prevent padding noise propagation
        if tcn_mask is not None:
            probs0_t = probs0_t * tcn_mask

        # ----------------------
        # Stage 1: Refinement
        # ----------------------
        # Input: Probs from Stage 0
        # Output: Logits [Batch, NumClasses, Time]
        logits1_t = self.stage1_tcn(probs0_t, tcn_mask)

        # Convert to probabilities for next stage
        probs1_t = F.softmax(logits1_t, dim=1)

        if tcn_mask is not None:
            probs1_t = probs1_t * tcn_mask

        # ----------------------
        # Stage 2: Refinement
        # ----------------------
        # Input: Probs from Stage 1
        # Output: Logits [Batch, NumClasses, Time]
        logits2_t = self.stage2_tcn(probs1_t, tcn_mask)

        # ----------------------
        # Formatting Outputs
        # ----------------------
        # Transpose TCN outputs back to [Batch, Time, NumClasses]
        logits1 = logits1_t.transpose(1, 2)
        logits2 = logits2_t.transpose(1, 2)

        # Return all stages for Deep Supervision
        return logits0, logits1, logits2
