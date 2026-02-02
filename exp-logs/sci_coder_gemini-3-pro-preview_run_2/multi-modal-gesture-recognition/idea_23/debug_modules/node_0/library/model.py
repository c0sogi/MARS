import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class GatedActivationUnit(nn.Module):
    """
    Gated Activation Unit for TCN.
    Structure: Input -> Dilated Conv1d -> Split(Filter, Gate) -> Tanh(Filter) * Sigmoid(Gate) -> Dropout -> Residual
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(GatedActivationUnit, self).__init__()

        # Calculate padding to maintain sequence length (Same padding)
        # For kernel_size=k, dilation=d, padding = (k-1)*d / 2
        self.padding = (kernel_size - 1) * dilation // 2

        # Convolution produces 2 * out_channels to be split into filter and gate
        self.conv = nn.Conv1d(
            in_channels,
            2 * out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

        self.dropout = nn.Dropout(dropout)

        # Residual alignment if channels differ
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        # x: (Batch, Channels, Time)

        residual = x if self.downsample is None else self.downsample(x)

        out = self.conv(x)

        # Split into filter and gate branches
        filter_out, gate_out = out.chunk(2, dim=1)

        # Gated Activation: tanh * sigmoid
        out = torch.tanh(filter_out) * torch.sigmoid(gate_out)

        out = self.dropout(out)

        # Residual connection
        return out + residual


class SymmetricRefinementStage(nn.Module):
    """
    Refinement Stage using Symmetric Gated TCN.
    Follows a 'Zoom-Out-Zoom-In' dilation schedule to capture global context
    while retaining local boundary fidelity.
    """

    def __init__(
        self, input_dim, num_classes, hidden_dim, kernel_size, dropout, dilations
    ):
        super(SymmetricRefinementStage, self).__init__()

        # Project input probabilities to hidden dimension
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)

        # Stack of Gated Units with symmetric dilations
        layers = []
        for dilation in dilations:
            layers.append(
                GatedActivationUnit(
                    hidden_dim, hidden_dim, kernel_size, dilation, dropout
                )
            )
        self.layers = nn.Sequential(*layers)

        # Output heads: Map hidden dim back to Class logits and Boundary logit
        self.cls_head = nn.Conv1d(hidden_dim, num_classes, 1)
        self.bnd_head = nn.Conv1d(hidden_dim, 1, 1)

    def forward(self, x, mask):
        # x: (Batch, InputDim, Time) - Input probabilities from previous stage
        # mask: (Batch, 1, Time) - Binary mask

        # Project
        out = self.input_proj(x)

        # Apply TCN layers
        out = self.layers(out)

        # Generate Logits
        cls_logits = self.cls_head(out)  # (B, NumClasses, T)
        bnd_logits = self.bnd_head(out)  # (B, 1, T)

        # Apply Mask to zero out padding
        cls_logits = cls_logits * mask
        bnd_logits = bnd_logits * mask

        return cls_logits, bnd_logits


class BiLSTMEncoder(nn.Module):
    """
    Stage 1: Multi-Task Recurrent Encoder.
    Uses Bi-Directional LSTM to process raw skeletal and audio features.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(BiLSTMEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Output dimension is 2 * hidden due to bidirectional
        lstm_out_dim = 2 * hidden_dim

        # Heads
        self.cls_head = nn.Linear(lstm_out_dim, num_classes)
        self.bnd_head = nn.Linear(lstm_out_dim, 1)

    def forward(self, x, lengths):
        # x: (Batch, Time, InputDim)
        # lengths: (Batch,)

        # Pack sequence for efficient LSTM processing
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_out, _ = self.lstm(packed_x)

        # Unpack
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # Generate Logits
        cls_logits = self.cls_head(out)  # (B, T, NumClasses)
        bnd_logits = self.bnd_head(out)  # (B, T, 1)

        return cls_logits, bnd_logits


class SymG_CRCN(nn.Module):
    """
    Symmetric Gated-Cascaded Recurrent-Convolutional Network.
    Stage 1: BiLSTM Encoder
    Stage 2: Symmetric Gated TCN Refinement
    Stage 3: Symmetric Gated TCN Sharpening
    """

    def __init__(self):
        super(SymG_CRCN, self).__init__()

        # Hyperparameters from Config
        self.num_classes = Config.NUM_CLASSES
        input_dim = Config.INPUT_DIM

        # LSTM Params
        lstm_hidden = Config.LSTM_HIDDEN_DIM
        lstm_layers = Config.LSTM_LAYERS

        # TCN Params
        tcn_channels = Config.TCN_CHANNELS
        tcn_kernel = Config.TCN_KERNEL_SIZE
        tcn_dropout = Config.TCN_DROPOUT
        dilations = Config.SYMMETRIC_DILATIONS

        # --- Stage 1 ---
        self.stage1 = BiLSTMEncoder(
            input_dim, lstm_hidden, lstm_layers, self.num_classes
        )

        # --- Stage 2 ---
        # Input: Concatenation of Class Probs (NumClasses) and Boundary Prob (1)
        refinement_input_dim = self.num_classes + 1
        self.stage2 = SymmetricRefinementStage(
            refinement_input_dim,
            self.num_classes,
            tcn_channels,
            tcn_kernel,
            tcn_dropout,
            dilations,
        )

        # --- Stage 3 ---
        self.stage3 = SymmetricRefinementStage(
            refinement_input_dim,
            self.num_classes,
            tcn_channels,
            tcn_kernel,
            tcn_dropout,
            dilations,
        )

    def forward(self, x, mask, lengths):
        """
        Args:
            x: (Batch, Time, InputDim)
            mask: (Batch, Time) - Boolean mask (True for valid frames)
            lengths: (Batch,)
        Returns:
            dict: Contains logits for all stages (stage1_cls, stage1_bnd, etc.)
        """
        # ---------------------------------------------------------------------
        # Stage 1: Recurrent Encoder
        # ---------------------------------------------------------------------
        s1_cls_logits, s1_bnd_logits = self.stage1(x, lengths)

        # Apply Mask (B, T) -> (B, T, 1) for broadcasting
        mask_expanded = mask.unsqueeze(-1).float()
        s1_cls_logits = s1_cls_logits * mask_expanded
        s1_bnd_logits = s1_bnd_logits * mask_expanded

        # ---------------------------------------------------------------------
        # Prepare Input for Stage 2
        # ---------------------------------------------------------------------
        # Convert logits to probabilities
        s1_probs = torch.softmax(s1_cls_logits, dim=2)  # (B, T, C)
        s1_bnd_probs = torch.sigmoid(s1_bnd_logits)  # (B, T, 1)

        # Concatenate and Permute for TCN: (B, T, C+1) -> (B, C+1, T)
        s2_in = torch.cat([s1_probs, s1_bnd_probs], dim=2)
        s2_in = s2_in.permute(0, 2, 1)

        # TCN Mask: (B, 1, T)
        mask_tcn = mask.unsqueeze(1).float()

        # ---------------------------------------------------------------------
        # Stage 2: Refinement
        # ---------------------------------------------------------------------
        s2_cls_logits, s2_bnd_logits = self.stage2(s2_in, mask_tcn)
        # Outputs are (B, C, T) and (B, 1, T)

        # ---------------------------------------------------------------------
        # Prepare Input for Stage 3
        # ---------------------------------------------------------------------
        s2_probs = torch.softmax(s2_cls_logits, dim=1)
        s2_bnd_probs = torch.sigmoid(s2_bnd_logits)
        s3_in = torch.cat([s2_probs, s2_bnd_probs], dim=1)  # (B, C+1, T)

        # ---------------------------------------------------------------------
        # Stage 3: Sharpening
        # ---------------------------------------------------------------------
        s3_cls_logits, s3_bnd_logits = self.stage3(s3_in, mask_tcn)

        # ---------------------------------------------------------------------
        # Formatting Outputs
        # ---------------------------------------------------------------------
        # Permute TCN outputs back to (B, T, C) for loss calculation
        s2_cls_logits = s2_cls_logits.permute(0, 2, 1)
        s2_bnd_logits = s2_bnd_logits.permute(0, 2, 1)
        s3_cls_logits = s3_cls_logits.permute(0, 2, 1)
        s3_bnd_logits = s3_bnd_logits.permute(0, 2, 1)

        return {
            "stage1_cls": s1_cls_logits,
            "stage1_bnd": s1_bnd_logits,
            "stage2_cls": s2_cls_logits,
            "stage2_bnd": s2_bnd_logits,
            "stage3_cls": s3_cls_logits,
            "stage3_bnd": s3_bnd_logits,
        }
