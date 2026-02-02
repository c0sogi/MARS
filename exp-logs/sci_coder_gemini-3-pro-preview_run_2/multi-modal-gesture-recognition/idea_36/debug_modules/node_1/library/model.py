import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    INPUT_DIM,
    NUM_CLASSES,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    STEM_KERNEL_SIZE,
    MSTCN_LAYERS,
    MSTCN_CHANNELS,
    MSTCN_KERNEL_SIZE,
    DROPOUT,
)


class TemporalConvStem(nn.Module):
    """
    Initial 1D Convolutional Stem for noise suppression and feature projection.
    Maps raw input features (B, T, D_in) -> (B, T, D_out).
    """

    def __init__(self, input_dim, output_dim, kernel_size=3):
        super(TemporalConvStem, self).__init__()
        self.conv = nn.Conv1d(
            input_dim, output_dim, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, T, C) -> Permute to (B, C, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # Permute back to (B, T, C) for LSTM
        x = x.permute(0, 2, 1)
        return x


class GatedConv1d(nn.Module):
    """
    Dilated Gated Convolutional Block.
    Z = tanh(W_f * X) * sigmoid(W_g * X)
    H = W_proj * Z
    Y = X + H
    """

    def __init__(self, input_channels, hidden_channels, kernel_size, dilation, dropout):
        super(GatedConv1d, self).__init__()

        self.conv_f = nn.Conv1d(
            input_channels,
            hidden_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )
        self.conv_g = nn.Conv1d(
            input_channels,
            hidden_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )

        self.conv_out = nn.Conv1d(hidden_channels, input_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T)
        f = self.conv_f(x)
        g = self.conv_g(x)

        # Gated Activation
        z = torch.tanh(f) * torch.sigmoid(g)

        # Projection and Dropout
        z = self.dropout(self.conv_out(z))

        # Residual Connection
        return x + z


class SingleStageTCN(nn.Module):
    """
    Single Stage MS-TCN containing a stack of GatedConv1d layers.
    """

    def __init__(self, input_dim, num_layers, num_f_maps, kernel_size, dropout):
        super(SingleStageTCN, self).__init__()

        # Initial projection to hidden dimension
        self.conv_in = nn.Conv1d(input_dim, num_f_maps, 1)

        self.layers = nn.ModuleList(
            [
                GatedConv1d(
                    num_f_maps, num_f_maps, kernel_size, dilation=2**i, dropout=dropout
                )
                for i in range(num_layers)
            ]
        )

        # Heads
        self.cls_head = nn.Conv1d(num_f_maps, NUM_CLASSES, 1)
        self.bnd_head = nn.Conv1d(num_f_maps, 1, 1)

    def forward(self, x, mask):
        # x: (B, C_in, T)
        out = self.conv_in(x)

        for layer in self.layers:
            out = layer(out)

        # Apply mask to internal features to prevent leakage (optional but good practice)
        if mask is not None:
            out = out * mask.unsqueeze(1).float()

        cls_logits = self.cls_head(out)  # (B, NumClasses, T)
        bnd_logits = self.bnd_head(out)  # (B, 1, T)

        return cls_logits, bnd_logits


class DCHGNet(nn.Module):
    """
    Dense-Cascaded Hybrid-Gated Network.
    Stage 1: Conv Stem + Bi-LSTM
    Stage 2: Gated MS-TCN (Refinement)
    Stage 3: Gated MS-TCN (Dense Sharpening)
    """

    def __init__(self):
        super(DCHGNet, self).__init__()

        # --- Stage 1: Hybrid Encoder ---
        # Stem projects input to LSTM hidden size for efficient processing
        self.stem = TemporalConvStem(
            INPUT_DIM, LSTM_HIDDEN_SIZE, kernel_size=STEM_KERNEL_SIZE
        )

        self.lstm = nn.LSTM(
            input_size=LSTM_HIDDEN_SIZE,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
        )

        # Stage 1 Heads (Linear layers applied to LSTM output)
        lstm_out_dim = LSTM_HIDDEN_SIZE * 2
        self.stage1_cls = nn.Linear(lstm_out_dim, NUM_CLASSES)
        self.stage1_bnd = nn.Linear(lstm_out_dim, 1)

        # --- Stage 2: Refinement ---
        # Input: Class Probs + Boundary Prob from Stage 1
        stage2_input_dim = NUM_CLASSES + 1
        self.stage2_tcn = SingleStageTCN(
            input_dim=stage2_input_dim,
            num_layers=MSTCN_LAYERS,
            num_f_maps=MSTCN_CHANNELS,
            kernel_size=MSTCN_KERNEL_SIZE,
            dropout=DROPOUT,
        )

        # --- Stage 3: Dense Sharpening ---
        # Input: Concat(Stage 1 Probs, Stage 2 Probs)
        stage3_input_dim = (NUM_CLASSES + 1) * 2
        self.stage3_tcn = SingleStageTCN(
            input_dim=stage3_input_dim,
            num_layers=MSTCN_LAYERS,
            num_f_maps=MSTCN_CHANNELS,
            kernel_size=MSTCN_KERNEL_SIZE,
            dropout=DROPOUT,
        )

    def forward(self, x, mask):
        """
        Args:
            x: (B, T, D) Input features
            mask: (B, T) Boolean mask indicating valid frames
        Returns:
            outputs: Dictionary containing logits for all stages
        """
        # Ensure mask is float for multiplication
        mask_float = mask.float().unsqueeze(-1)  # (B, T, 1)

        # --- Stage 1 ---
        # 1. Stem
        x_stem = self.stem(x)  # (B, T, Hidden)

        # 2. LSTM
        # Pack padded sequence could be used here, but we use explicit masking
        lstm_out, _ = self.lstm(x_stem)  # (B, T, Hidden*2)

        # 3. Heads
        s1_cls_logits = self.stage1_cls(lstm_out)  # (B, T, C)
        s1_bnd_logits = self.stage1_bnd(lstm_out)  # (B, T, 1)

        # 4. Probabilities & Masking
        s1_cls_probs = F.softmax(s1_cls_logits, dim=2)
        s1_bnd_probs = torch.sigmoid(s1_bnd_logits)

        # Zero out padding
        s1_cls_probs = s1_cls_probs * mask_float
        s1_bnd_probs = s1_bnd_probs * mask_float

        # Prepare input for Stage 2 (Concatenate Class + Boundary)
        # TCN expects (B, Channels, Time)
        s1_input = torch.cat([s1_cls_probs, s1_bnd_probs], dim=2)  # (B, T, C+1)
        s1_input = s1_input.permute(0, 2, 1)  # (B, C+1, T)

        # --- Stage 2 ---
        s2_cls_logits, s2_bnd_logits = self.stage2_tcn(s1_input, mask)

        # Permute back to (B, T, C) for consistency and concat
        s2_cls_logits = s2_cls_logits.permute(0, 2, 1)
        s2_bnd_logits = s2_bnd_logits.permute(0, 2, 1)

        # Probabilities & Masking
        s2_cls_probs = F.softmax(s2_cls_logits, dim=2)
        s2_bnd_probs = torch.sigmoid(s2_bnd_logits)

        s2_cls_probs = s2_cls_probs * mask_float
        s2_bnd_probs = s2_bnd_probs * mask_float

        # Prepare input for Stage 3 (Dense Connection)
        # Concat Stage 1 Probs and Stage 2 Probs
        s2_input_feats = torch.cat([s2_cls_probs, s2_bnd_probs], dim=2)  # (B, T, C+1)
        s2_input_feats = s2_input_feats.permute(0, 2, 1)  # (B, C+1, T)

        # Dense Concatenation: (B, 2*(C+1), T)
        s3_input = torch.cat([s1_input, s2_input_feats], dim=1)

        # --- Stage 3 ---
        s3_cls_logits, s3_bnd_logits = self.stage3_tcn(s3_input, mask)

        # Permute back
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
