import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class TemporalConvStem(nn.Module):
    """
    Input Stem: 3 Temporal Convolutional Layers to extract local primitives
    and suppress sensor noise before the recurrent modeling stage.
    """

    def __init__(self, input_dim, hidden_dim):
        super(TemporalConvStem, self).__init__()
        layers = []
        in_channels = input_dim
        for _ in range(Config.CONV_STEM_LAYERS):
            layers.append(
                nn.Conv1d(
                    in_channels,
                    hidden_dim,
                    kernel_size=Config.CONV_STEM_KERNEL,
                    stride=Config.CONV_STEM_STRIDE,
                    padding=Config.CONV_STEM_PADDING,
                )
            )
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_channels = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, T)
        return self.net(x)


class BiLSTMEncoder(nn.Module):
    """
    Backbone: Bi-Directional LSTM for continuous temporal dynamics.
    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(BiLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=Config.LSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=Config.LSTM_BIDIRECTIONAL,
        )

        encoder_output_dim = hidden_dim * 2 if Config.LSTM_BIDIRECTIONAL else hidden_dim
        self.cls_head = nn.Linear(encoder_output_dim, num_classes)
        self.bnd_head = nn.Linear(encoder_output_dim, 1)

    def forward(self, x):
        # x: (B, C, T) -> (B, T, C) for LSTM
        x = x.transpose(1, 2)

        # LSTM Output: (B, T, Hidden*2)
        out, _ = self.lstm(x)

        cls_logits = self.cls_head(out)  # (B, T, NumClasses)
        bnd_logits = self.bnd_head(out)  # (B, T, 1)

        # Transpose back to (B, C, T)
        return cls_logits.transpose(1, 2), bnd_logits.transpose(1, 2)


class GatedRefinementBlock(nn.Module):
    """
    Gated Activation Block with 1x1 Output Projection.
    Z = tanh(W_f * X) * sigmoid(W_g * X)
    Y = X + W_proj * Z
    """

    def __init__(self, channels, kernel_size, dilation, dropout):
        super(GatedRefinementBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv_f = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.conv_g = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.conv_out = nn.Conv1d(channels, channels, 1)  # 1x1 Projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        f = torch.tanh(self.conv_f(x))
        g = torch.sigmoid(self.conv_g(x))
        z = f * g
        z = self.conv_out(z)
        z = self.dropout(z)
        return x + z


class SingleStageTCN(nn.Module):
    """
    Gated MS-TCN Stage.
    """

    def __init__(
        self, input_dim, hidden_dim, num_classes, num_layers, kernel_size, dropout
    ):
        super(SingleStageTCN, self).__init__()
        self.conv_in = nn.Conv1d(input_dim, hidden_dim, 1)

        layers = []
        for i in range(num_layers):
            # Monotonically increasing dilation: 2^0, 2^1, ...
            dilation = 2**i
            layers.append(
                GatedRefinementBlock(hidden_dim, kernel_size, dilation, dropout)
            )
        self.layers = nn.Sequential(*layers)

        self.conv_cls = nn.Conv1d(hidden_dim, num_classes, 1)
        self.conv_bnd = nn.Conv1d(hidden_dim, 1, 1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.layers(out)
        cls_logits = self.conv_cls(out)
        bnd_logits = self.conv_bnd(out)
        return cls_logits, bnd_logits


class CRGCN(nn.Module):
    """
    Convolutional-Recurrent Gated-Cascaded Network (CRG-CN).
    Stage 1: Conv Stem + BiLSTM Encoder
    Stage 2: Gated TCN Refinement
    Stage 3: Gated TCN Sharpening
    """

    def __init__(self):
        super(CRGCN, self).__init__()

        # Stage 1: Hybrid Encoder
        # Stem maps InputDim (49) -> 64 (Local Primitives)
        self.stem_dim = 64
        self.stem = TemporalConvStem(Config.INPUT_DIM, self.stem_dim)

        # LSTM maps 64 -> Hidden (256) -> Heads
        self.encoder = BiLSTMEncoder(
            self.stem_dim, Config.LSTM_HIDDEN_DIM, Config.NUM_CLASSES
        )

        # Stages 2 & 3: Refinement
        # Input is Concatenated Probs: NumClasses + 1 (Boundary)
        refine_input_dim = Config.NUM_CLASSES + 1

        self.stage2 = SingleStageTCN(
            refine_input_dim,
            Config.TCN_CHANNELS,
            Config.NUM_CLASSES,
            Config.TCN_NUM_LAYERS,
            Config.TCN_KERNEL_SIZE,
            Config.TCN_DROPOUT,
        )

        self.stage3 = SingleStageTCN(
            refine_input_dim,
            Config.TCN_CHANNELS,
            Config.NUM_CLASSES,
            Config.TCN_NUM_LAYERS,
            Config.TCN_KERNEL_SIZE,
            Config.TCN_DROPOUT,
        )

    def forward(self, x, mask):
        # x: (B, T, InputDim)
        # mask: (B, T)

        # Transpose input for Conv1d: (B, InputDim, T)
        x = x.transpose(1, 2)

        # Expand mask for broadcasting: (B, 1, T)
        mask_expanded = mask.unsqueeze(1).float()

        # --- Stage 1 ---
        feat = self.stem(x)
        s1_cls, s1_bnd = self.encoder(feat)

        # Apply mask to logits
        s1_cls = s1_cls * mask_expanded
        s1_bnd = s1_bnd * mask_expanded

        # Prepare input for Stage 2
        # Use probabilities for refinement input
        s1_cls_prob = F.softmax(s1_cls, dim=1)
        s1_bnd_prob = torch.sigmoid(s1_bnd)
        s2_in = torch.cat([s1_cls_prob, s1_bnd_prob], dim=1)
        s2_in = s2_in * mask_expanded

        # --- Stage 2 ---
        s2_cls, s2_bnd = self.stage2(s2_in)
        s2_cls = s2_cls * mask_expanded
        s2_bnd = s2_bnd * mask_expanded

        # Prepare input for Stage 3
        s2_cls_prob = F.softmax(s2_cls, dim=1)
        s2_bnd_prob = torch.sigmoid(s2_bnd)
        s3_in = torch.cat([s2_cls_prob, s2_bnd_prob], dim=1)
        s3_in = s3_in * mask_expanded

        # --- Stage 3 ---
        s3_cls, s3_bnd = self.stage3(s3_in)
        s3_cls = s3_cls * mask_expanded
        s3_bnd = s3_bnd * mask_expanded

        # Return logits for all stages (for deep supervision loss)
        return {
            "stage1": (s1_cls, s1_bnd),
            "stage2": (s2_cls, s2_bnd),
            "stage3": (s3_cls, s3_bnd),
        }
