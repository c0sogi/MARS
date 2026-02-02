import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class TemporalConvStem(nn.Module):
    """
    Temporal Convolutional Stem for noise suppression and local feature extraction.
    Structure: Conv1D -> BatchNorm -> ReLU.
    """

    def __init__(self, input_dim, output_dim, kernel_size):
        super(TemporalConvStem, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,  # Bias not needed with BN
        )
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (Batch, Time, Dim) -> (Batch, Dim, Time)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # Return: (Batch, Time, Dim)
        return x.permute(0, 2, 1)


class GatedBlock(nn.Module):
    """
    Gated Activation Block with 1x1 Projection and Residual Connection.
    Z = tanh(W_f * X) * sigmoid(W_g * X)
    H = W_proj * Z
    Y = X + H
    """

    def __init__(self, hidden_dim, kernel_size, dilation):
        super(GatedBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv_f = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation
        )
        self.conv_g = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation
        )
        self.conv_proj = nn.Conv1d(hidden_dim, hidden_dim, 1)

    def forward(self, x):
        # x: (Batch, Channel, Time)
        f = torch.tanh(self.conv_f(x))
        g = torch.sigmoid(self.conv_g(x))
        z = f * g
        h = self.conv_proj(z)
        return x + h


class GatedRefinementStage(nn.Module):
    """
    Gated MS-TCN Refinement Stage.
    Input: Probabilities (Classes + Boundary)
    Output: Refined Probabilities
    """

    def __init__(self, input_dim, hidden_dim, num_layers=10, kernel_size=3):
        super(GatedRefinementStage, self).__init__()

        # Project input probabilities to hidden dimension
        self.conv_in = nn.Conv1d(input_dim, hidden_dim, 1)

        # Stack of Gated Blocks with increasing dilation
        self.layers = nn.ModuleList(
            [
                GatedBlock(hidden_dim, kernel_size, dilation=2**i)
                for i in range(num_layers)
            ]
        )

        # Output Heads
        self.cls_head = nn.Conv1d(hidden_dim, Config.NUM_CLASSES, 1)
        self.bnd_head = nn.Conv1d(hidden_dim, 1, 1)

    def forward(self, x, mask):
        # x: (Batch, Time, InputDim)
        # mask: (Batch, Time)

        # Permute to (B, C, T)
        x = x.permute(0, 2, 1)

        out = self.conv_in(x)

        for layer in self.layers:
            out = layer(out)

        # Heads
        cls_logits = self.cls_head(out)  # (B, NumClasses, T)
        bnd_logits = self.bnd_head(out)  # (B, 1, T)

        # Permute back to (B, T, C)
        cls_logits = cls_logits.permute(0, 2, 1)
        bnd_logits = bnd_logits.permute(0, 2, 1)

        # Apply Masking and Activation
        mask_expanded = mask.unsqueeze(-1).float()

        cls_probs = F.softmax(cls_logits, dim=2) * mask_expanded
        bnd_probs = torch.sigmoid(bnd_logits) * mask_expanded

        return cls_probs, bnd_probs


class HCRGCN(nn.Module):
    """
    Hybrid Convolutional-Recurrent Gated-Cascaded Network.
    Stage 1: Stem -> BiLSTM -> Heads
    Stage 2: Gated MS-TCN Refinement
    Stage 3: Gated MS-TCN Refinement
    """

    def __init__(self):
        super(HCRGCN, self).__init__()

        # --- Stage 1: Hybrid Encoder ---
        # Temporal Convolutional Stem
        self.stem = TemporalConvStem(
            input_dim=Config.INPUT_DIM,
            output_dim=Config.HIDDEN_DIM,
            kernel_size=Config.KERNEL_SIZE_STEM,
        )

        # Bi-Directional LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=Config.HIDDEN_DIM,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
        )

        # Stage 1 Heads (Input is 2 * Hidden due to BiDirectional)
        self.s1_cls_head = nn.Linear(Config.HIDDEN_DIM * 2, Config.NUM_CLASSES)
        self.s1_bnd_head = nn.Linear(Config.HIDDEN_DIM * 2, 1)

        # --- Refinement Stages ---
        # Input to refinement is concatenation of Class Probs (21) and Boundary Prob (1)
        refine_input_dim = Config.NUM_CLASSES + 1

        self.stage2 = GatedRefinementStage(
            input_dim=refine_input_dim, hidden_dim=Config.HIDDEN_DIM
        )

        self.stage3 = GatedRefinementStage(
            input_dim=refine_input_dim, hidden_dim=Config.HIDDEN_DIM
        )

    def forward(self, x, mask):
        # x: (Batch, Time, 85)
        # mask: (Batch, Time)

        # --- Stage 1 ---
        stem_out = self.stem(x)  # (B, T, Hidden)

        # LSTM
        lstm_out, _ = self.lstm(stem_out)  # (B, T, 2*Hidden)

        s1_cls_logits = self.s1_cls_head(lstm_out)
        s1_bnd_logits = self.s1_bnd_head(lstm_out)

        # Activation & Masking
        mask_expanded = mask.unsqueeze(-1).float()
        s1_cls_probs = F.softmax(s1_cls_logits, dim=2) * mask_expanded
        s1_bnd_probs = torch.sigmoid(s1_bnd_logits) * mask_expanded

        # --- Stage 2 ---
        # Input: Concatenated probabilities
        s2_in = torch.cat([s1_cls_probs, s1_bnd_probs], dim=2)  # (B, T, 22)
        s2_cls_probs, s2_bnd_probs = self.stage2(s2_in, mask)

        # --- Stage 3 ---
        # Cascaded connection: Stage 3 sees Stage 2 output
        s3_in = torch.cat([s2_cls_probs, s2_bnd_probs], dim=2)
        s3_cls_probs, s3_bnd_probs = self.stage3(s3_in, mask)

        # Return all stages for Deep Supervision
        return {
            "stage1": (s1_cls_probs, s1_bnd_probs),
            "stage2": (s2_cls_probs, s2_bnd_probs),
            "stage3": (s3_cls_probs, s3_bnd_probs),
        }
