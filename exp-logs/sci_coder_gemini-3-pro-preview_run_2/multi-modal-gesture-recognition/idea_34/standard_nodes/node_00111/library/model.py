import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import HYPERPARAMS


class GatedConvBlock(nn.Module):
    """
    Gated Convolutional Block for MS-TCN.
    Structure:
    1. Filter Conv (tanh)
    2. Gate Conv (sigmoid)
    3. Element-wise multiplication (Gating)
    4. 1x1 Projection (W_proj)
    5. Residual Connection (Y = X + H)
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(GatedConvBlock, self).__init__()
        # Padding for 'same' convolution with dilation
        padding = (kernel_size - 1) * dilation // 2

        self.filter_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=padding
        )
        self.gate_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=padding
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T)
        f = torch.tanh(self.filter_conv(x))
        g = torch.sigmoid(self.gate_conv(x))
        z = f * g

        # Projection and Dropout
        h = self.conv_1x1(z)
        h = self.dropout(h)

        # Residual connection
        return x + h


class EncoderStage(nn.Module):
    """
    Stage 1: Multi-Task Recurrent Encoder
    Backbone: Bi-Directional LSTM
    Outputs: Initial Class Logits and Boundary Logits
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super(EncoderStage, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Output projections (Bi-LSTM outputs hidden_dim * 2)
        self.fc_cls = nn.Linear(hidden_dim * 2, num_classes)
        self.fc_bnd = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x: (B, T, InputDim)
        # We rely on the mask applied later to handle padding effects on the loss/next stages
        features, _ = self.lstm(x)  # (B, T, H*2)

        cls_logits = self.fc_cls(features)  # (B, T, NumClasses)
        bnd_logits = self.fc_bnd(features)  # (B, T, 1)

        return cls_logits, bnd_logits


class RefinementStage(nn.Module):
    """
    Stage 2 & 3: Residual Gated Refinement
    Backbone: Gated MS-TCN
    Input: Concatenated Probabilities (Class + Boundary)
    Output: Delta Logits (Residual correction)
    """

    def __init__(
        self, input_dim, num_layers, channels, kernel_size, dropout, num_classes
    ):
        super(RefinementStage, self).__init__()

        # Input projection (1x1 Conv)
        self.conv_in = nn.Conv1d(input_dim, channels, 1)

        # Stack of Gated Conv Blocks
        self.layers = nn.ModuleList(
            [
                GatedConvBlock(
                    channels, channels, kernel_size, dilation=2**i, dropout=dropout
                )
                for i in range(num_layers)
            ]
        )

        # Output projections for Deltas
        self.conv_out_cls = nn.Conv1d(channels, num_classes, 1)
        self.conv_out_bnd = nn.Conv1d(channels, 1, 1)

    def forward(self, x):
        # x: (B, T, InputDim) -> Needs transpose for Conv1d -> (B, InputDim, T)
        x = x.transpose(1, 2)

        feature = self.conv_in(x)

        for layer in self.layers:
            feature = layer(feature)

        # Predict Delta Logits
        delta_cls = self.conv_out_cls(feature)  # (B, NumClasses, T)
        delta_bnd = self.conv_out_bnd(feature)  # (B, 1, T)

        # Transpose back to (B, T, C)
        delta_cls = delta_cls.transpose(1, 2)
        delta_bnd = delta_bnd.transpose(1, 2)

        return delta_cls, delta_bnd


class RLSGCN(nn.Module):
    """
    Residual-Logit Supervised Gated-Cascaded Network (RLSG-CN)
    Connects Encoder -> Refinement 1 -> Refinement 2 with Residual Logit Updates.
    """

    def __init__(self):
        super(RLSGCN, self).__init__()
        hp = HYPERPARAMS["model"]

        # --- Stage 1: Encoder ---
        self.encoder = EncoderStage(
            input_dim=hp["input_dim"],
            hidden_dim=hp["lstm_hidden_dim"],
            num_layers=hp["lstm_layers"],
            num_classes=hp["num_classes"],
            dropout=hp["lstm_dropout"],
        )

        # --- Refinement Stages ---
        # Input is P_cls (21) + P_bnd (1) = 22 channels
        refine_input_dim = hp["num_classes"] + 1

        self.stage2 = RefinementStage(
            input_dim=refine_input_dim,
            num_layers=hp["tcn_num_layers"],
            channels=hp["tcn_channels"],
            kernel_size=hp["tcn_kernel_size"],
            dropout=hp["tcn_dropout"],
            num_classes=hp["num_classes"],
        )

        self.stage3 = RefinementStage(
            input_dim=refine_input_dim,
            num_layers=hp["tcn_num_layers"],
            channels=hp["tcn_channels"],
            kernel_size=hp["tcn_kernel_size"],
            dropout=hp["tcn_dropout"],
            num_classes=hp["num_classes"],
        )

    def forward(self, x, mask):
        """
        Args:
            x: (B, T, InputDim) - Input features
            mask: (B, T) - Sequence mask (1 for valid, 0 for pad)
        Returns:
            outputs: Dictionary containing logits/probs for all stages
        """
        outputs = {}
        mask_expanded = mask.unsqueeze(2)  # (B, T, 1)

        # ==========================================
        # Stage 1: Encoder (Bi-LSTM)
        # ==========================================
        l_cls_0, l_bnd_0 = self.encoder(x)

        # Activation
        p_cls_0 = F.softmax(l_cls_0, dim=2)
        p_bnd_0 = torch.sigmoid(l_bnd_0)

        outputs["stage1"] = {
            "cls_logits": l_cls_0,
            "bnd_logits": l_bnd_0,
            "cls_probs": p_cls_0,
            "bnd_probs": p_bnd_0,
        }

        # ==========================================
        # Stage 2: Refinement (Gated MS-TCN)
        # ==========================================
        # Inter-Stage Masking: Zero out padding in probabilities
        stage2_in = torch.cat([p_cls_0, p_bnd_0], dim=2)
        stage2_in = stage2_in * mask_expanded

        # Predict Residuals (Delta Logits)
        d_cls_1, d_bnd_1 = self.stage2(stage2_in)

        # Residual Update: L_1 = L_0 + Delta_1
        l_cls_1 = l_cls_0 + d_cls_1
        l_bnd_1 = l_bnd_0 + d_bnd_1

        # Activation
        p_cls_1 = F.softmax(l_cls_1, dim=2)
        p_bnd_1 = torch.sigmoid(l_bnd_1)

        outputs["stage2"] = {
            "cls_logits": l_cls_1,
            "bnd_logits": l_bnd_1,
            "cls_probs": p_cls_1,
            "bnd_probs": p_bnd_1,
        }

        # ==========================================
        # Stage 3: Sharpening (Gated MS-TCN)
        # ==========================================
        # Inter-Stage Masking
        stage3_in = torch.cat([p_cls_1, p_bnd_1], dim=2)
        stage3_in = stage3_in * mask_expanded

        # Predict Residuals
        d_cls_2, d_bnd_2 = self.stage3(stage3_in)

        # Residual Update: L_2 = L_1 + Delta_2
        l_cls_2 = l_cls_1 + d_cls_2
        l_bnd_2 = l_bnd_1 + d_bnd_2

        # Activation
        p_cls_2 = F.softmax(l_cls_2, dim=2)
        p_bnd_2 = torch.sigmoid(l_bnd_2)

        outputs["stage3"] = {
            "cls_logits": l_cls_2,
            "bnd_logits": l_bnd_2,
            "cls_probs": p_cls_2,
            "bnd_probs": p_bnd_2,
        }

        return outputs
