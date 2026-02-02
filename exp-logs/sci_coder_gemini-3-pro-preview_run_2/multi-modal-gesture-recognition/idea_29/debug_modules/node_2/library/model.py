import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class MaskedSEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block with Mask-Aware Global Average Pooling.
    Explicitly ignores padding during the pooling operation to capture
    accurate global context.
    """

    def __init__(self, channels, reduction=16):
        super(MaskedSEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        # x: (Batch, Channels, Time)
        # mask: (Batch, Time)

        B, C, T = x.shape

        # Expand mask for broadcasting: (B, 1, T)
        mask_expanded = mask.unsqueeze(1).float()

        # Masked Global Average Pooling
        # Sum features over time where mask is 1
        x_sum = torch.sum(x * mask_expanded, dim=2)  # (B, C)

        # Count valid frames
        mask_sum = torch.sum(mask_expanded, dim=2)  # (B, 1)
        mask_sum = torch.clamp(mask_sum, min=1.0)  # Avoid division by zero

        # Average
        y = x_sum / mask_sum  # (B, C)

        # Bottleneck MLP
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)  # (B, C)

        # Recalibrate
        y = y.unsqueeze(2)  # (B, C, 1)
        return x * y


class GatedRefinementBlock(nn.Module):
    """
    Temporal Convolutional Block with Gated Activation and Masked SE.
    Structure:
    1. Dilated Conv (Feature) & Dilated Conv (Gating)
    2. Gated Activation: tanh(f) * sigmoid(g)
    3. Dropout
    4. 1x1 Projection
    5. Masked SE
    6. Residual Connection
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(GatedRefinementBlock, self).__init__()

        # Calculate padding to maintain temporal dimension
        # Padding = (dilation * (kernel_size - 1)) / 2
        padding = (dilation * (kernel_size - 1)) // 2

        self.conv_f = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.conv_g = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )

        self.dropout = nn.Dropout(dropout)
        self.conv_proj = nn.Conv1d(out_channels, out_channels, 1)

        self.se = MaskedSEBlock(out_channels)

    def forward(self, x, mask):
        # x: (B, C, T)

        # Gated Activation
        f = torch.tanh(self.conv_f(x))
        g = torch.sigmoid(self.conv_g(x))
        out = f * g

        out = self.dropout(out)
        out = self.conv_proj(out)

        # Masked Channel Attention
        out = self.se(out, mask)

        # Residual
        return x + out


class BiLSTMEncoder(nn.Module):
    """
    Stage 1 Backbone: Bi-Directional LSTM.
    Projects input features to hidden dimension.
    """

    def __init__(self, input_dim, hidden_dim, num_layers):
        super(BiLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True
        )
        # Project from 2*hidden (BiDir) to hidden
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, lengths):
        # x: (B, T, InputDim)
        # lengths: (B,)

        # Pack sequence for efficient processing
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_out, _ = self.lstm(packed)

        # Unpack
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # Project
        out = self.proj(out)  # (B, T, Hidden)
        return out


class RefinementStage(nn.Module):
    """
    Generic Refinement Stage (used for Stage 2 and 3).
    Stack of GatedRefinementBlocks with increasing dilation.
    """

    def __init__(
        self, input_dim, hidden_dim, num_classes, dilations, kernel_size, dropout
    ):
        super(RefinementStage, self).__init__()

        # Project input probabilities to hidden dimension
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)

        self.layers = nn.ModuleList()
        for d in dilations:
            self.layers.append(
                GatedRefinementBlock(hidden_dim, hidden_dim, kernel_size, d, dropout)
            )

        # Output Heads
        self.cls_head = nn.Conv1d(hidden_dim, num_classes, 1)
        self.bnd_head = nn.Conv1d(hidden_dim, 1, 1)

    def forward(self, x, mask):
        # x: (B, InputDim, T)

        out = self.input_proj(x)

        for layer in self.layers:
            out = layer(out, mask)

        # Heads
        cls_out = self.cls_head(out)  # (B, NumClasses, T)
        bnd_out = self.bnd_head(out)  # (B, 1, T)

        return cls_out, bnd_out


class CASGCN(nn.Module):
    """
    Channel-Attentive Supervised Gated-Cascaded Network.
    Three-stage architecture with Deep Supervision.
    """

    def __init__(self):
        super(CASGCN, self).__init__()

        # Hyperparameters
        input_dim = config.INPUT_DIM
        hidden_dim = config.HIDDEN_DIM
        num_classes = config.NUM_CLASSES
        lstm_layers = config.LSTM_LAYERS
        dilations = config.DILATIONS
        kernel_size = config.KERNEL_SIZE
        dropout = config.DROPOUT

        # --- Stage 1: LSTM Encoder ---
        self.stage1_encoder = BiLSTMEncoder(input_dim, hidden_dim, lstm_layers)
        self.stage1_cls = nn.Linear(hidden_dim, num_classes)
        self.stage1_bnd = nn.Linear(hidden_dim, 1)

        # --- Stage 2: Refinement ---
        # Input: Class Probs + Boundary Prob
        refine_input_dim = num_classes + 1
        self.stage2 = RefinementStage(
            refine_input_dim, hidden_dim, num_classes, dilations, kernel_size, dropout
        )

        # --- Stage 3: Sharpening ---
        self.stage3 = RefinementStage(
            refine_input_dim, hidden_dim, num_classes, dilations, kernel_size, dropout
        )

    def forward(self, features, mask, lengths):
        """
        Args:
            features: (B, T, InputDim)
            mask: (B, T) Boolean/Int mask (1=valid, 0=pad)
            lengths: (B,) Sequence lengths
        Returns:
            Dictionary containing outputs from all stages.
        """

        # --- Stage 1 ---
        s1_feat = self.stage1_encoder(features, lengths)  # (B, T, Hidden)

        s1_cls_logits = self.stage1_cls(s1_feat)  # (B, T, C)
        s1_bnd_logits = self.stage1_bnd(s1_feat)  # (B, T, 1)

        # Prepare input for Stage 2 (Probabilities)
        s1_cls_prob = F.softmax(s1_cls_logits, dim=2)
        s1_bnd_prob = torch.sigmoid(s1_bnd_logits)

        # Concatenate and Transpose for Conv1D: (B, C+1, T)
        s2_in = torch.cat([s1_cls_prob, s1_bnd_prob], dim=2)
        s2_in = s2_in.permute(0, 2, 1)

        # Explicit Masking (Inter-stage)
        mask_expanded = mask.unsqueeze(1).float()  # (B, 1, T)
        s2_in = s2_in * mask_expanded

        # --- Stage 2 ---
        s2_cls_logits, s2_bnd_logits = self.stage2(s2_in, mask)

        # Prepare input for Stage 3
        s2_cls_prob = F.softmax(s2_cls_logits, dim=1)
        s2_bnd_prob = torch.sigmoid(s2_bnd_logits)

        s3_in = torch.cat([s2_cls_prob, s2_bnd_prob], dim=1)
        s3_in = s3_in * mask_expanded

        # --- Stage 3 ---
        s3_cls_logits, s3_bnd_logits = self.stage3(s3_in, mask)

        # Permute Refinement outputs back to (B, T, C) for consistent loss calculation
        outputs = {
            "stage1": (s1_cls_logits, s1_bnd_logits),
            "stage2": (s2_cls_logits.permute(0, 2, 1), s2_bnd_logits.permute(0, 2, 1)),
            "stage3": (s3_cls_logits.permute(0, 2, 1), s3_bnd_logits.permute(0, 2, 1)),
        }

        return outputs
