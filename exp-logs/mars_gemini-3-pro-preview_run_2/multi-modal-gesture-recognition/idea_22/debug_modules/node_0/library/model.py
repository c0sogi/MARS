import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DualScaleGatedBlock(nn.Module):
    """
    Dual-Scale Gated Block processing input through parallel Global and Local branches.
    Global Branch: Dilated Convolution (d = 2^i) -> Gated Activation.
    Local Branch: Fixed Convolution (d = 1) -> Gated Activation.
    Fusion: Concatenation -> 1x1 Conv.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(DualScaleGatedBlock, self).__init__()

        # Padding to maintain temporal dimension
        # padding = (kernel_size - 1) * dilation // 2
        pad_global = (kernel_size - 1) * dilation // 2
        pad_local = (kernel_size - 1) * 1 // 2

        # Global Branch (Dilated)
        # Output channels * 2 for Gated Activation (Tanh + Sigmoid)
        self.global_conv = nn.Conv1d(
            in_channels,
            out_channels * 2,
            kernel_size,
            padding=pad_global,
            dilation=dilation,
        )

        # Local Branch (Fixed Dilation = 1)
        self.local_conv = nn.Conv1d(
            in_channels, out_channels * 2, kernel_size, padding=pad_local, dilation=1
        )

        # Fusion Layer
        # Concatenates outputs of both branches (out_channels + out_channels) -> out_channels
        self.fusion_conv = nn.Conv1d(out_channels * 2, out_channels, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T)

        # Global Branch
        g_out = self.global_conv(x)
        g_tanh, g_sigmoid = torch.split(g_out, g_out.shape[1] // 2, dim=1)
        g_act = torch.tanh(g_tanh) * torch.sigmoid(g_sigmoid)

        # Local Branch
        l_out = self.local_conv(x)
        l_tanh, l_sigmoid = torch.split(l_out, l_out.shape[1] // 2, dim=1)
        l_act = torch.tanh(l_tanh) * torch.sigmoid(l_sigmoid)

        # Fusion
        # Concatenate along channel dimension
        fused = torch.cat([g_act, l_act], dim=1)
        out = self.fusion_conv(fused)
        out = self.dropout(out)

        # Residual Connection
        return x + out


class BiLSTMEncoder(nn.Module):
    """
    Stage 1: Multi-Task Recurrent Encoder.
    Backbone: Bi-Directional LSTM.
    Heads: Class Probabilities and Boundary Probability.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(BiLSTMEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Projection heads
        # Input to heads is hidden_dim * 2 (bidirectional)
        self.cls_head = nn.Linear(hidden_dim * 2, num_classes)
        self.bnd_head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x: (B, T, Input_Dim)

        # LSTM output: (B, T, Hidden*2)
        # We don't pack sequences here for simplicity, assuming masking handles padding downstream
        lstm_out, _ = self.lstm(x)

        # Heads
        cls_logits = self.cls_head(lstm_out)  # (B, T, Num_Classes)
        bnd_logits = self.bnd_head(lstm_out)  # (B, T, 1)

        # Apply activations
        cls_probs = F.softmax(cls_logits, dim=2)
        bnd_probs = torch.sigmoid(bnd_logits)

        return cls_probs, bnd_probs


class RefinementStage(nn.Module):
    """
    Stage 2 & 3: Dual-Scale Gated Refinement.
    Refines probabilities using a stack of DualScaleGatedBlocks.
    """

    def __init__(
        self, input_dim, hidden_dim, num_layers, kernel_size, dropout, num_classes
    ):
        super(RefinementStage, self).__init__()

        # Project input probabilities to hidden dimension
        self.input_conv = nn.Conv1d(input_dim, hidden_dim, 1)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Exponential dilation: 1, 2, 4, 8, ...
            dilation = 2**i
            self.layers.append(
                DualScaleGatedBlock(
                    hidden_dim, hidden_dim, kernel_size, dilation, dropout
                )
            )

        # Projection heads (implemented as 1x1 Convs since working on (B, C, T))
        self.cls_head = nn.Conv1d(hidden_dim, num_classes, 1)
        self.bnd_head = nn.Conv1d(hidden_dim, 1, 1)

    def forward(self, x):
        # x: (B, T, Input_Dim) -> Needs transpose for Conv1d -> (B, Input_Dim, T)
        x = x.transpose(1, 2)

        out = self.input_conv(x)

        for layer in self.layers:
            out = layer(out)

        cls_logits = self.cls_head(out)  # (B, Num_Classes, T)
        bnd_logits = self.bnd_head(out)  # (B, 1, T)

        # Transpose back to (B, T, C)
        cls_logits = cls_logits.transpose(1, 2)
        bnd_logits = bnd_logits.transpose(1, 2)

        cls_probs = F.softmax(cls_logits, dim=2)
        bnd_probs = torch.sigmoid(bnd_logits)

        return cls_probs, bnd_probs


class DSG_CRCN(nn.Module):
    """
    Dual-Scale Supervised Gated-Cascaded Recurrent-Convolutional Network.
    Stage 1: Bi-LSTM Encoder
    Stage 2: Refinement 1
    Stage 3: Refinement 2 (Sharpening)
    """

    def __init__(self):
        super(DSG_CRCN, self).__init__()

        # Config params
        input_dim = Config.INPUT_DIM
        num_classes = Config.NUM_CLASSES
        lstm_hidden = Config.LSTM_HIDDEN_SIZE
        lstm_layers = Config.LSTM_LAYERS

        refine_channels = Config.REFINEMENT_CHANNELS
        refine_layers = Config.REFINEMENT_LAYERS
        kernel_size = Config.KERNEL_SIZE_GLOBAL
        dropout = Config.DROPOUT

        # Stage 1
        self.stage1 = BiLSTMEncoder(input_dim, lstm_hidden, lstm_layers, num_classes)

        # Stage 2
        # Input is concatenation of Class Probs + Boundary Prob = Num_Classes + 1
        refine_input_dim = num_classes + 1
        self.stage2 = RefinementStage(
            refine_input_dim,
            refine_channels,
            refine_layers,
            kernel_size,
            dropout,
            num_classes,
        )

        # Stage 3
        self.stage3 = RefinementStage(
            refine_input_dim,
            refine_channels,
            refine_layers,
            kernel_size,
            dropout,
            num_classes,
        )

    def forward(self, x, mask):
        """
        Args:
            x: (B, T, Input_Dim)
            mask: (B, T) Boolean mask where True indicates valid frames.
        Returns:
            outputs: Dictionary containing outputs from all stages.
        """
        outputs = {}

        # Expand mask for multiplication: (B, T, 1)
        mask_expanded = mask.unsqueeze(-1).float()

        # --- Stage 1 ---
        s1_cls, s1_bnd = self.stage1(x)

        # Apply mask
        s1_cls = s1_cls * mask_expanded
        s1_bnd = s1_bnd * mask_expanded

        outputs["stage1_cls"] = s1_cls
        outputs["stage1_bnd"] = s1_bnd

        # --- Stage 2 ---
        # Concatenate outputs from Stage 1
        s2_in = torch.cat([s1_cls, s1_bnd], dim=2)  # (B, T, C+1)

        s2_cls, s2_bnd = self.stage2(s2_in)

        # Apply mask
        s2_cls = s2_cls * mask_expanded
        s2_bnd = s2_bnd * mask_expanded

        outputs["stage2_cls"] = s2_cls
        outputs["stage2_bnd"] = s2_bnd

        # --- Stage 3 ---
        s3_in = torch.cat([s2_cls, s2_bnd], dim=2)

        s3_cls, s3_bnd = self.stage3(s3_in)

        # Apply mask
        s3_cls = s3_cls * mask_expanded
        s3_bnd = s3_bnd * mask_expanded

        outputs["stage3_cls"] = s3_cls
        outputs["stage3_bnd"] = s3_bnd

        # Final output for inference is Stage 3 Class Probs
        outputs["final_cls"] = s3_cls

        return outputs
