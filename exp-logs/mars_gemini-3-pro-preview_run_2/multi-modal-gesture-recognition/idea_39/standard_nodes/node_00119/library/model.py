import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import get_hyperparams


class MultiGranularityStem(nn.Module):
    """
    Multi-Granularity Stem: Extracts motion features at multiple temporal resolutions
    using parallel convolutional branches with different kernel sizes.
    """

    def __init__(self, input_dim, branch_channels, kernels):
        super(MultiGranularityStem, self).__init__()
        self.branches = nn.ModuleList()

        for k in kernels:
            # Calculate padding to maintain temporal dimension T
            # padding = (k - 1) // 2 for centered convolution
            pad = (k - 1) // 2
            branch = nn.Sequential(
                nn.Conv1d(input_dim, branch_channels, kernel_size=k, padding=pad),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU(inplace=True),
            )
            self.branches.append(branch)

    def forward(self, x):
        # x: (B, InputDim, T)
        outputs = [branch(x) for branch in self.branches]
        # Concatenate along channel dimension
        return torch.cat(outputs, dim=1)


class GatedActivationBlock(nn.Module):
    """
    Gated Activation Block for the Refinement Stages (MS-TCN).
    Structure: Dilated Conv -> Split (Filter/Gate) -> Activation -> Dropout -> 1x1 Conv -> Residual
    """

    def __init__(self, in_channels, kernel_size, dilation, dropout):
        super(GatedActivationBlock, self).__init__()

        self.padding = (kernel_size - 1) * dilation // 2

        # Dilated Convolution: Maps C -> 2C (for filter and gate)
        self.conv_dilated = nn.Conv1d(
            in_channels,
            in_channels * 2,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

        self.dropout = nn.Dropout(dropout)

        # 1x1 Projection for residual path
        self.conv_1x1 = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, T)
        out = self.conv_dilated(x)

        # Split into Filter (P) and Gate (Q)
        P, Q = torch.chunk(out, 2, dim=1)

        # Gated Activation: tanh(P) * sigmoid(Q)
        Z = torch.tanh(P) * torch.sigmoid(Q)

        # Dropout
        Z = self.dropout(Z)

        # 1x1 Projection
        H = self.conv_1x1(Z)

        # Residual Connection
        return x + H


class GatedRefinementStage(nn.Module):
    """
    Refinement Stage based on Gated MS-TCN.
    Refines the probabilities from the previous stage.
    """

    def __init__(
        self, input_dim, hidden_dim, num_layers, kernel_size, dropout, num_classes
    ):
        super(GatedRefinementStage, self).__init__()

        # Input Projection: Probabilities -> Hidden Dim
        self.conv_in = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        # Stack of Gated Blocks with increasing dilation
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2**i
            self.layers.append(
                GatedActivationBlock(hidden_dim, kernel_size, dilation, dropout)
            )

        # Output Heads
        self.cls_head = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)
        self.bnd_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, x, mask):
        # x: (B, InputDim, T) - Input probabilities
        # mask: (B, 1, T)

        out = self.conv_in(x)

        for layer in self.layers:
            out = layer(out)
            # Apply mask inside the block sequence?
            # Standard MS-TCN applies mask after the stage, but applying it
            # after each residual block can help stability.
            # Here we follow the prompt: "Inter-Stage Masking".
            # So we just process through layers.

        # Heads
        cls_logits = self.cls_head(out)
        bnd_logits = self.bnd_head(out)

        # Activations
        cls_probs = F.softmax(cls_logits, dim=1)
        bnd_probs = torch.sigmoid(bnd_logits)

        # Concatenate: (B, NumClasses + 1, T)
        final_out = torch.cat([cls_probs, bnd_probs], dim=1)

        return final_out


class MG_CRGN(nn.Module):
    """
    Multi-Granularity Convolutional-Recurrent Gated Network.
    Stage 1: Multi-Granularity Stem + BiLSTM
    Stage 2: Gated Refinement
    Stage 3: Gated Refinement (Cascaded)
    """

    def __init__(self):
        super(MG_CRGN, self).__init__()
        hp = get_hyperparams()

        self.num_classes = hp["num_classes"]
        input_dim = 85  # 36 (Pos) + 36 (Vel) + 13 (Audio)

        # --- Stage 1: Encoder ---
        stem_kernels = hp["stem_kernels"]
        stem_branch_ch = 64
        self.stem = MultiGranularityStem(input_dim, stem_branch_ch, stem_kernels)

        lstm_input_dim = stem_branch_ch * len(stem_kernels)  # 64 * 3 = 192
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hp["lstm_hidden_size"],
            num_layers=hp["lstm_layers"],
            batch_first=True,
            bidirectional=hp["lstm_bidirectional"],
        )

        lstm_out_dim = (
            hp["lstm_hidden_size"] * 2
            if hp["lstm_bidirectional"]
            else hp["lstm_hidden_size"]
        )

        # Stage 1 Heads
        self.stage1_cls = nn.Linear(lstm_out_dim, self.num_classes)
        self.stage1_bnd = nn.Linear(lstm_out_dim, 1)

        # --- Stage 2 & 3: Refinement ---
        # Input to refinement is (NumClasses + 1)
        refine_in_dim = self.num_classes + 1

        self.stage2 = GatedRefinementStage(
            refine_in_dim,
            hp["tcn_channels"],
            hp["tcn_layers"],
            hp["tcn_kernel_size"],
            hp["tcn_dropout"],
            self.num_classes,
        )

        self.stage3 = GatedRefinementStage(
            refine_in_dim,
            hp["tcn_channels"],
            hp["tcn_layers"],
            hp["tcn_kernel_size"],
            hp["tcn_dropout"],
            self.num_classes,
        )

    def forward(self, x, mask):
        """
        Args:
            x: (B, C, T) Input features
            mask: (B, T) Valid frame mask (1.0 valid, 0.0 padded)
        Returns:
            outputs: List of tensors [out1, out2, out3]
                     Each tensor shape: (B, NumClasses+1, T)
        """
        # Mask shape adjustment for broadcasting: (B, 1, T)
        mask_expanded = mask.unsqueeze(1)

        # --- Stage 1 ---
        # Stem: (B, C, T) -> (B, 192, T)
        stem_out = self.stem(x)

        # LSTM requires (B, T, C)
        lstm_in = stem_out.transpose(1, 2)

        # LSTM Forward
        # Pack padded sequence could be used here, but we use simple masking for simplicity consistent with TCN
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(lstm_in)  # (B, T, Hidden*2)

        # Heads
        s1_cls_logits = self.stage1_cls(lstm_out)  # (B, T, NumClasses)
        s1_bnd_logits = self.stage1_bnd(lstm_out)  # (B, T, 1)

        # Activations
        s1_cls_probs = F.softmax(s1_cls_logits, dim=2)
        s1_bnd_probs = torch.sigmoid(s1_bnd_logits)

        # Concat and Transpose back to (B, C, T)
        out1 = torch.cat([s1_cls_probs, s1_bnd_probs], dim=2)
        out1 = out1.transpose(1, 2)

        # Apply Mask
        out1 = out1 * mask_expanded

        # --- Stage 2 ---
        out2 = self.stage2(out1, mask_expanded)
        out2 = out2 * mask_expanded

        # --- Stage 3 ---
        out3 = self.stage3(out2, mask_expanded)
        out3 = out3 * mask_expanded

        return [out1, out2, out3]
