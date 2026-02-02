import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class MultiGranularityStem(nn.Module):
    """
    Processes the input through parallel 1D convolutional branches with different kernel sizes
    to capture features at multiple temporal granularities.
    """

    def __init__(self, input_dim, hidden_dim, kernel_sizes):
        super(MultiGranularityStem, self).__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            # Padding to maintain temporal dimension: (k - 1) // 2
            padding = (k - 1) // 2
            branch = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=k, padding=padding),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.branches.append(branch)

        self.output_dim = hidden_dim * len(kernel_sizes)

    def forward(self, x):
        # x: (B, C_in, T)
        outputs = [branch(x) for branch in self.branches]
        return torch.cat(outputs, dim=1)  # (B, C_out, T)


class GatedBlock(nn.Module):
    """
    Gated Activation Block for MS-TCN.
    Z = tanh(W_f * X) * sigmoid(W_g * X)
    H = W_proj * Z
    Y = X + H
    """

    def __init__(self, channels, kernel_size, dilation, dropout=0.0):
        super(GatedBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv_f = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.conv_g = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.conv_out = nn.Conv1d(channels, channels, 1)  # 1x1 projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T)
        f = torch.tanh(self.conv_f(x))
        g = torch.sigmoid(self.conv_g(x))
        z = f * g
        z = self.dropout(z)
        h = self.conv_out(z)
        return x + h


class RefinementStage(nn.Module):
    """
    A single refinement stage consisting of stacked GatedBlocks with increasing dilation.
    Takes probabilities as input, projects to hidden dim, refines, and outputs logits.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(RefinementStage, self).__init__()
        self.conv_in = nn.Conv1d(input_dim, hidden_dim, 1)

        layers = []
        for i in range(num_layers):
            # Monotonically increasing dilation: 1, 2, 4, ..., 512
            dilation = 2 ** (i % 10)
            layers.append(
                GatedBlock(
                    hidden_dim, kernel_size=3, dilation=dilation, dropout=dropout
                )
            )

        self.layers = nn.ModuleList(layers)
        self.conv_out = nn.Conv1d(hidden_dim, input_dim, 1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, C_in, T) Input probabilities.
            mask: (B, T) Boolean mask where True indicates padding.
        Returns:
            out: (B, C_in, T) Output logits.
        """
        out = self.conv_in(x)

        for layer in self.layers:
            out = layer(out)

        out = self.conv_out(out)

        # Apply mask to zero out padding in the output logits
        if mask is not None:
            # mask is (B, T), True for padding
            # Expand to (B, 1, T) and cast to float (Valid=0, Padding=1)
            mask_expanded = mask.unsqueeze(1).float()
            # We want to keep valid (0->1) and zero padding (1->0)
            # Actually mask is True(1) for padding. So (1 - mask) is 1 for valid.
            out = out * (1.0 - mask_expanded)

        return out


class GMG_CRGN(nn.Module):
    """
    Geometric Multi-Granularity Convolutional-Recurrent Gated Network.
    """

    def __init__(self):
        super(GMG_CRGN, self).__init__()

        # --- Stage 1: Geometric Multi-Granularity Encoder ---

        # Multi-Granularity Stem
        # Processes input (B, InputDim, T) -> (B, StemDim, T)
        self.stem = MultiGranularityStem(
            input_dim=Config.INPUT_DIM,
            hidden_dim=Config.HIDDEN_DIM,  # Each branch outputs 256 channels
            kernel_sizes=Config.STEM_KERNEL_SIZES,
        )

        # LSTM
        # Input size is the concatenation of stem branches
        lstm_input_size = Config.HIDDEN_DIM * len(Config.STEM_KERNEL_SIZES)

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=Config.LSTM_BIDIRECTIONAL,
        )

        lstm_output_size = (
            Config.HIDDEN_DIM * 2 if Config.LSTM_BIDIRECTIONAL else Config.HIDDEN_DIM
        )

        # Stage 1 Heads
        self.fc_cls = nn.Linear(lstm_output_size, Config.NUM_CLASSES)
        self.fc_bnd = nn.Linear(lstm_output_size, 1)

        # --- Refinement Stages ---
        # Input to stages is [P_cls, P_bnd] (Probabilities)
        refine_input_dim = Config.NUM_CLASSES + 1

        self.stages = nn.ModuleList()
        for _ in range(Config.MSTCN_STAGES):
            self.stages.append(
                RefinementStage(
                    input_dim=refine_input_dim,
                    hidden_dim=Config.MSTCN_CHANNELS,
                    num_layers=Config.MSTCN_LAYERS,
                    dropout=Config.MSTCN_DROPOUT,
                )
            )

    def forward(self, x, mask):
        """
        Args:
            x: (B, T, InputDim) Feature tensor.
            mask: (B, T) Boolean mask where True indicates padding.

        Returns:
            outputs: List of tensors [Stage1_Logits, Stage2_Logits, ...].
                     Each tensor has shape (B, T, NumClasses + 1).
        """
        # Permute for Conv1d: (B, InputDim, T)
        x_t = x.transpose(1, 2)

        # --- Stage 1 ---
        stem_out = self.stem(x_t)  # (B, StemDim, T)

        # Permute back for LSTM: (B, T, StemDim)
        lstm_in = stem_out.transpose(1, 2)

        # Mask input to LSTM (optional but good practice)
        if mask is not None:
            lstm_in = lstm_in * (~mask.unsqueeze(-1)).float()

        lstm_out, _ = self.lstm(lstm_in)  # (B, T, Hidden*2)

        # Heads
        logits_cls_s1 = self.fc_cls(lstm_out)  # (B, T, Classes)
        logits_bnd_s1 = self.fc_bnd(lstm_out)  # (B, T, 1)

        # Concatenate logits for output list
        out_s1 = torch.cat([logits_cls_s1, logits_bnd_s1], dim=2)  # (B, T, C+1)
        outputs = [out_s1]

        # Prepare input for Stage 2 (Probabilities)
        probs_cls = F.softmax(logits_cls_s1, dim=2)
        probs_bnd = torch.sigmoid(logits_bnd_s1)
        stage_in = torch.cat([probs_cls, probs_bnd], dim=2)  # (B, T, C+1)

        # Inter-Stage Masking: Zero out padding in probabilities
        if mask is not None:
            stage_in = stage_in * (~mask.unsqueeze(-1)).float()

        # Transpose for Conv1d: (B, C+1, T)
        stage_in = stage_in.transpose(1, 2)

        # --- Refinement Stages ---
        for stage in self.stages:
            # Stage forward: (B, C, T) -> (B, C, T) (Logits)
            stage_out_logits = stage(stage_in, mask)

            # Transpose back to (B, T, C+1) for output list
            out_s_t = stage_out_logits.transpose(1, 2)
            outputs.append(out_s_t)

            # Prepare input for next stage (if needed)
            # We check if we are not at the last stage to avoid unnecessary computation
            if len(outputs) <= Config.MSTCN_STAGES:
                # Logits -> Probs
                l_cls = out_s_t[:, :, : Config.NUM_CLASSES]
                l_bnd = out_s_t[:, :, Config.NUM_CLASSES :]

                p_cls = F.softmax(l_cls, dim=2)
                p_bnd = torch.sigmoid(l_bnd)

                stage_in = torch.cat([p_cls, p_bnd], dim=2)

                # Masking
                if mask is not None:
                    stage_in = stage_in * (~mask.unsqueeze(-1)).float()

                stage_in = stage_in.transpose(1, 2)

        return outputs
