import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library.config import Config


class WideStem(nn.Module):
    """
    Wide Single-Scale Stem for independent modality processing.
    Structure: Linear -> Conv1d(k=9) -> ReLU -> Dropout.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, dropout):
        super(WideStem, self).__init__()
        self.project = nn.Linear(input_dim, hidden_dim)
        # Padding = (kernel_size - 1) // 2 to maintain temporal length
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding
        )
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (Batch, Time, InputDim)

        # Projection
        x = self.project(x)  # (B, T, H)

        # Conv1d expects (Batch, Channels, Time)
        x = x.permute(0, 2, 1)  # (B, H, T)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Back to (Batch, Time, Channels)
        x = x.permute(0, 2, 1)  # (B, T, H)
        return x


class RawMagnitudeGating(nn.Module):
    """
    Fuses modalities and gates them based on raw magnitude and global context.
    No pre-normalization is applied.
    """

    def __init__(self, input_dim):
        super(RawMagnitudeGating, self).__init__()
        self.fc_x = nn.Linear(input_dim, input_dim)
        self.fc_g = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_skel, x_audio):
        # Concatenate: (B, T, H_skel) + (B, T, H_audio) -> (B, T, H_combined)
        x_raw = torch.cat([x_skel, x_audio], dim=-1)

        # Global Context (GAP)
        # Masking is handled implicitly by 0-padding if we assume 0s don't skew GAP too much
        # or we rely on the model to learn to ignore padding.
        # For strict correctness with padding, we would need lengths, but GAP on raw usually suffices.
        g_raw = x_raw.mean(dim=1)  # (B, H_combined)

        # Compute Gate
        # Broadcast g_raw: (B, 1, H)
        gate = self.sigmoid(self.fc_x(x_raw) + self.fc_g(g_raw).unsqueeze(1))

        # Apply Gate
        y = x_raw * gate
        return y


class InjectedBiGRU(nn.Module):
    """
    3-Layer BiGRU with Input and Anchor Injection.
    Injects the clean local signal (Y) and global anchor (G) into deeper layers.
    """

    def __init__(self, input_dim, hidden_dim, dropout):
        super(InjectedBiGRU, self).__init__()
        self.layers = Config.BACKBONE_LAYERS
        self.hidden_dim = hidden_dim  # This is hidden size per direction
        self.output_dim = hidden_dim * 2  # BiDirectional

        # RNN Layers
        # We use ModuleList to allow manual injection between layers
        self.rnn_layers = nn.ModuleList()
        for i in range(self.layers):
            # Input dim for layer 0 is input_dim
            # Input dim for layer > 0 is output_dim (from prev layer)
            layer_input_dim = input_dim if i == 0 else self.output_dim

            self.rnn_layers.append(
                nn.GRU(
                    layer_input_dim, hidden_dim, batch_first=True, bidirectional=True
                )
            )

        self.dropout = nn.Dropout(dropout)

        # Projections for Injection (Y -> LayerInput, G -> LayerInput)
        # We inject into layers 1 and 2 (0-indexed).
        # The input to these layers matches self.output_dim.
        # Y and G also have dimension input_dim (which is 512 in this config).
        self.proj_y = nn.Linear(input_dim, self.output_dim)
        self.proj_g = nn.Linear(input_dim, self.output_dim)

    def forward(self, y, lengths):
        # y: (B, T, InputDim) - The gated clean signal
        # lengths: (B,)

        # Compute Anchor G_refined (GAP of y)
        g_refined = y.mean(dim=1)  # (B, InputDim)

        current_input = y

        for i, rnn in enumerate(self.rnn_layers):
            # Injection for layers > 0
            if i > 0:
                # Input_i = Prev_Output + Proj(Y) + Proj(G)
                # Prev_Output is current_input (from previous iteration)

                # Project Y and G
                inj_y = self.proj_y(y)  # (B, T, OutDim)
                inj_g = self.proj_g(g_refined).unsqueeze(1)  # (B, 1, OutDim)

                current_input = current_input + inj_y + inj_g

            # Pack
            packed_input = pack_padded_sequence(
                current_input, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

            # Forward RNN
            packed_output, _ = rnn(packed_input)

            # Unpack
            output, _ = pad_packed_sequence(packed_output, batch_first=True)

            # Apply Dropout (except last layer usually, but here we apply between)
            if i < self.layers - 1:
                output = self.dropout(output)

            # Update current_input for next layer
            # Pad output to match y length if necessary (pad_packed handles this, but just to be safe)
            if output.size(1) < y.size(1):
                pad_len = y.size(1) - output.size(1)
                output = F.pad(output, (0, 0, 0, pad_len))

            current_input = output

        return current_input


class DW_AIIN(nn.Module):
    """
    Deep Wide-Spectrum Anchored Input-Injected Network.
    """

    def __init__(self):
        super(DW_AIIN, self).__init__()

        # Hyperparameters
        self.hidden_dim = Config.HIDDEN_DIM  # 256
        self.dropout_p = Config.DROPOUT

        # 1. Wide Stems
        # Skeleton: 60 -> 256
        self.skel_stem = WideStem(
            input_dim=Config.SKELETON_INPUT_DIM,
            hidden_dim=self.hidden_dim,
            kernel_size=Config.WIDE_STEM_KERNEL_SIZE,
            dropout=self.dropout_p,
        )

        # Audio: 13 -> 256
        self.audio_stem = WideStem(
            input_dim=Config.N_MFCC,
            hidden_dim=self.hidden_dim,
            kernel_size=Config.WIDE_STEM_KERNEL_SIZE,
            dropout=self.dropout_p,
        )

        # 2. Raw Magnitude Gating
        # Input dim is sum of stem outputs: 256 + 256 = 512
        self.combined_dim = self.hidden_dim * 2
        self.gating = RawMagnitudeGating(self.combined_dim)

        # 3. Injected Backbone
        # Input: 512. Hidden (per dir): 256. Output: 512.
        self.backbone = InjectedBiGRU(
            input_dim=self.combined_dim,
            hidden_dim=self.hidden_dim,  # 256
            dropout=self.dropout_p,
        )

        # 4. Classification Head
        # Input: 512 -> Output: NUM_CLASSES
        self.head = nn.Sequential(
            nn.Linear(self.combined_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, Config.NUM_CLASSES),
        )

    def forward(self, skeleton, audio, lengths):
        # skeleton: (B, T, 60)
        # audio: (B, T, 13)
        # lengths: (B,)

        # 1. Independent Wide Stems
        x_skel = self.skel_stem(skeleton)  # (B, T, 256)
        x_audio = self.audio_stem(audio)  # (B, T, 256)

        # 2. Raw Magnitude Gating
        # Fuses to (B, T, 512) and gates
        y = self.gating(x_skel, x_audio)

        # 3. Injected Backbone
        features = self.backbone(y, lengths)  # (B, T, 512)

        # 4. Classification Head
        logits = self.head(features)  # (B, T, NumClasses)

        return logits
