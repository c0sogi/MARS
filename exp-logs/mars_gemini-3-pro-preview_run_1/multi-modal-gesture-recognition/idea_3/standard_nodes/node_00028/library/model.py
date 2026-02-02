import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from library.config import Config


class TemporalInceptionBlock(nn.Module):
    """
    A temporal inception block that projects inputs into parallel 1D convolutional branches
    with varying kernel sizes to capture features at multiple temporal resolutions.
    Outputs are concatenated and aggregated via a Linear -> ReLU -> Dropout block.
    """

    def __init__(self, in_channels, out_channels, kernels, dropout=0.5):
        super(TemporalInceptionBlock, self).__init__()

        # Internal channel dimension for each branch.
        # We give each branch enough capacity (e.g., half of output size).
        inter_channels = out_channels // 2

        self.branches = nn.ModuleList()
        for k in kernels:
            # Calculate padding to maintain temporal dimension: (k - 1) // 2
            pad = (k - 1) // 2
            self.branches.append(
                nn.Conv1d(in_channels, inter_channels, kernel_size=k, padding=pad)
            )

        # Total channels after concatenation
        concat_channels = inter_channels * len(kernels)

        # Aggregation block: Linear -> ReLU -> Dropout
        # Note: We apply this point-wise across time
        self.aggregator = nn.Sequential(
            nn.Linear(concat_channels, out_channels), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: (Batch, Time, Channels)

        # Permute to (Batch, Channels, Time) for Conv1d
        x_in = x.permute(0, 2, 1)

        branch_outputs = []
        for branch in self.branches:
            out = branch(x_in)
            branch_outputs.append(out)

        # Concatenate along channel dimension: (Batch, Sum(Inter), Time)
        x_concat = torch.cat(branch_outputs, dim=1)

        # Permute back to (Batch, Time, Channels) for Linear layer
        x_concat = x_concat.permute(0, 2, 1)

        # Apply aggregation
        out = self.aggregator(x_concat)

        return out


class MSDIGModel(nn.Module):
    """
    Multi-Scale Decoupled Inception-GRU (MS-DIG) Model.

    Architecture:
    1. Decoupled Input Stems (Skeleton & Audio) using Temporal Inception Blocks.
    2. Feature Fusion with Layer Normalization.
    3. Bidirectional GRU Backbone.
    4. Frame-wise Linear Classifier.
    """

    def __init__(self):
        super(MSDIGModel, self).__init__()

        # --- 1. Decoupled Input Stems ---
        # Skeleton Stream
        # Input: (B, T, 60) -> Output: (B, T, 128)
        self.skel_stem = TemporalInceptionBlock(
            in_channels=Config.SKELETON_INPUT_SIZE,
            out_channels=128,
            kernels=Config.INCEPTION_KERNELS,
            dropout=Config.DROPOUT,
        )

        # Audio Stream
        # Input: (B, T, 13) -> Output: (B, T, 64)
        self.audio_stem = TemporalInceptionBlock(
            in_channels=Config.AUDIO_INPUT_SIZE,
            out_channels=64,
            kernels=Config.INCEPTION_KERNELS,
            dropout=Config.DROPOUT,
        )

        # --- 2. Fusion ---
        # Concatenation of stem outputs
        fusion_dim = 128 + 64
        self.fusion_norm = nn.LayerNorm(fusion_dim)

        # --- 3. Recurrent Core ---
        self.gru = nn.GRU(
            input_size=fusion_dim,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_GRU_LAYERS,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
            dropout=Config.DROPOUT if Config.NUM_GRU_LAYERS > 1 else 0,
        )

        # --- 4. Classifier ---
        gru_out_dim = (
            Config.HIDDEN_SIZE * 2 if Config.BIDIRECTIONAL else Config.HIDDEN_SIZE
        )
        self.classifier = nn.Linear(gru_out_dim, Config.NUM_CLASSES)

    def forward(self, skeleton, audio, lengths=None):
        """
        Forward pass of the model.

        Args:
            skeleton (torch.Tensor): Shape (Batch, Time, 60)
            audio (torch.Tensor): Shape (Batch, Time, 13)
            lengths (torch.Tensor, optional): Shape (Batch,). Valid lengths for packing.

        Returns:
            logits (torch.Tensor): Shape (Batch, Time, NumClasses)
        """
        # 1. Feature Extraction (Decoupled Stems)
        skel_features = self.skel_stem(skeleton)  # (B, T, 128)
        audio_features = self.audio_stem(audio)  # (B, T, 64)

        # 2. Fusion
        fused = torch.cat([skel_features, audio_features], dim=2)  # (B, T, 192)
        fused = self.fusion_norm(fused)

        # 3. Recurrent Processing
        if lengths is not None:
            # Pack sequence for efficient RNN processing ignoring padding
            # Ensure lengths are on CPU and int64 for pack_padded_sequence
            lengths_cpu = lengths.cpu().to(torch.int64)
            packed_input = rnn_utils.pack_padded_sequence(
                fused, lengths_cpu, batch_first=True, enforce_sorted=False
            )

            packed_output, _ = self.gru(packed_input)

            # Unpack back to padded sequence
            gru_out, _ = rnn_utils.pad_packed_sequence(
                packed_output, batch_first=True, total_length=fused.size(1)
            )
        else:
            gru_out, _ = self.gru(fused)

        # 4. Classification
        logits = self.classifier(gru_out)  # (B, T, NumClasses)

        return logits
