import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from library.config import (
    NUM_JOINTS,
    JOINT_CHANNELS,
    MFCC_N_MFCC,
    HIDDEN_DIM,
    NUM_CLASSES,
    DROPOUT,
    SKELETON_EDGES,
)
from library.graph_layers import AdaptiveGraphConv


class ContextGating(nn.Module):
    """
    Context Gating block: Y = X * Sigmoid(W*X + b)
    Dynamically re-weights features based on their content.
    """

    def __init__(self, dimension):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)

    def forward(self, x):
        # x: (Batch, Time, Features)
        gates = torch.sigmoid(self.fc(x))
        return x * gates


class SkeletonStem(nn.Module):
    """
    Processes skeleton data using Adaptive Graph Convolution followed by temporal convolution.
    Input: (Batch, Time, Joints, Channels)
    Output: (Batch, Time, Hidden_Dim // 2)
    """

    def __init__(self, in_channels, out_dim, graph_hidden=64):
        super(SkeletonStem, self).__init__()

        # Graph Convolution: (V, C) -> (V, graph_hidden)
        self.graph_conv = AdaptiveGraphConv(in_channels, graph_hidden)

        # Flattened dimension after graph conv: V * graph_hidden
        self.flattened_dim = NUM_JOINTS * graph_hidden

        # Temporal Projection: Conv1d
        # We use a kernel size of 5 to capture local temporal context
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.flattened_dim, out_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        # x: (Batch, Time, Joints, Channels)
        B, T, V, C = x.shape

        # 1. Adaptive Graph Conv
        # Expects (B, T, V, C) -> Returns (B, T, V, graph_hidden)
        x = self.graph_conv(x)

        # 2. Flatten Joints
        # (B, T, V, graph_hidden) -> (B, T, V * graph_hidden)
        x = x.view(B, T, -1)

        # 3. Temporal Convolution
        # Conv1d expects (Batch, Channels, Time)
        x = x.permute(0, 2, 1)  # (B, C_in, T)
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1)  # (B, T, C_out)

        return x


class AudioStem(nn.Module):
    """
    Processes Audio MFCCs using Linear projection and 1D Convolution.
    Input: (Batch, Time, MFCC)
    Output: (Batch, Time, Hidden_Dim // 2)
    """

    def __init__(self, in_channels, out_dim):
        super(AudioStem, self).__init__()

        # Initial projection
        self.linear = nn.Linear(in_channels, out_dim)

        # Temporal Convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        # x: (Batch, Time, MFCC)

        # 1. Linear Projection
        x = self.linear(x)  # (B, T, out_dim)

        # 2. Temporal Convolution
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1)  # (B, T, C)

        return x


class ResidualBiGRU(nn.Module):
    """
    Bidirectional GRU with a residual connection.
    """

    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.0):
        super(ResidualBiGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Projection for residual connection if dimensions mismatch
        # Output of BiGRU is hidden_dim * 2
        self.output_dim = hidden_dim * 2

        if input_dim != self.output_dim:
            self.residual_proj = nn.Linear(input_dim, self.output_dim)
        else:
            self.residual_proj = nn.Identity()

        self.ln = nn.LayerNorm(self.output_dim)

    def forward(self, x, lengths):
        # x: (Batch, Time, Input_Dim)
        # lengths: Tensor of sequence lengths

        # Pack sequence
        packed_input = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # GRU Forward
        packed_output, _ = self.gru(packed_input)

        # Unpack
        output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=x.size(1)
        )

        # Residual Connection
        res = self.residual_proj(x)
        output = self.ln(output + res)

        return output


class AGGRN(nn.Module):
    """
    Adaptive Graph-Gated Residual Network.

    Architecture:
    1. Skeleton Stem (Adaptive Graph Conv + Temporal Conv)
    2. Audio Stem (Linear + Temporal Conv)
    3. Fusion (Concat + LayerNorm + Context Gating)
    4. Backbone (Residual BiGRU)
    5. Classifier (Linear)
    """

    def __init__(self):
        super(AGGRN, self).__init__()

        # Dimensions
        self.stem_dim = HIDDEN_DIM // 2

        # 1. Stems
        self.skeleton_stem = SkeletonStem(
            in_channels=JOINT_CHANNELS, out_dim=self.stem_dim, graph_hidden=64
        )

        self.audio_stem = AudioStem(in_channels=MFCC_N_MFCC, out_dim=self.stem_dim)

        # 2. Fusion
        self.fusion_dim = self.stem_dim * 2
        self.fusion_ln = nn.LayerNorm(self.fusion_dim)
        self.context_gating = ContextGating(self.fusion_dim)

        # 3. Backbone
        self.backbone = ResidualBiGRU(
            input_dim=self.fusion_dim,
            hidden_dim=HIDDEN_DIM,
            num_layers=2,
            dropout=DROPOUT,
        )

        # 4. Classifier
        # Input: BiGRU output (HIDDEN_DIM * 2)
        self.classifier = nn.Linear(HIDDEN_DIM * 2, NUM_CLASSES)

    def forward(self, skeleton, audio, lengths):
        """
        Args:
            skeleton: (Batch, Time, Joints, Channels)
            audio: (Batch, Time, MFCC_Features)
            lengths: (Batch,) Sequence lengths
        """
        # 1. Extract Features
        skel_feat = self.skeleton_stem(skeleton)  # (B, T, stem_dim)
        audio_feat = self.audio_stem(audio)  # (B, T, stem_dim)

        # 2. Fusion
        # Concatenate
        fused = torch.cat([skel_feat, audio_feat], dim=2)  # (B, T, fusion_dim)

        # Normalize
        fused = self.fusion_ln(fused)

        # Context Gating
        fused = self.context_gating(fused)

        # 3. Backbone (Residual BiGRU)
        # Note: We pass lengths to handle padding correctly in RNN
        context = self.backbone(fused, lengths)  # (B, T, HIDDEN_DIM*2)

        # 4. Classification
        logits = self.classifier(context)  # (B, T, NUM_CLASSES)

        return logits
