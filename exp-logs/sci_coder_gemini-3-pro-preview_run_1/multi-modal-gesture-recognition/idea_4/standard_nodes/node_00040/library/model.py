import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class InputStem(nn.Module):
    """
    Independent processing stem for a modality (Skeleton or Audio).
    Structure: Linear -> Permute -> Conv1d (k=5) -> ReLU -> Dropout -> Permute
    """

    def __init__(self, input_dim, embed_dim, dropout=0.3):
        super(InputStem, self).__init__()
        self.project = nn.Linear(input_dim, embed_dim)
        # Temporal Convolution: k=7, p=3 maintains length. Cite Lesson 00011.
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=7, padding=3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Time, InputDim)

        # Linear Projection
        x = self.project(x)  # (Batch, Time, EmbedDim)

        # Permute for Conv1d: (Batch, EmbedDim, Time)
        x = x.permute(0, 2, 1)

        # Temporal Conv
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Permute back: (Batch, Time, EmbedDim)
        x = x.permute(0, 2, 1)
        return x


class ContextGating(nn.Module):
    """
    Context Gating Mechanism: Y = X * Sigmoid(WX + b)
    Dynamically re-weights feature channels based on context.
    """

    def __init__(self, input_dim):
        super(ContextGating, self).__init__()
        self.gate_fc = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        gates = self.sigmoid(self.gate_fc(x))
        return x * gates


class ResidualBiGRU(nn.Module):
    """
    Residual Bidirectional GRU Backbone.
    Stack of 2 BiGRU layers.
    Input to Layer 2 = Output of Layer 1 + Projection(Input of Layer 1).
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super(ResidualBiGRU, self).__init__()
        self.hidden_dim = hidden_dim

        # Layer 1
        self.gru1 = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        # Layer 2
        # Input dim is 2 * hidden_dim because Layer 1 is bidirectional
        self.gru2 = nn.GRU(
            hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)

        # Projection to match dimensions for residual connection
        # Input (input_dim) -> Layer 1 Output (2 * hidden_dim)
        self.residual_proj = nn.Linear(input_dim, hidden_dim * 2)

    def forward(self, x, lengths=None):
        # x: (Batch, Time, InputDim)
        # lengths: Tensor of sequence lengths for packing (optional but recommended)

        if lengths is not None:
            # Move lengths to cpu for pack_padded_sequence if on gpu
            lengths_cpu = lengths.cpu()
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )

            # Layer 1
            packed_out1, _ = self.gru1(packed_input)
            out1, _ = nn.utils.rnn.pad_packed_sequence(packed_out1, batch_first=True)
        else:
            out1, _ = self.gru1(x)

        out1 = self.dropout1(out1)

        # Residual Connection preparation
        # Project input x to match out1 dimensions
        res = self.residual_proj(x)

        # Input to Layer 2 is Sum of Layer 1 Output and Projected Input
        in2 = out1 + res

        if lengths is not None:
            packed_in2 = nn.utils.rnn.pack_padded_sequence(
                in2, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out2, _ = self.gru2(packed_in2)
            out2, _ = nn.utils.rnn.pad_packed_sequence(packed_out2, batch_first=True)
        else:
            out2, _ = self.gru2(in2)

        out2 = self.dropout2(out2)

        return out2


class CGR_GRU(nn.Module):
    """
    Context-Gated Residual GRU Network.
    Multi-stream input -> Gated Fusion -> Residual BiGRU -> Classification.
    """

    def __init__(self):
        super(CGR_GRU, self).__init__()

        # Hyperparameters
        self.skel_in = Config.SKELETON_INPUT_DIM
        self.audio_in = Config.AUDIO_INPUT_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_classes = Config.NUM_CLASSES
        self.dropout_p = Config.DROPOUT

        # Embedding dimensions for stems (can be tuned, setting to hidden_dim/2 for balance)
        self.skel_embed_dim = 128
        self.audio_embed_dim = 64

        # 1. Decoupled Input Stems
        self.skel_stem = InputStem(self.skel_in, self.skel_embed_dim, self.dropout_p)
        self.audio_stem = InputStem(self.audio_in, self.audio_embed_dim, self.dropout_p)

        # Fusion Dimension
        self.fusion_dim = self.skel_embed_dim + self.audio_embed_dim

        # 2. Gated Fusion
        self.layer_norm = nn.LayerNorm(self.fusion_dim)
        self.context_gating = ContextGating(self.fusion_dim)

        # 3. Residual BiGRU Backbone
        self.backbone = ResidualBiGRU(self.fusion_dim, self.hidden_dim, self.dropout_p)

        # 4. Classifier Head
        # BiGRU outputs 2 * hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, skeleton, audio, lengths=None):
        """
        Args:
            skeleton: (Batch, Time, 60)
            audio: (Batch, Time, 13)
            lengths: (Batch,) Sequence lengths
        Returns:
            logits: (Batch, Time, NumClasses)
        """
        # Pass through stems
        skel_feat = self.skel_stem(skeleton)  # (B, T, 128)
        audio_feat = self.audio_stem(audio)  # (B, T, 64)

        # Concatenate
        fused = torch.cat([skel_feat, audio_feat], dim=2)  # (B, T, 192)

        # Layer Norm
        fused = self.layer_norm(fused)

        # Context Gating
        fused = self.context_gating(fused)

        # Residual BiGRU
        rnn_out = self.backbone(fused, lengths=lengths)  # (B, T, 512)

        # Classification
        logits = self.classifier(rnn_out)  # (B, T, 21)

        return logits
