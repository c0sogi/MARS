import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ContextGating(nn.Module):
    """
    Context Gating block: Y = X * Sigmoid(Linear(X))
    Dynamically re-weights features based on their own values.
    """

    def __init__(self, dimension):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)

    def forward(self, x):
        # x: (B, T, D)
        gates = torch.sigmoid(self.fc(x))
        return x * gates


class InputStem(nn.Module):
    """
    Modality-specific processing stem.
    Linear -> Conv1d (k=7) -> ReLU -> Dropout
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, dropout):
        super(InputStem, self).__init__()
        self.project = nn.Linear(input_dim, hidden_dim)
        # Padding = (kernel_size - 1) // 2 to maintain temporal length
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, InputDim)
        x = self.project(x)  # (B, T, HiddenDim)

        # Conv1d expects (B, Channels, Time)
        x = x.transpose(1, 2)  # (B, HiddenDim, T)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)  # (B, T, HiddenDim)
        return x


class CGRNet(nn.Module):
    """
    Context-Gated Residual BiGRU Network.
    Cite {lesson_id: solution_lesson_node_00036}
    Cite {lesson_id: solution_lesson_node_00049}

    Structure:
    1. Independent Skeleton & Audio Stems
    2. Fusion (Concat + LN + ContextGating)
    3. Backbone:
       - Stacked BiGRU with Residual Connection
    4. Classifier
    """

    def __init__(self):
        super(CGRNet, self).__init__()

        # Hyperparameters
        skel_in = Config.INPUT_DIM_SKELETON
        audio_in = Config.INPUT_DIM_AUDIO
        hidden_dim = Config.HIDDEN_DIM
        kernel_size = Config.KERNEL_SIZE
        dropout = Config.DROPOUT
        num_classes = Config.NUM_CLASSES

        # 1. Input Stems
        self.skel_stem = InputStem(skel_in, hidden_dim, kernel_size, dropout)
        self.audio_stem = InputStem(audio_in, hidden_dim, kernel_size, dropout)

        # 2. Fusion
        # Concatenated dimension = hidden_dim * 2
        fusion_dim = hidden_dim * 2
        self.ln_fusion = nn.LayerNorm(fusion_dim)
        self.cg_fusion = ContextGating(fusion_dim)

        # 3. Backbone (Stacked BiGRU with Residual)
        # We use hidden_dim for the GRU hidden state.
        # Since it's bidirectional, output dim is hidden_dim * 2.
        # We set hidden_size = fusion_dim // 2 to maintain consistent dimension (fusion_dim)
        gru_hidden = fusion_dim // 2

        self.gru1 = nn.GRU(
            input_size=fusion_dim,
            hidden_size=gru_hidden,
            bidirectional=True,
            batch_first=True,
        )

        self.gru2 = nn.GRU(
            input_size=fusion_dim,  # Output of gru1 is fusion_dim
            hidden_size=gru_hidden,
            bidirectional=True,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        # 4. Classifier
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, skeleton, audio, mask=None):
        """
        Args:
            skeleton: (B, T, 60)
            audio: (B, T, 13)
            mask: (B, T) - Not used explicitly in GRU but kept for interface consistency
        """
        # 1. Process Stems
        skel_feat = self.skel_stem(skeleton)
        audio_feat = self.audio_stem(audio)

        # 2. Fusion
        fused = torch.cat([skel_feat, audio_feat], dim=-1)
        fused = self.ln_fusion(fused)
        fused = self.cg_fusion(fused)

        # 3. Backbone
        h1, _ = self.gru1(fused)
        h1 = self.dropout(h1)

        h2, _ = self.gru2(h1)
        h2 = self.dropout(h2)

        # Residual Connection (Cite {lesson_id: solution_lesson_node_00036})
        out_backbone = h2 + h1

        # 4. Classifier
        logits = self.classifier(out_backbone)

        return logits
