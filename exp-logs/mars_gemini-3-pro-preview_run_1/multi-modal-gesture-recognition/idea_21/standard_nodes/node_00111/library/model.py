import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library.config import Config


class InputStem(nn.Module):
    """
    Decoupled Input Stem for independent modality processing.
    Structure: Linear -> Permute -> Conv1d(k=7) -> ReLU -> Dropout -> Permute
    """

    def __init__(self, input_dim, hidden_dim, kernel_size=7, dropout=0.3):
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
        # x: (B, T, input_dim)

        # Linear Projection
        x = self.project(x)  # (B, T, hidden_dim)

        # Permute for Conv1d: (B, hidden_dim, T)
        x = x.permute(0, 2, 1)

        # Temporal Convolution
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Permute back: (B, T, hidden_dim)
        x = x.permute(0, 2, 1)
        return x


class GatedFusion(nn.Module):
    """
    Fuses modality streams and applies Global-Conditioned Gating.
    Also computes Refined Global Context from the gated output.
    Cite solution_lesson_node_00106: "re-computing the GAP on the gated sequence".
    """

    def __init__(self, input_dim):
        super(GatedFusion, self).__init__()
        self.norm = nn.LayerNorm(input_dim)

        # Gating weights
        self.fc_x = nn.Linear(input_dim, input_dim)
        self.fc_g = nn.Linear(input_dim, input_dim)

    def forward(self, x, mask):
        """
        x: (B, T, input_dim) - Concatenated stem outputs
        mask: (B, T) - Boolean mask (True for valid frames)
        Returns:
            y: (B, T, input_dim) - Gated features
            g_refined: (B, input_dim) - Global context from gated features
        """
        # 1. Layer Norm
        x = self.norm(x)

        # 2. Compute Global Average Pooling (G_raw)
        mask_expanded = mask.unsqueeze(-1).float()  # (B, T, 1)
        lengths = torch.sum(mask_expanded, dim=1)  # (B, 1)
        denom = lengths + 1e-8

        sum_pooled_raw = torch.sum(x * mask_expanded, dim=1)
        g_raw = sum_pooled_raw / denom  # (B, input_dim)

        # 3. Conditioned Gating
        # Gate = sigmoid(Wx * X + Wg * G + b)
        gate = torch.sigmoid(self.fc_x(x) + self.fc_g(g_raw).unsqueeze(1))

        y = x * gate

        # 4. Compute Refined Global Context (Cite solution_lesson_node_00106)
        sum_pooled_refined = torch.sum(y * mask_expanded, dim=1)
        g_refined = sum_pooled_refined / denom

        return y, g_refined


class ContextInjectedBiGRU(nn.Module):
    """
    2-Layer BiGRU with Input Injection and Refined Global Context Injection.
    Cite solution_lesson_node_00110: "Input-Level Global Gating Outperforms Hidden-State Refinement"
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super(ContextInjectedBiGRU, self).__init__()
        self.hidden_dim = hidden_dim

        # Layer 1
        self.gru1 = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Layer 2 Input Projections
        # Layer 1 output is (B, T, 2*hidden_dim)
        gru_out_dim = hidden_dim * 2

        # Project Local Input (Y) to match GRU output dim
        self.proj_local = nn.Linear(input_dim, gru_out_dim)

        # Project Refined Global Context (G_refined) to match GRU output dim
        self.proj_context = nn.Linear(input_dim, gru_out_dim)

        # Layer 2
        # Input to Layer 2 is sum of H1, Proj(Y), Proj(G_refined)
        self.gru2 = nn.GRU(
            gru_out_dim, hidden_dim, batch_first=True, bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, g_refined, lengths):
        """
        x: (B, T, input_dim) - Gated Fusion output
        g_refined: (B, input_dim) - Refined Global Context
        lengths: (B,) - Sequence lengths for packing
        """
        # --- Layer 1 ---
        packed_input = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_h1, _ = self.gru1(packed_input)
        h1, _ = pad_packed_sequence(packed_h1, batch_first=True)  # (B, T, 2*hidden_dim)

        # --- Layer 2 Input Construction ---
        # Input_2 = H1 + Proj_local(Y) + Proj_context(G_refined)

        # 1. H1 is already (B, T, 2*hidden_dim)

        # 2. Proj_local(Y)
        local_proj = self.proj_local(x)  # (B, T, 2*hidden_dim)

        # 3. Proj_context(G_refined)
        context_proj = self.proj_context(g_refined).unsqueeze(1)  # (B, 1, 2*hidden_dim)

        # Sum (Input Injection + Context Injection)
        input_2 = h1 + local_proj + context_proj
        input_2 = self.dropout(input_2)

        # --- Layer 2 ---
        packed_input_2 = pack_padded_sequence(
            input_2, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_h2, _ = self.gru2(packed_input_2)
        h2, _ = pad_packed_sequence(packed_h2, batch_first=True)  # (B, T, 2*hidden_dim)

        return h2


class GCINet(nn.Module):
    """
    Global-Context Injected Network (GCINet).
    Replaces SR-IIN to align with Lesson solution_lesson_node_00110.
    """

    def __init__(self):
        super(GCINet, self).__init__()

        # 1. Decoupled Input Stems
        self.skel_stem = InputStem(
            input_dim=Config.NUM_JOINTS * Config.SKELETON_CHANNELS,
            hidden_dim=Config.HIDDEN_DIM,
            kernel_size=Config.KERNEL_SIZE,
            dropout=Config.DROPOUT,
        )

        self.audio_stem = InputStem(
            input_dim=Config.AUDIO_N_MFCC,
            hidden_dim=Config.HIDDEN_DIM,
            kernel_size=Config.KERNEL_SIZE,
            dropout=Config.DROPOUT,
        )

        # Fusion Dimension: HIDDEN_DIM + HIDDEN_DIM
        fusion_dim = Config.HIDDEN_DIM * 2

        # 2. Gated Fusion (returns Y and G_refined)
        self.fusion = GatedFusion(fusion_dim)

        # 3. Context-Injected Backbone
        self.backbone = ContextInjectedBiGRU(
            input_dim=fusion_dim,
            hidden_dim=Config.HIDDEN_DIM,
            dropout=Config.DROPOUT,
        )

        # 4. Classifier Head
        classifier_input_dim = Config.HIDDEN_DIM * 2

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(classifier_input_dim // 2, Config.NUM_CLASSES + 1),
        )

    def forward(self, skeleton, audio, lengths, mask):
        # 1. Independent Stems
        s_feat = self.skel_stem(skeleton)
        a_feat = self.audio_stem(audio)

        # 2. Concatenation
        fused = torch.cat([s_feat, a_feat], dim=2)

        # 3. Gated Fusion
        gated_fused, g_refined = self.fusion(fused, mask)

        # 4. Backbone (Injects G_refined)
        features = self.backbone(gated_fused, g_refined, lengths)

        # 5. Classifier
        logits = self.classifier(features)

        return logits
