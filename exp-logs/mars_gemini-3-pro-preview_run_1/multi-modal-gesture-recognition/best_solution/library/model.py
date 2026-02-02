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
    Y_t = X_t * sigmoid(W_x X_t + W_g G_raw + b)
    """

    def __init__(self, input_dim):
        super(GatedFusion, self).__init__()
        # Removed LayerNorm (Cite 00111)

        # Gating weights
        self.fc_x = nn.Linear(input_dim, input_dim)
        self.fc_g = nn.Linear(input_dim, input_dim)

    def forward(self, x, mask):
        """
        x: (B, T, input_dim) - Concatenated stem outputs
        mask: (B, T) - Boolean mask (True for valid frames)
        """
        # Removed LayerNorm (Cite 00111)

        # 2. Compute Global Average Pooling (G_raw)
        # Handle masking to avoid averaging padding
        mask_expanded = mask.unsqueeze(-1).float()  # (B, T, 1)
        sum_pooled = torch.sum(x * mask_expanded, dim=1)  # (B, input_dim)
        lengths = torch.sum(mask_expanded, dim=1)  # (B, 1)
        # Avoid division by zero
        g_raw = sum_pooled / (lengths + 1e-8)  # (B, input_dim)

        # 3. Conditioned Gating
        # Gate = sigmoid(Wx * X + Wg * G + b)
        # Wg * G needs to be expanded to (B, T, input_dim)
        gate = torch.sigmoid(self.fc_x(x) + self.fc_g(g_raw).unsqueeze(1))

        y = x * gate
        return y


class DualInjectedBiGRU(nn.Module):
    """
    2-Layer BiGRU with Input-Level Anchoring and Dual Injection.
    Reverts to using Input Features for Context Injection (Cite 00110).
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super(DualInjectedBiGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1

        # Layer 1
        self.gru1 = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Layer 2 Input Projections
        # Layer 1 output is (B, T, 2*hidden_dim)
        gru_out_dim = hidden_dim * 2

        # Project Local Input (Y) to match GRU output dim
        self.proj_local = nn.Linear(input_dim, gru_out_dim)

        # Project Refined Context (C_refined) to match GRU output dim
        # C_refined comes from Gated Input, not Hidden State (Cite 00110)
        self.proj_context = nn.Linear(input_dim, gru_out_dim)

        # Layer 2
        # Input to Layer 2 is sum of H1, Proj(Y), Proj(C) -> size is gru_out_dim
        self.gru2 = nn.GRU(
            gru_out_dim, hidden_dim, batch_first=True, bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths, mask):
        """
        x: (B, T, input_dim) - Gated Fusion output
        lengths: (B,) - Sequence lengths for packing
        mask: (B, T) - Mask for GAP
        """
        # --- Layer 1 ---
        packed_input = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_h1, _ = self.gru1(packed_input)
        h1, _ = pad_packed_sequence(packed_h1, batch_first=True)  # (B, T, 2*hidden_dim)

        # --- Refined Context Extraction ---
        # GAP of x (The Gated Input) - (Cite 00106, 00110)
        mask_expanded = mask.unsqueeze(-1).float()  # (B, T, 1)
        sum_x = torch.sum(x * mask_expanded, dim=1)
        len_expanded = torch.sum(mask_expanded, dim=1)
        context_refined = sum_x / (len_expanded + 1e-8)  # (B, input_dim)

        # --- Layer 2 Input Construction ---
        # Input_2 = H1 + Proj_local(Y) + Proj_context(C_refined)

        # 1. H1 is already (B, T, 2*hidden_dim)

        # 2. Proj_local(Y)
        local_proj = self.proj_local(x)  # (B, T, 2*hidden_dim)

        # 3. Proj_context(C_refined)
        context_proj = self.proj_context(context_refined).unsqueeze(
            1
        )  # (B, 1, 2*hidden_dim)

        # Sum
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
    Global-Context Input-Injected Network (GCINet)
    Restored from best performing architecture.
    """

    def __init__(self):
        super(GCINet, self).__init__()

        # 1. Decoupled Input Stems
        # Skeleton: 20 joints * 3 coords = 60
        self.skel_stem = InputStem(
            input_dim=Config.NUM_JOINTS * Config.SKELETON_CHANNELS,
            hidden_dim=Config.HIDDEN_DIM,
            kernel_size=Config.KERNEL_SIZE,
            dropout=Config.DROPOUT,
        )

        # Audio: 13 MFCCs
        self.audio_stem = InputStem(
            input_dim=Config.AUDIO_N_MFCC,
            hidden_dim=Config.HIDDEN_DIM,
            kernel_size=Config.KERNEL_SIZE,
            dropout=Config.DROPOUT,
        )

        # Fusion Dimension: HIDDEN_DIM + HIDDEN_DIM
        fusion_dim = Config.HIDDEN_DIM * 2

        # 2. Gated Fusion
        self.fusion = GatedFusion(fusion_dim)

        # 3. Dual-Injected Backbone
        self.backbone = DualInjectedBiGRU(
            input_dim=fusion_dim,
            hidden_dim=Config.HIDDEN_DIM,  # Backbone hidden dim
            dropout=Config.DROPOUT,
        )

        # 4. Classifier Head
        # Backbone output is (B, T, 2*HIDDEN_DIM)
        classifier_input_dim = Config.HIDDEN_DIM * 2

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(
                classifier_input_dim // 2, Config.NUM_CLASSES + 1
            ),  # +1 for Background
        )

    def forward(self, skeleton, audio, lengths, mask):
        """
        skeleton: (B, T, 60)
        audio: (B, T, 13)
        lengths: (B,)
        mask: (B, T)
        """
        # 1. Independent Stems
        s_feat = self.skel_stem(skeleton)  # (B, T, H)
        a_feat = self.audio_stem(audio)  # (B, T, H)

        # 2. Concatenation
        fused = torch.cat([s_feat, a_feat], dim=2)  # (B, T, 2H)

        # 3. Gated Fusion
        gated_fused = self.fusion(fused, mask)  # (B, T, 2H)

        # 4. Backbone
        # Returns (B, T, 2H)
        features = self.backbone(gated_fused, lengths, mask)

        # 5. Classifier
        logits = self.classifier(features)  # (B, T, NumClasses+1)

        return logits
