import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library.config import Config


class SingleScaleStem(nn.Module):
    """
    Single-Scale Stem with high channel capacity.
    Prioritizes feature width over multi-scale complexity (Cite solution_lesson_node_00117).
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(SingleScaleStem, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x):
        # x: (Batch, Time, Channels) -> (Batch, Channels, Time) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return self.dropout(self.activation(x))


class GlobalGatedMechanism(nn.Module):
    """
    Global-Conditioned Gating mechanism.
    Uses Global Average Pooling (GAP) to establish context and gates the input sequence.
    Acts as a Soft Scale Selection mechanism.
    """

    def __init__(self, input_dim):
        super(GlobalGatedMechanism, self).__init__()

        self.fc_x = nn.Linear(input_dim, input_dim)
        self.fc_c = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        # x: (Batch, Time, Dim)
        # lengths: (Batch,)

        B, T, D = x.shape

        # Create mask for valid time steps
        # mask: (B, T)
        mask = torch.arange(T, device=x.device).expand(B, T) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()  # (B, T, 1)

        # Global Average Pooling (masked)
        sum_x = (x * mask_f).sum(dim=1)  # (B, D)
        # Avoid division by zero
        c_raw = sum_x / lengths.unsqueeze(1).float().clamp(min=1.0)

        # Compute Gate
        # Gate = sigmoid(Wx * Xt + Wc * C_raw + b)
        term_x = self.fc_x(x)  # (B, T, D)
        term_c = self.fc_c(c_raw).unsqueeze(1)  # (B, 1, D) broadcast over T

        gate = self.sigmoid(term_x + term_c)

        # Apply Gate
        y = x * gate

        return y, mask_f


class MSC_IIN(nn.Module):
    """
    Multi-Scale Context-Gated Input-Injected Network.
    Integrates multi-scale temporal features, global context gating, and dual-injection backbone.
    """

    def __init__(self):
        super(MSC_IIN, self).__init__()

        # ==========================================
        # 1. Multi-Scale Stems
        # ==========================================
        # Determine output dimension per branch to approximate target embed dim
        num_branches = len(Config.KERNEL_SIZES)
        self.skel_branch_dim = Config.SKELETON_EMBED_DIM // num_branches
        self.audio_branch_dim = Config.AUDIO_EMBED_DIM // num_branches

        self.skel_stem = MultiScaleStem(
            Config.SKELETON_INPUT_CHANNELS, self.skel_branch_dim, Config.KERNEL_SIZES
        )

        self.audio_stem = MultiScaleStem(
            Config.N_MFCC, self.audio_branch_dim, Config.KERNEL_SIZES
        )

        # Calculate actual concatenated dimension
        self.skel_out_dim = self.skel_branch_dim * num_branches
        self.audio_out_dim = self.audio_branch_dim * num_branches
        self.fused_input_dim = self.skel_out_dim + self.audio_out_dim

        # Fusion Projection
        self.fusion_proj = nn.Linear(self.fused_input_dim, Config.FUSED_DIM)
        self.dropout = nn.Dropout(Config.DROPOUT)

        # ==========================================
        # 2. Global Gated Mechanism
        # ==========================================
        self.gate = GlobalGatedMechanism(Config.FUSED_DIM)

        # ==========================================
        # 3. Backbone (Dual-Injected BiGRU)
        # ==========================================
        self.gru_hidden = Config.HIDDEN_DIM
        self.gru_out_dim = Config.HIDDEN_DIM * 2  # Bidirectional

        # Layer 1
        self.gru1 = nn.GRU(
            Config.FUSED_DIM, self.gru_hidden, batch_first=True, bidirectional=True
        )

        # Injection Projections
        # Projects Local Features (Y) and Global Context (G_refined) to match GRU output space
        self.proj_local = nn.Linear(Config.FUSED_DIM, self.gru_out_dim)
        self.proj_global = nn.Linear(Config.FUSED_DIM, self.gru_out_dim)

        # Layer 2
        # Input is summation of H1, Proj(Y), Proj(G)
        self.gru2 = nn.GRU(
            self.gru_out_dim, self.gru_hidden, batch_first=True, bidirectional=True
        )

        # ==========================================
        # 4. Output Head
        # ==========================================
        self.classifier = nn.Sequential(
            nn.Linear(self.gru_out_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, Config.NUM_CLASSES),
        )

    def forward(self, skeleton, audio, lengths):
        # skeleton: (B, T, 60)
        # audio: (B, T, 13)
        # lengths: (B,)

        # ------------------------------------------
        # 1. Feature Extraction & Fusion
        # ------------------------------------------
        skel_feat = self.skel_stem(skeleton)
        audio_feat = self.audio_stem(audio)

        fused = torch.cat([skel_feat, audio_feat], dim=2)
        x = self.dropout(self.fusion_proj(fused))  # (B, T, FUSED_DIM)

        # ------------------------------------------
        # 2. Global Gating
        # ------------------------------------------
        # y: Gated features, mask: Valid frame mask
        y, mask = self.gate(x, lengths)

        # ------------------------------------------
        # 3. Refined Context Anchor
        # ------------------------------------------
        # Compute GAP on the *gated* features
        sum_y = (y * mask).sum(dim=1)
        g_refined = sum_y / lengths.unsqueeze(1).float().clamp(
            min=1.0
        )  # (B, FUSED_DIM)

        # ------------------------------------------
        # 4. Backbone Layer 1
        # ------------------------------------------
        # Pack sequence for RNN
        packed_y = pack_padded_sequence(
            y, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_h1, _ = self.gru1(packed_y)

        # Unpack
        h1, _ = pad_packed_sequence(packed_h1, batch_first=True, total_length=y.size(1))

        # ------------------------------------------
        # 5. Dual Injection
        # ------------------------------------------
        # Input2 = H1 + Proj_local(Y) + Proj_global(G_refined)

        p_local = self.proj_local(y)  # (B, T, Hidden*2)
        p_global = self.proj_global(g_refined).unsqueeze(1)  # (B, 1, Hidden*2)

        input2 = h1 + p_local + p_global

        # ------------------------------------------
        # 6. Backbone Layer 2
        # ------------------------------------------
        packed_input2 = pack_padded_sequence(
            input2, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_h2, _ = self.gru2(packed_input2)

        h2, _ = pad_packed_sequence(packed_h2, batch_first=True, total_length=y.size(1))

        # ------------------------------------------
        # 7. Classification
        # ------------------------------------------
        logits = self.classifier(h2)  # (B, T, NumClasses)

        return logits
