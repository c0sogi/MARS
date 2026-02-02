import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library.config import Config


class WideStem(nn.Module):
    """
    Processes a single modality via a wide Linear -> Conv1d block.
    Preserves temporal resolution while projecting to a high-dimensional space.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, dropout):
        super().__init__()
        # Linear projection to expand/compress dimension first
        self.linear = nn.Linear(input_dim, hidden_dim)

        # Conv1d for local temporal context
        # Padding ensures output length equals input length
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, InputDim)
        x = self.linear(x)  # (B, T, HiddenDim)

        # Permute for Conv1d: (B, HiddenDim, T)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Permute back: (B, T, HiddenDim)
        x = x.permute(0, 2, 1)
        return x


class MagnitudePreservingFusion(nn.Module):
    """
    Fuses modalities using a gated mechanism without Normalization layers.
    Uses Masked Global Average Pooling to generate the gating context.
    """

    def __init__(self, input_dim):
        super().__init__()
        # input_dim is the size of concatenated features (e.g., 512)

        # Projections for the gating mechanism
        # Gate = sigmoid(Wx * X + Wg * G + b)
        self.gate_x_proj = nn.Linear(input_dim, input_dim)
        self.gate_g_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x_skel, x_audio, mask):
        # x_skel, x_audio: (B, T, C)
        # mask: (B, T) float tensor (1.0 for valid, 0.0 for pad)

        # 1. Concatenate
        x_raw = torch.cat([x_skel, x_audio], dim=-1)  # (B, T, 2*C)

        # 2. Masked Global Average Pooling (G_raw)
        # Expand mask for broadcasting: (B, T, 1)
        mask_expanded = mask.unsqueeze(-1)

        # Sum valid frames
        sum_features = torch.sum(x_raw * mask_expanded, dim=1)  # (B, 2*C)
        # Count valid frames (clamp to avoid div by zero)
        sum_mask = torch.clamp(torch.sum(mask_expanded, dim=1), min=1.0)  # (B, 1)

        g_raw = sum_features / sum_mask  # (B, 2*C)

        # 3. Conditioned Gating
        # Gate_t = sigmoid(Wx(X_raw_t) + Wg(G_raw))
        gate_in_x = self.gate_x_proj(x_raw)  # (B, T, 2*C)
        gate_in_g = self.gate_g_proj(g_raw).unsqueeze(1)  # (B, 1, 2*C)

        gate = torch.sigmoid(gate_in_x + gate_in_g)

        # 4. Apply Gate (Element-wise multiplication)
        y = x_raw * gate

        # Explicitly apply mask again to ensure padded regions are exactly zero
        y = y * mask_expanded

        return y


class AuxiliaryHead(nn.Module):
    """
    Predicts the set of gestures present in the sequence.
    Provides semantic supervision to the global context vector.
    """

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x, mask):
        # x: (B, T, C) - this is the gated output Y
        # mask: (B, T)

        # Compute Refined Global Context (G_refined) via Masked GAP
        mask_expanded = mask.unsqueeze(-1)
        sum_features = torch.sum(x * mask_expanded, dim=1)
        sum_mask = torch.clamp(torch.sum(mask_expanded, dim=1), min=1.0)
        g_refined = sum_features / sum_mask  # (B, C)

        # Predict set (multi-hot)
        out = self.net(g_refined)

        return out, g_refined


class InputInjectedBiGRU(nn.Module):
    """
    2-Layer BiGRU with Dual Injection in the second layer.
    Layer 2 Input = Layer 1 Output + Proj(Raw_Input) + Proj(Global_Context)
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Bidirectional GRU output dim is 2 * hidden_dim
        gru_out_dim = 2 * hidden_dim

        # Layer 1
        self.gru1 = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Projections for Injection
        # We project Y and G (size input_dim) to match GRU output size (size 2*hidden_dim)
        self.proj_y = nn.Linear(input_dim, gru_out_dim)
        self.proj_g = nn.Linear(input_dim, gru_out_dim)

        # Layer 2
        # Input size is gru_out_dim (since we sum H1 and projections)
        self.gru2 = nn.GRU(
            gru_out_dim, hidden_dim, batch_first=True, bidirectional=True
        )

    def forward(self, y, g_refined, lengths):
        # y: (B, T, input_dim)
        # g_refined: (B, input_dim)
        # lengths: (B,) cpu tensor

        # Pack sequence for Layer 1
        packed_y = pack_padded_sequence(
            y, lengths, batch_first=True, enforce_sorted=False
        )

        # Layer 1 Forward
        packed_h1, _ = self.gru1(packed_y)
        h1, _ = pad_packed_sequence(packed_h1, batch_first=True)  # (B, T, 2*hidden)

        # Injection
        # Input_2 = H1 + Proj(Y) + Proj(G)
        proj_y = self.proj_y(y)  # (B, T, 2*hidden)
        proj_g = self.proj_g(g_refined).unsqueeze(1)  # (B, 1, 2*hidden)

        input_2 = h1 + proj_y + proj_g

        # Pack for Layer 2
        packed_input_2 = pack_padded_sequence(
            input_2, lengths, batch_first=True, enforce_sorted=False
        )
        packed_h2, _ = self.gru2(packed_input_2)
        h2, _ = pad_packed_sequence(packed_h2, batch_first=True)

        return h2


class SAMPNet(nn.Module):
    """
    Semantic-Anchored Magnitude-Preserving Network.
    """

    def __init__(self):
        super().__init__()

        # Hyperparameters from Config
        self.skel_dim = Config.SKELETON_INPUT_DIM
        self.audio_dim = Config.AUDIO_INPUT_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        self.kernel_size = Config.KERNEL_SIZE
        self.dropout = Config.DROPOUT
        self.num_classes = Config.NUM_CLASSES

        # 1. Wide Independent Stems
        self.skel_stem = WideStem(
            self.skel_dim, self.hidden_dim, self.kernel_size, self.dropout
        )
        self.audio_stem = WideStem(
            self.audio_dim, self.hidden_dim, self.kernel_size, self.dropout
        )

        # Fused Dimension (Concat of two stems)
        self.fused_dim = self.hidden_dim * 2

        # 2. Magnitude-Preserving Fusion
        self.fusion = MagnitudePreservingFusion(self.fused_dim)

        # 3. Semantic Auxiliary Head
        self.aux_head = AuxiliaryHead(self.fused_dim, self.num_classes)

        # 4. Input-Injected Backbone
        self.backbone = InputInjectedBiGRU(self.fused_dim, self.hidden_dim)

        # 5. Non-Linear Output Head
        # Backbone output is 2 * hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, skel, audio, lengths):
        """
        Args:
            skel: (B, T, 60)
            audio: (B, T, 13)
            lengths: (B,) LongTensor containing sequence lengths
        Returns:
            logits: (B, T, NumClasses)
            aux_preds: (B, NumClasses)
        """
        # Create boolean mask (B, T) where 1 is valid, 0 is padding
        max_len = skel.size(1)
        device = skel.device

        # arange: [0, 1, ..., max_len-1]
        # mask[b, t] is True if t < lengths[b]
        mask = torch.arange(max_len, device=device).expand(
            len(lengths), max_len
        ) < lengths.to(device).unsqueeze(1)
        mask_float = mask.float()

        # 1. Stems Processing
        s_feat = self.skel_stem(skel)
        a_feat = self.audio_stem(audio)

        # 2. Fusion
        y = self.fusion(s_feat, a_feat, mask_float)

        # 3. Aux Head (Get Semantic Set Predictions and Refined Global Context)
        aux_preds, g_refined = self.aux_head(y, mask_float)

        # 4. Backbone (BiGRU with injection)
        # Lengths must be on CPU for pack_padded_sequence
        lengths_cpu = lengths.cpu()
        features = self.backbone(y, g_refined, lengths_cpu)

        # 5. Classification
        logits = self.classifier(features)

        return logits, aux_preds
