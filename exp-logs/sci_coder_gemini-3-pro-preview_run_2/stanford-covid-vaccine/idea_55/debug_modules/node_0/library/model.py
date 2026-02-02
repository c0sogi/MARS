import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DirectAccessBlock(nn.Module):
    """
    Dilated Convolutional Block with Post-Activation structure.
    """

    def __init__(self, in_channels, hidden_dim, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.SiLU()
        self.pw = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.SiLU()

    def forward(self, x):
        # x: (N, C_in, L)
        out = self.conv(x)

        # LayerNorm expects (N, L, C)
        out = out.permute(0, 2, 1)
        out = self.ln1(out)
        out = self.act1(out)
        out = out.permute(0, 2, 1)

        out = self.pw(out)

        out = out.permute(0, 2, 1)
        out = self.ln2(out)
        out = self.act2(out)
        out = out.permute(0, 2, 1)
        return out


class FeedbackEncoder(nn.Module):
    """
    Encodes recycled predictions with Direct Access to Raw Topology.
    """

    def __init__(self, hidden_dim=32):
        super().__init__()
        # Input: 5 (pred) + 10 (topo: 3 struct + 7 loop) = 15
        self.stem = nn.Conv1d(15, hidden_dim, kernel_size=1)

        self.blocks = nn.ModuleList()
        dilations = [1, 2, 4]
        current_dim = hidden_dim

        for d in dilations:
            # Input to block is concat(prev_out, raw_topo)
            in_dim = current_dim + 10
            self.blocks.append(DirectAccessBlock(in_dim, hidden_dim, d))

    def forward(self, pred, raw_topo):
        # pred: (N, 5, L)
        # raw_topo: (N, 10, L)
        x = torch.cat([pred, raw_topo], dim=1)
        x = self.stem(x)

        for block in self.blocks:
            inp = torch.cat([x, raw_topo], dim=1)
            x = block(inp)

        return x


class DDARN(nn.Module):
    """
    Dual Direct-Access Recurrent Network (DDA-RN).
    Combines a Direct-Access backbone with a Direct-Access feedback loop
    and partner-aware RNN aggregation.
    """

    def __init__(self):
        super().__init__()

        self.raw_dim = 18
        self.topo_indices = slice(4, 14)  # Indices for Struct(3) + Loop(7) in X
        self.hidden_dim = Config.HIDDEN_DIM
        self.fb_dim = 32

        # --- Backbone ---
        self.stem = nn.Conv1d(self.raw_dim, self.hidden_dim, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList()
        dilations = [1, 2, 4, 8, 16, 32]

        # Direct Access Wiring: Input to block k is concat(all_prev_outputs, raw_input)
        # Initial input to block 0 is stem_out (hidden) + raw_input (raw)
        current_in_dim = self.hidden_dim + self.raw_dim
        self.out_dims = [self.hidden_dim]  # Stem output

        for d in dilations:
            self.blocks.append(DirectAccessBlock(current_in_dim, self.hidden_dim, d))
            self.out_dims.append(self.hidden_dim)
            current_in_dim += self.hidden_dim  # Accumulate output width

        # Latent Projection (1x1 Conv)
        total_backbone_out = sum(self.out_dims)
        self.proj = nn.Conv1d(total_backbone_out, self.hidden_dim, kernel_size=1)

        # --- Feedback ---
        self.feedback = FeedbackEncoder(self.fb_dim)

        # --- Interaction & Aggregation ---
        # Input: Self(Hidden + FB) + Partner(Hidden + FB)
        rnn_input_dim = (self.hidden_dim + self.fb_dim) * 2
        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # --- Head ---
        self.head = nn.Linear(self.hidden_dim * 2, 5)

    def forward(self, x, partners, prev_pred=None):
        # x: (N, 18, L)
        # partners: (N, L)
        # prev_pred: (N, 5, L)

        B, C, L = x.shape
        raw_topo = x[:, self.topo_indices, :]  # (N, 10, L)

        # 1. Backbone Pass
        stem_out = self.stem(x)
        outputs = [stem_out]

        current_in = torch.cat([stem_out, x], dim=1)

        for block in self.blocks:
            out = block(current_in)
            outputs.append(out)
            current_in = torch.cat([current_in, out], dim=1)

        z_all = torch.cat(outputs, dim=1)
        z = self.proj(z_all)  # (N, 64, L)

        # 2. Feedback Pass
        if prev_pred is None:
            prev_pred = torch.zeros((B, 5, L), device=x.device, dtype=x.dtype)

        fb_emb = self.feedback(prev_pred, raw_topo)  # (N, 32, L)

        # 3. Interaction
        node_feat = torch.cat([z, fb_emb], dim=1)  # (N, 96, L)

        # Gather partner features
        p_idx = partners.clone()
        mask_unpaired = p_idx == -1
        p_idx[mask_unpaired] = 0  # Safe index for gather

        idx_expanded = p_idx.unsqueeze(1).expand(-1, node_feat.size(1), -1)
        partner_feat = torch.gather(node_feat, 2, idx_expanded)

        # Mask unpaired
        partner_feat = partner_feat * (~mask_unpaired.unsqueeze(1))

        # Concatenate
        combined = torch.cat([node_feat, partner_feat], dim=1)  # (N, 192, L)

        # 4. RNN & Head
        combined = combined.permute(0, 2, 1)  # (N, L, C)
        rnn_out, _ = self.rnn(combined)
        logits = self.head(rnn_out)  # (N, L, 5)

        return logits.permute(0, 2, 1)  # (N, 5, L)
