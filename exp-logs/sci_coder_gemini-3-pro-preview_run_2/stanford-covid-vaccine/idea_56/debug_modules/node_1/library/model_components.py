import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class Permute(nn.Module):
    """
    Helper module to permute tensor dimensions.
    Useful for switching between (N, C, L) and (N, L, C) within nn.Sequential.
    """

    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class HybridStem(nn.Module):
    """
    Splits input into two branches:
    - Branch A: Identity (Raw Features)
    - Branch B: Spatial Context (Conv1d k=3)
    Outputs the concatenation of both.
    """

    def __init__(self, in_channels, context_dim):
        super().__init__()
        self.branch_b = nn.Sequential(
            nn.Conv1d(in_channels, context_dim, kernel_size=3, padding=1),
            Permute(0, 2, 1),
            nn.LayerNorm(context_dim),
            nn.SiLU(),
            Permute(0, 2, 1),
        )

    def forward(self, x):
        # x: (N, C, L)
        # Branch A is just x
        out_b = self.branch_b(x)
        return torch.cat([x, out_b], dim=1)


class DilatedDenseBlock(nn.Module):
    """
    Single-Layer Dilated Block with Post-Activation structure.
    """

    def __init__(self, in_channels, growth_rate, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            # Spatial Mixing
            nn.Conv1d(
                in_channels,
                growth_rate,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            Permute(0, 2, 1),
            nn.LayerNorm(growth_rate),
            nn.SiLU(),
            Permute(0, 2, 1),
            # Channel Mixing
            nn.Conv1d(growth_rate, growth_rate, kernel_size=1),
            Permute(0, 2, 1),
            nn.LayerNorm(growth_rate),
            nn.SiLU(),
            Permute(0, 2, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class FeedbackTCN(nn.Module):
    """
    Lightweight Dense TCN for processing feedback.
    Injects topology features (Structure/Loop) at every layer.
    """

    def __init__(self, input_dim, topo_dim, growth_rate=16, output_dim=32, layers=4):
        super().__init__()
        self.layers = nn.ModuleList()

        # In a dense block, input size grows
        current_dim = input_dim + topo_dim

        for _ in range(layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(current_dim, growth_rate, kernel_size=3, padding=1),
                    Permute(0, 2, 1),
                    nn.LayerNorm(growth_rate),
                    nn.SiLU(),
                    Permute(0, 2, 1),
                )
            )
            current_dim += growth_rate

        self.project = nn.Conv1d(current_dim, output_dim, kernel_size=1)

    def forward(self, preds, topo_feats):
        # preds: (N, 5, L)
        # topo_feats: (N, 10, L)

        features = [preds, topo_feats]

        for layer in self.layers:
            # Dense connection: Concat all previous features
            in_tensor = torch.cat(features, dim=1)
            out = layer(in_tensor)
            features.append(out)

        # Final projection of all accumulated features
        total_concat = torch.cat(features, dim=1)
        return self.project(total_concat)


class HSDARNModel(nn.Module):
    """
    Hybrid-Stem Direct-Access Recurrent Network (HS-DARN).
    """

    def __init__(self):
        super().__init__()

        # --- Dimensions ---
        # Input channels: 4 (Seq) + 3 (Struct) + 7 (Loop) + 4 (Partner) = 18
        self.in_channels = 18

        # --- Hybrid Stem ---
        self.stem = HybridStem(self.in_channels, Config.LATENT_DIM)
        # Stem Output: 18 (Identity) + 64 (Context) = 82
        self.stem_out_dim = self.in_channels + Config.LATENT_DIM

        # --- Direct-Access Backbone ---
        self.blocks = nn.ModuleList()
        current_dim = self.stem_out_dim

        for d in Config.DILATIONS:
            blk = DilatedDenseBlock(
                in_channels=current_dim,
                growth_rate=Config.LATENT_DIM,
                dilation=d,
                dropout=Config.DROPOUT,
            )
            self.blocks.append(blk)
            # Dense Growth: Next block receives all previous outputs
            current_dim += Config.LATENT_DIM

        # Latent Projection (Z)
        self.latent_proj = nn.Conv1d(current_dim, Config.LATENT_DIM, kernel_size=1)

        # --- Feedback Module ---
        # Input: 5 predictions. Topo: 3 (Struct) + 7 (Loop) = 10.
        self.feedback_net = FeedbackTCN(
            input_dim=5,
            topo_dim=10,
            growth_rate=16,
            output_dim=Config.FEEDBACK_DIM,
            layers=3,
        )

        # --- Interaction & Aggregation ---
        # Input to RNN: (Z + E_fb) for Self AND Partner
        # (64 + 32) * 2 = 192
        rnn_input_dim = (Config.LATENT_DIM + Config.FEEDBACK_DIM) * 2

        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Head (Bidirectional -> 2 * Hidden)
        self.head = nn.Linear(Config.HIDDEN_DIM * 2, 5)

        # --- Buffers for Masking ---
        # Identify unscored columns to mask in feedback
        unscored_indices = [
            i
            for i, col in enumerate(Config.TARGET_COLS)
            if col not in Config.SCORED_COLS
        ]
        self.register_buffer(
            "unscored_idxs", torch.tensor(unscored_indices, dtype=torch.long)
        )

    def forward_backbone(self, x):
        # x: (N, 18, L)
        stem_out = self.stem(x)  # (N, 82, L)

        features = [stem_out]
        for block in self.blocks:
            # Direct Access: Concat everything so far
            in_tensor = torch.cat(features, dim=1)
            out = block(in_tensor)
            features.append(out)

        total_features = torch.cat(features, dim=1)
        z = self.latent_proj(total_features)  # (N, 64, L)
        return z

    def forward_head(self, z, feedback_emb, partner_map):
        # z: (N, 64, L)
        # feedback_emb: (N, 32, L)
        # partner_map: (N, L)

        # 1. Self Vector
        self_vec = torch.cat([z, feedback_emb], dim=1)  # (N, 96, L)

        # 2. Partner Vector (Gather)
        self_vec_t = self_vec.permute(0, 2, 1)  # (N, L, 96)
        batch_size, seq_len, dim = self_vec_t.shape

        # Handle -1 in partner_map (unpaired)
        p_indices = partner_map.clone()
        mask_unpaired = p_indices == -1
        p_indices[mask_unpaired] = 0  # Dummy index for gather

        idx_expanded = p_indices.unsqueeze(-1).expand(-1, -1, dim)
        partner_vec_t = torch.gather(self_vec_t, 1, idx_expanded)

        # Zero out unpaired
        partner_vec_t[mask_unpaired] = 0.0

        # 3. Fusion
        combined = torch.cat([self_vec_t, partner_vec_t], dim=2)  # (N, L, 192)

        # 4. Global Aggregation
        rnn_out, _ = self.rnn(combined)  # (N, L, 128)

        # 5. Prediction
        logits = self.head(rnn_out)  # (N, L, 5)
        return logits

    def forward(self, inputs, partner_map, targets=None):
        # inputs: (N, L, 18) -> Permute to (N, 18, L) for Conv1d
        x = inputs.permute(0, 2, 1)

        # Extract Topology Features (Struct + Loop)
        # Channels 4-6 (Struct), 7-13 (Loop) -> Indices 4 to 14
        topo_feats = x[:, 4:14, :]  # (N, 10, L)

        # 1. Compute Backbone Latent Z (Static)
        z = self.forward_backbone(x)

        # 2. Iterative Refinement Loop

        # --- Pass 1: Zero Feedback ---
        batch_size, _, seq_len = x.shape
        zero_preds = torch.zeros(batch_size, 5, seq_len, device=x.device)

        fb_emb_1 = self.feedback_net(zero_preds, topo_feats)
        y_hat_1 = self.forward_head(z, fb_emb_1, partner_map)  # (N, L, 5)

        # --- Pass 2: Feedback from Pass 1 ---
        # Masking strategy:
        # 1. Mask unscored columns (indices 2, 4 usually)
        # 2. Mask unscored positions (index >= 68) to reduce tail noise

        y_hat_1_masked = y_hat_1.clone()

        # Mask unscored columns
        if len(self.unscored_idxs) > 0:
            y_hat_1_masked[:, :, self.unscored_idxs] = 0.0

        # Mask unscored sequence positions
        if Config.PRED_LEN < seq_len:
            y_hat_1_masked[:, Config.PRED_LEN :, :] = 0.0

        # Detach gradients and permute for feedback net
        y_hat_1_in = y_hat_1_masked.permute(0, 2, 1).detach()  # (N, 5, L)

        fb_emb_2 = self.feedback_net(y_hat_1_in, topo_feats)
        y_hat_2 = self.forward_head(z, fb_emb_2, partner_map)

        return y_hat_1, y_hat_2
