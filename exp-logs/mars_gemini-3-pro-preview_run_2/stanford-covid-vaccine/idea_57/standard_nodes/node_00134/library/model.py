import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DirectAccessTCNBlock(nn.Module):
    """
    Single-Layer Dilated Block with Post-Activation structure.
    Architecture: Dilated Conv (k=3) -> LN -> SiLU -> Pointwise Conv (k=1) -> LN -> SiLU -> Dropout.
    """

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.ln1 = nn.LayerNorm(out_channels)

        self.conv_pointwise = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.ln2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x):
        # x: (B, C_in, L)

        # 1. Dilated Conv
        out = self.conv_dilated(x)

        # 2. LN + SiLU
        out = out.transpose(1, 2)  # (B, L, C)
        out = self.ln1(out)
        out = F.silu(out)
        out = out.transpose(1, 2)  # (B, C, L)

        # 3. Pointwise Conv
        out = self.conv_pointwise(out)

        # 4. LN + SiLU + Dropout
        out = out.transpose(1, 2)
        out = self.ln2(out)
        out = F.silu(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)

        return out


class DirectAccessBackbone(nn.Module):
    """
    Direct-Access Dense Dilated TCN Backbone.
    Input to block k is Concat(All Previous Outputs, Raw Features).
    """

    def __init__(self, in_channels, raw_channels, growth_rate=32):
        super().__init__()
        self.dilations = Config.DILATIONS
        self.growth_rate = growth_rate
        self.blocks = nn.ModuleList()

        # Initial input channels (Branch A + Branch B)
        current_channels = in_channels

        for d in self.dilations:
            # Input to block includes current accumulated features + Raw Features (Branch A)
            # Note: If 'current_channels' already includes Branch A (via dense connection from start),
            # we strictly follow the prompt: "concatenation of the outputs of all prior blocks AND the raw One-Hot input features".
            # In a dense setup, 'current_channels' grows. We add 'raw_channels' to the input of the block conv.

            block_in_channels = current_channels + raw_channels

            block = DirectAccessTCNBlock(block_in_channels, growth_rate, dilation=d)
            self.blocks.append(block)

            # Dense connection: Output is concatenated to the stream
            current_channels += growth_rate

        self.out_channels = current_channels
        self.projection = nn.Linear(current_channels, Config.LATENT_DIM)

    def forward(self, x, raw_x):
        # x: Initial fused input (B, C_in, L)
        # raw_x: Branch A raw features (B, C_raw, L)

        features = [x]

        for block in self.blocks:
            # Concatenate all previous features to form the dense stream
            dense_input = torch.cat(features, dim=1)

            # Explicit Direct Access: Concatenate Raw Features to the block input
            block_input = torch.cat([dense_input, raw_x], dim=1)

            # Compute block output
            out = block(block_input)

            # Add output to features list
            features.append(out)

        # Final concatenation of all outputs
        final_dense = torch.cat(features, dim=1)  # (B, Total_C, L)

        # Project to Latent Dim
        final_dense = final_dense.transpose(1, 2)  # (B, L, Total_C)
        z = self.projection(final_dense)  # (B, L, Latent)

        return z


class FeedbackModule(nn.Module):
    """
    Global-Context Feedback Module with Topology Access.
    Lightweight Dense TCN.
    """

    def __init__(self, in_channels, topology_channels, growth_rate=16):
        super().__init__()
        # 3 layers of feedback processing as a lightweight TCN
        self.num_layers = 3
        self.growth_rate = growth_rate
        self.blocks = nn.ModuleList()
        self.dilations = [1, 2, 4]

        current_channels = in_channels

        for i in range(self.num_layers):
            d = self.dilations[i]
            # Input: Current Stream + Topology Features
            block_in = current_channels + topology_channels

            block = DirectAccessTCNBlock(block_in, growth_rate, dilation=d)
            self.blocks.append(block)

            current_channels += growth_rate

        self.projection = nn.Linear(current_channels, Config.FEEDBACK_DIM)

    def forward(self, y_pred, topology_features):
        # y_pred: (B, 5, L)
        # topology_features: (B, Topo_C, L)

        features = [y_pred]

        for block in self.blocks:
            dense_input = torch.cat(features, dim=1)

            # Inject topology features at every block
            block_input = torch.cat([dense_input, topology_features], dim=1)

            out = block(block_input)
            features.append(out)

        final_dense = torch.cat(features, dim=1)
        final_dense = final_dense.transpose(1, 2)  # (B, L, C)
        e_fb = self.projection(final_dense)  # (B, L, Feedback_Dim)

        return e_fb


class GCDARN(nn.Module):
    """
    Global-Context Direct-Access Recurrent Network.
    """

    def __init__(self):
        super().__init__()

        # Dimensions
        self.dim_seq = len(Config.BASES)
        self.dim_struct = len(Config.STRUCTURES)
        self.dim_loop = len(Config.LOOP_TYPES)
        self.dim_partner = self.dim_seq

        self.raw_dim = (
            self.dim_seq + self.dim_struct + self.dim_loop + self.dim_partner
        )  # 18
        self.topo_dim = self.dim_struct + self.dim_loop  # 10

        # Branch B (Context)
        self.branch_b_conv = nn.Conv1d(self.raw_dim, 32, kernel_size=3, padding=1)
        self.branch_b_ln = nn.LayerNorm(32)

        # Backbone
        # Input to backbone is Concat(Branch A, Branch B) = 18 + 32 = 50
        self.backbone = DirectAccessBackbone(
            in_channels=50, raw_channels=self.raw_dim, growth_rate=32
        )

        # Feedback Module
        self.feedback_net = FeedbackModule(
            in_channels=5, topology_channels=self.topo_dim, growth_rate=16
        )

        # Interaction & Aggregation
        # Input to GRU: (Latent + Feedback) * 2 (Self + Partner)
        gru_input_dim = (Config.LATENT_DIM + Config.FEEDBACK_DIM) * 2

        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT,
        )

        self.head = nn.Linear(Config.HIDDEN_DIM * 2, 5)

        # Channel Mask for Feedback (Register buffer to handle device placement)
        # Targets: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        # Scored: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
        # Unscored: deg_pH10(2), deg_50C(4)
        # Mask: 1 for kept, 0 for zeroed
        mask_val = [1.0, 1.0, 0.0, 1.0, 0.0]
        self.register_buffer(
            "channel_mask", torch.tensor(mask_val, dtype=torch.float32).view(1, 5, 1)
        )

    def forward(self, features, partner_indices):
        """
        features: (B, L, 18)
        partner_indices: (B, L)
        """
        B, L, _ = features.shape

        # 1. Prepare Inputs
        x_raw = features.transpose(1, 2)  # (B, 18, L)

        # Extract Topology Features for Feedback (Struct + Loop)
        # Struct is index 4:7, Loop is 7:14 in the 18-dim vector
        # 0-4: Seq, 4-7: Struct, 7-14: Loop, 14-18: Partner
        x_topo = x_raw[:, 4:14, :]  # (B, 10, L)

        # 2. Hybrid Input Stem
        # Branch A: x_raw
        # Branch B: Conv -> LN -> SiLU
        x_ctx = self.branch_b_conv(x_raw)
        x_ctx = x_ctx.transpose(1, 2)
        x_ctx = self.branch_b_ln(x_ctx)
        x_ctx = F.silu(x_ctx)
        x_ctx = x_ctx.transpose(1, 2)  # (B, 32, L)

        # Fusion
        x_backbone_in = torch.cat([x_raw, x_ctx], dim=1)  # (B, 50, L)

        # 3. Backbone (Static)
        z = self.backbone(x_backbone_in, x_raw)  # (B, L, Latent)

        # 4. Iterative Refinement Loop

        # Initial prediction (Zero)
        y_prev = torch.zeros(B, 5, L, device=features.device)

        outputs = []

        # We perform 2 passes
        # Pass 1: Uses zero feedback
        # Pass 2: Uses prediction from Pass 1 (detached, channel masked)

        for i in range(2):
            # A. Prepare Feedback
            if i == 0:
                # First pass: use zero init
                curr_feedback_input = y_prev
            else:
                # Subsequent pass: use previous output
                # Detach gradients
                prev_out = outputs[-1].detach()  # (B, L, 5)
                prev_out = prev_out.transpose(1, 2)  # (B, 5, L)
                # Apply Channel Mask (zero out unscored channels)
                curr_feedback_input = prev_out * self.channel_mask

            # B. Feedback Module
            e_fb = self.feedback_net(
                curr_feedback_input, x_topo
            )  # (B, L, Feedback_Dim)

            # C. Interaction (Gathering)
            # Node Vector: [Z, E_fb]
            node_vec = torch.cat([z, e_fb], dim=2)  # (B, L, Latent+Feedback)

            # Gather Partner Vectors
            # Flatten for gathering
            node_vec_flat = node_vec.reshape(B * L, -1)  # (B*L, Dim)

            # Create gather indices
            batch_idx = torch.arange(B, device=features.device).view(B, 1).expand(B, L)
            gather_idx = batch_idx * L + partner_indices  # (B, L)

            # Handle -1 indices (unpaired) by clamping to 0 temporarily, then masking
            safe_gather_idx = gather_idx.clamp(min=0)

            partner_vec = node_vec_flat[safe_gather_idx.view(-1)].view(B, L, -1)

            # Mask unpaired positions (where partner_indices == -1)
            mask_unpaired = (partner_indices == -1).unsqueeze(-1)  # (B, L, 1)
            partner_vec = partner_vec.masked_fill(mask_unpaired, 0.0)

            # D. Fusion & Aggregation
            gru_in = torch.cat([node_vec, partner_vec], dim=2)  # (B, L, Dim*2)

            gru_out, _ = self.gru(gru_in)

            # E. Prediction Head
            y_curr = self.head(gru_out)  # (B, L, 5)
            outputs.append(y_curr)

        # Return both outputs for loss calculation
        return outputs[0], outputs[1]
