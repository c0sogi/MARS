import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class PostActivationBlock(nn.Module):
    """
    Post-Activation Block:
    Conv1d(k=3) -> LN -> SiLU -> Conv1d(k=1) -> LN -> SiLU -> Dropout

    Designed to handle sparse inputs robustly by applying normalization
    after the convolution.
    """

    def __init__(self, in_channels, out_channels, dilation, dropout=0.1):
        super(PostActivationBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.ln1 = nn.LayerNorm(out_channels)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.ln2 = nn.LayerNorm(out_channels)
        self.act2 = nn.SiLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, L)
        out = self.conv1(x)

        # LayerNorm expects (B, L, C)
        out = out.permute(0, 2, 1)
        out = self.ln1(out)
        out = self.act1(out)
        out = out.permute(0, 2, 1)

        out = self.conv2(out)

        out = out.permute(0, 2, 1)
        out = self.ln2(out)
        out = self.act2(out)
        out = out.permute(0, 2, 1)

        out = self.dropout(out)
        return out


class DirectAccessDenseNet(nn.Module):
    """
    Backbone with Direct Raw Access.
    Input to block k is concatenation of [Raw_Input, Out_0, ..., Out_{k-1}].
    """

    def __init__(self, input_dim):
        super(DirectAccessDenseNet, self).__init__()

        self.dilations = Config.DILATIONS
        self.growth_rate = Config.HIDDEN_DIM
        self.blocks = nn.ModuleList()

        # Build blocks
        # Block 0 input: Raw Input (input_dim)
        # Block 1 input: Raw Input + Block 0 Out (input_dim + growth_rate)
        # ...
        current_dim = input_dim
        for d in self.dilations:
            self.blocks.append(
                PostActivationBlock(
                    in_channels=current_dim,
                    out_channels=self.growth_rate,
                    dilation=d,
                    dropout=Config.DROPOUT,
                )
            )
            current_dim += self.growth_rate

        # Final projection to Latent Dim Z
        # Input is concatenation of all blocks + raw input
        self.project_out = nn.Conv1d(current_dim, Config.LATENT_DIM, kernel_size=1)

    def forward(self, x):
        # x: (B, C_in, L)
        features = [x]

        for block in self.blocks:
            # Concatenate all previous features (Dense connection)
            # Note: features[0] is the raw input, ensuring Direct Raw Access
            dense_input = torch.cat(features, dim=1)
            out = block(dense_input)
            features.append(out)

        # Final concatenation
        all_features = torch.cat(features, dim=1)
        z = self.project_out(all_features)
        return z


class FeedbackTCN(nn.Module):
    """
    Lightweight Dense TCN for processing recycled predictions.
    Strictly masked input.
    """

    def __init__(self):
        super(FeedbackTCN, self).__init__()

        self.input_dim = Config.NUM_TARGETS
        self.embedding_dim = 32  # Initial projection
        self.growth_rate = Config.FEEDBACK_GROWTH_RATE
        # Use reduced dilations for lightweight feedback (Cite solution_lesson_node_00103)
        self.dilations = Config.FEEDBACK_DILATIONS

        # Initial embedding
        self.embedding = nn.Conv1d(self.input_dim, self.embedding_dim, kernel_size=1)

        self.blocks = nn.ModuleList()
        current_dim = self.embedding_dim

        for d in self.dilations:
            self.blocks.append(
                PostActivationBlock(
                    in_channels=current_dim,
                    out_channels=self.growth_rate,
                    dilation=d,
                    dropout=Config.DROPOUT,
                )
            )
            current_dim += self.growth_rate

        # Final projection to Feedback Dim
        self.project_out = nn.Conv1d(current_dim, Config.FEEDBACK_DIM, kernel_size=1)

    def forward(self, y):
        # y: (B, L, 5) -> Permute to (B, 5, L)
        x = y.permute(0, 2, 1)

        # Embed
        x = self.embedding(x)

        features = [x]
        for block in self.blocks:
            dense_input = torch.cat(features, dim=1)
            out = block(dense_input)
            features.append(out)

        all_features = torch.cat(features, dim=1)
        e_fb = self.project_out(all_features)
        return e_fb


class DARDN(nn.Module):
    """
    Direct-Access Recurrent Dense Network (DA-RDN).
    Combines Static Backbone, Feedback Loop, Partner Interaction, and Bi-GRU.
    """

    def __init__(self):
        super(DARDN, self).__init__()

        # Calculate input dimension
        # Sequence(4) + Structure(3) + Loop(7) + Partner(4) = 18
        self.input_dim = 18

        # Components
        self.backbone = DirectAccessDenseNet(self.input_dim)
        self.feedback_module = FeedbackTCN()

        # Interaction & Aggregation
        # Fusion Input: Z (64) + E_fb (32) = 96
        # Partner Vector also 96. Total Fusion = 192.
        self.fusion_dim = (Config.LATENT_DIM + Config.FEEDBACK_DIM) * 2

        self.rnn = nn.GRU(
            input_size=self.fusion_dim,
            hidden_size=64,  # "Compact Hidden Size"
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Head
        # Bi-GRU outputs 64*2 = 128
        self.head = nn.Linear(128, Config.NUM_TARGETS)

        # Masking indices for feedback (reactivity, deg_Mg_pH10, deg_Mg_50C)
        # Indices: 0, 1, 3
        self.register_buffer(
            "scored_mask", torch.tensor([1, 1, 0, 1, 0], dtype=torch.float32)
        )

    def _gather_partner(self, features, partner_indices):
        """
        Gathers features from partner positions.
        features: (B, L, C)
        partner_indices: (B, L)
        """
        B, L, C = features.shape

        # Mask for unpaired bases (-1)
        # (B, L, 1)
        mask = (partner_indices != -1).unsqueeze(-1).float()

        # Replace -1 with 0 for gather safety
        safe_indices = partner_indices.clone()
        safe_indices[safe_indices == -1] = 0

        # Expand indices for gather: (B, L, C)
        expanded_indices = safe_indices.unsqueeze(-1).expand(-1, -1, C)

        # Gather
        partner_features = torch.gather(features, 1, expanded_indices)

        # Apply mask (unpaired bases get 0 vector)
        partner_features = partner_features * mask

        return partner_features

    def forward_pass(self, z, e_fb, partner_indices):
        """
        Executes Interaction, Fusion, RNN, and Head.
        z: (B, C_z, L)
        e_fb: (B, C_fb, L)
        partner_indices: (B, L)
        """
        # Prepare features: (B, L, C)
        z_t = z.permute(0, 2, 1)
        e_fb_t = e_fb.permute(0, 2, 1)

        # Self Vector
        h_self = torch.cat([z_t, e_fb_t], dim=2)  # (B, L, 96)

        # Partner Vector
        h_partner = self._gather_partner(h_self, partner_indices)  # (B, L, 96)

        # Fusion
        h_combined = torch.cat([h_self, h_partner], dim=2)  # (B, L, 192)

        # RNN
        rnn_out, _ = self.rnn(h_combined)  # (B, L, 128)

        # Head
        y_pred = self.head(rnn_out)  # (B, L, 5)

        return y_pred

    def forward(self, inputs, partner_indices):
        """
        inputs: (B, L, 18)
        partner_indices: (B, L)
        """
        # 1. Static Backbone
        # Permute inputs to (B, 18, L) for CNN
        x = inputs.permute(0, 2, 1)
        z = self.backbone(x)  # (B, 64, L)

        # 2. Iterative Refinement Loop

        # --- Pass 1 ---
        # Initialize Feedback with Zeros
        B, L, _ = inputs.shape
        y_0 = torch.zeros((B, L, Config.NUM_TARGETS), device=inputs.device)

        # Feedback Module (Input 0 -> Output E_fb0)
        e_fb0 = self.feedback_module(y_0)  # (B, 32, L)

        # Predict Y1
        y_1 = self.forward_pass(z, e_fb0, partner_indices)

        # --- Pass 2 ---
        # Detach Y1 for feedback
        y_1_detached = y_1.detach()

        # Strict Masking: Zero out unscored columns (indices 2 and 4)
        # scored_mask is [1, 1, 0, 1, 0]
        y_1_masked = y_1_detached * self.scored_mask.view(1, 1, -1)

        # Feedback Module
        e_fb1 = self.feedback_module(y_1_masked)

        # Predict Y2
        y_2 = self.forward_pass(z, e_fb1, partner_indices)

        return y_1, y_2
