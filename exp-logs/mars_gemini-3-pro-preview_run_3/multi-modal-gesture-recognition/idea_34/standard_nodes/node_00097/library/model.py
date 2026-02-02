import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class ProjectedGatedEncoder(nn.Module):
    """
    Stage 1: Projected-Gated High-Capacity Encoder.
    Handles multi-modal input (Skeleton + Audio) via learnable projections,
    gating, and a Bi-GRU backbone.
    """

    def __init__(self):
        super(ProjectedGatedEncoder, self).__init__()

        # Dimensions
        self.skel_dim = config.INPUT_DIM_SKELETON
        self.audio_dim = config.INPUT_DIM_AUDIO
        self.proj_dim = config.PROJECTION_DIM
        self.hidden_dim = config.HIDDEN_DIM
        self.num_classes = config.NUM_CLASSES

        # 1. Modality-Specific Projections
        self.skel_proj = nn.Linear(self.skel_dim, self.proj_dim)
        self.audio_proj = nn.Linear(self.audio_dim, self.proj_dim)

        # Fused dimension
        self.fused_dim = self.proj_dim * 2

        # 2. Normalization & Gating
        self.layer_norm = nn.LayerNorm(self.fused_dim)
        self.gate_fc = nn.Linear(self.fused_dim, self.fused_dim)

        # 3. Backbone (Bi-GRU)
        # hidden_dim is total, so per direction is hidden_dim // 2
        self.gru = nn.GRU(
            input_size=self.fused_dim,
            hidden_size=self.hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # 4. Dropout & Output
        self.dropout = nn.Dropout(config.DROPOUT_ENCODER)
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x):
        # x shape: (Batch, Time, Input_Dim) where Input_Dim = 180 + 13

        # Split inputs
        skel_in = x[:, :, : self.skel_dim]
        audio_in = x[:, :, self.skel_dim :]

        # Project
        skel_emb = self.skel_proj(skel_in)
        audio_emb = self.audio_proj(audio_in)

        # Fuse
        fused = torch.cat([skel_emb, audio_emb], dim=2)

        # Normalize
        fused = self.layer_norm(fused)

        # Gating: E = E * sigmoid(W*E + b)
        gate = torch.sigmoid(self.gate_fc(fused))
        gated_features = fused * gate

        # Backbone
        # self.gru returns (output, h_n). output shape: (B, T, hidden_dim)
        gru_out, _ = self.gru(gated_features)

        gru_out = self.dropout(gru_out)

        # Classification
        logits = self.classifier(gru_out)

        return logits


class GatedDilatedConv(nn.Module):
    """
    Building block for the Refinement Stages.
    Standard Dilated Convolution with Gating mechanism (Tanh * Sigmoid).
    Uses centered padding (non-causal).
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(GatedDilatedConv, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation

        # Filter Convolution
        self.filter_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )

        # Gate Convolution
        self.gate_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )

        self.dropout = nn.Dropout(dropout)

        # 1x1 Conv for residual or mixing if needed?
        # In MS-TCN, usually just the gated output is the block output.
        # Since input/output channels are same (num_classes), we can add residual connection directly.

    def forward(self, x):
        # x shape: (Batch, Channels, Time)

        # Calculate padding for "Centered" alignment
        # Output length L_out = L_in + 2*padding - dilation*(kernel_size-1) - 1 + 1
        # We want L_out = L_in
        # 2*padding = dilation*(kernel_size-1)
        # For k=3, 2*padding = 2*dilation => padding = dilation
        padding = self.dilation

        # Apply padding manually or via conv argument?
        # Conv1d 'padding' arg adds zeros to both sides.

        # Filter
        filter_out = self.filter_conv(x)
        # Gate
        gate_out = self.gate_conv(x)

        # Since we didn't pad in the layer definition (default 0), we pad input or handle it.
        # Let's use F.pad on input to be explicit about centered padding.
        # Pad last dim (Time): (padding_left, padding_right)
        # x_padded = F.pad(x, (padding, padding)) -> This would be correct if we didn't use conv padding arg.
        # However, to be efficient, let's re-define layers with padding if possible.
        # But PyTorch Conv1d padding is symmetric. That works for us.
        # Wait, I initialized convs without padding. Let's fix that logic in execution.

        # Re-implementation of forward with functional padding for clarity
        x_padded = F.pad(x, (padding, padding))

        filter_out = self.filter_conv(x_padded)
        gate_out = self.gate_conv(x_padded)

        # Activation
        out = torch.tanh(filter_out) * torch.sigmoid(gate_out)
        out = self.dropout(out)

        return out


class RefinementStage(nn.Module):
    """
    Stage 2 & 3: Monotonic Non-Causal Refinement.
    Stack of Gated Dilated Temporal Convolutions.
    """

    def __init__(self):
        super(RefinementStage, self).__init__()

        self.num_classes = config.NUM_CLASSES
        self.dilations = config.TCN_DILATIONS
        self.kernel_size = config.TCN_KERNEL_SIZE
        self.dropout = config.DROPOUT_TCN

        self.layers = nn.ModuleList()

        # Input is class probabilities (num_classes)
        # Output is refined logits (num_classes)
        # We keep channel dimension constant = num_classes

        for dilation in self.dilations:
            self.layers.append(
                GatedDilatedConv(
                    in_channels=self.num_classes,
                    out_channels=self.num_classes,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    dropout=self.dropout,
                )
            )

    def forward(self, x):
        # x shape: (Batch, Time, Classes) -> Needs transpose for Conv1d
        out = x.permute(0, 2, 1)  # (B, C, T)

        for layer in self.layers:
            res = out
            out = layer(out)
            out = out + res  # Residual connection

        # Transpose back
        out = out.permute(0, 2, 1)  # (B, T, C)
        return out


class PG_HCKN(nn.Module):
    """
    Projected-Gated High-Capacity Kinematic Network.
    Three-Stage Cascaded Network:
    1. Encoder -> Logits1
    2. Softmax(Logits1) -> Refinement1 -> Logits2
    3. Softmax(Logits2) -> Refinement2 -> Logits3
    """

    def __init__(self):
        super(PG_HCKN, self).__init__()

        self.encoder = ProjectedGatedEncoder()
        self.refinement_1 = RefinementStage()
        self.refinement_2 = RefinementStage()

    def forward(self, x):
        # x: (Batch, Time, Features)

        # Stage 1
        logits_1 = self.encoder(x)
        probs_1 = F.softmax(logits_1, dim=2)

        # Stage 2
        logits_2 = self.refinement_1(probs_1)
        probs_2 = F.softmax(logits_2, dim=2)

        # Stage 3
        logits_3 = self.refinement_2(probs_2)

        # Return all logits for deep supervision
        return {"stage1": logits_1, "stage2": logits_2, "stage3": logits_3}
