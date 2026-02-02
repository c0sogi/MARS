import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class GatedResidualBlock(nn.Module):
    """
    Gated Activation Block with 1x1 Output Projection.
    Structure:
        Z = tanh(W_f * X) * sigmoid(W_g * X)
        H = W_proj * Z
        Y = X + H
    """

    def __init__(self, channels, kernel_size, dilation, dropout):
        super(GatedResidualBlock, self).__init__()

        # Padding for 'same' output size with dilation
        # Assuming kernel_size is odd (Config.TCN_KERNEL_SIZE is 3)
        padding = (kernel_size - 1) * dilation // 2

        self.filter_conv = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding=padding
        )
        self.gate_conv = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding=padding
        )
        # 1x1 Output Projection
        self.proj = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, C, T)
        Returns:
            y: (B, C, T)
        """
        # Gated Activation
        filt = torch.tanh(self.filter_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        z = filt * gate

        # Projection and Dropout
        h = self.proj(z)
        h = self.dropout(h)

        # Residual Connection
        return x + h


class BiLSTMEncoder(nn.Module):
    """
    Stage 1: Multi-Task Recurrent Encoder.
    Backbone: Bi-Directional LSTM.
    Outputs: Latent Features (H_enc), Class Probs, Boundary Probs.
    """

    def __init__(self):
        super(BiLSTMEncoder, self).__init__()

        self.input_dim = Config.INPUT_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.ENCODER_LAYERS
        self.num_classes = Config.NUM_CLASSES

        # Bi-LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=Config.ENCODER_BIDIRECTIONAL,
            dropout=Config.ENCODER_DROPOUT if self.num_layers > 1 else 0,
        )

        # Projection to reduce dimension if bidirectional (2*hidden -> hidden)
        # This creates the 256-dim H_enc expected by refinement stages
        lstm_out_dim = (
            self.hidden_dim * 2 if Config.ENCODER_BIDIRECTIONAL else self.hidden_dim
        )
        self.feature_proj = nn.Linear(lstm_out_dim, self.hidden_dim)

        # Prediction Heads
        self.cls_head = nn.Linear(self.hidden_dim, self.num_classes)
        self.bnd_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, x, mask):
        """
        Args:
            x: (B, T, InputDim)
            mask: (B, T)
        Returns:
            h_enc: (B, T, HiddenDim) - Latent features
            cls_logits: (B, T, NumClasses)
            bnd_logits: (B, T, 1)
        """
        # LSTM Forward
        # We don't pack sequences here for simplicity, assuming masking handles padding downstream
        lstm_out, _ = self.lstm(x)

        # Project to H_enc
        h_enc = self.feature_proj(lstm_out)

        # Apply Mask to features immediately
        mask_expanded = mask.unsqueeze(-1)  # (B, T, 1)
        h_enc = h_enc * mask_expanded

        # Heads
        cls_logits = self.cls_head(h_enc)
        bnd_logits = self.bnd_head(h_enc)

        # Apply mask to logits
        cls_logits = cls_logits * mask_expanded
        bnd_logits = bnd_logits * mask_expanded

        return h_enc, cls_logits, bnd_logits


class RefinementStage(nn.Module):
    """
    Refinement Stage (Stage 2 & 3).
    Uses Feature Injection: Inputs = [Previous Probs, Previous Bnd, H_enc].
    Backbone: Gated TCN.
    """

    def __init__(self):
        super(RefinementStage, self).__init__()

        self.num_classes = Config.NUM_CLASSES
        self.hidden_dim = Config.HIDDEN_DIM
        self.layers = Config.TCN_LAYERS
        self.kernel_size = Config.TCN_KERNEL_SIZE
        self.dropout = Config.TCN_DROPOUT

        # Input Dimension Calculation
        # Probs (C) + Boundary (1) + H_enc (HiddenDim)
        self.input_dim = self.num_classes + 1 + self.hidden_dim

        # Input Adapter (1x1 Conv)
        self.input_proj = nn.Conv1d(self.input_dim, self.hidden_dim, 1)

        # TCN Blocks with Monotonically Increasing Dilation (1, 2, 4, ..., 512)
        self.blocks = nn.ModuleList()
        for i in range(self.layers):
            dilation = 2**i
            self.blocks.append(
                GatedResidualBlock(
                    self.hidden_dim, self.kernel_size, dilation, self.dropout
                )
            )

        # Prediction Heads (1x1 Convs)
        self.cls_head = nn.Conv1d(self.hidden_dim, self.num_classes, 1)
        self.bnd_head = nn.Conv1d(self.hidden_dim, 1, 1)

    def forward(self, prev_cls_logits, prev_bnd_logits, h_enc, mask):
        """
        Args:
            prev_cls_logits: (B, T, C)
            prev_bnd_logits: (B, T, 1)
            h_enc: (B, T, HiddenDim)
            mask: (B, T)
        Returns:
            cls_logits: (B, T, C)
            bnd_logits: (B, T, 1)
        """
        # Convert logits to probabilities for input
        prev_cls_probs = F.softmax(prev_cls_logits, dim=-1)
        prev_bnd_probs = torch.sigmoid(prev_bnd_logits)

        # Feature Injection: Concatenate [Probs, Bnd, H_enc]
        # Shape: (B, T, C+1+HiddenDim)
        concat_input = torch.cat([prev_cls_probs, prev_bnd_probs, h_enc], dim=-1)

        # Transpose for Conv1d: (B, Channels, T)
        x = concat_input.transpose(1, 2)

        # Apply Mask to input
        mask_expanded = mask.unsqueeze(1)  # (B, 1, T)
        x = x * mask_expanded

        # Adapter
        x = self.input_proj(x)

        # TCN Blocks
        for block in self.blocks:
            x = block(x)
            x = x * mask_expanded  # Inter-layer masking

        # Heads
        cls_logits = self.cls_head(x)
        bnd_logits = self.bnd_head(x)

        # Transpose back to (B, T, C)
        cls_logits = cls_logits.transpose(1, 2)
        bnd_logits = bnd_logits.transpose(1, 2)

        # Final Masking
        mask_t = mask.unsqueeze(-1)
        cls_logits = cls_logits * mask_t
        bnd_logits = bnd_logits * mask_t

        return cls_logits, bnd_logits


class FISGCN(nn.Module):
    """
    Feature-Injected Supervised Gated-Cascaded Network.
    Combines Encoder and multiple Refinement Stages.
    """

    def __init__(self):
        super(FISGCN, self).__init__()

        self.encoder = BiLSTMEncoder()

        self.stages = nn.ModuleList()
        for _ in range(Config.NUM_REFINEMENT_STAGES):
            self.stages.append(RefinementStage())

    def forward(self, x, mask):
        """
        Args:
            x: (B, T, InputDim)
            mask: (B, T)
        Returns:
            outputs: List of dicts [{'cls': ..., 'bnd': ...}, ...]
        """
        outputs = []

        # Stage 1: Encoder
        h_enc, cls_logits, bnd_logits = self.encoder(x, mask)

        outputs.append({"cls": cls_logits, "bnd": bnd_logits})

        # Refinement Stages
        current_cls = cls_logits
        current_bnd = bnd_logits

        for stage in self.stages:
            # Pass H_enc explicitly (Feature Injection)
            current_cls, current_bnd = stage(current_cls, current_bnd, h_enc, mask)

            outputs.append({"cls": current_cls, "bnd": current_bnd})

        return outputs
