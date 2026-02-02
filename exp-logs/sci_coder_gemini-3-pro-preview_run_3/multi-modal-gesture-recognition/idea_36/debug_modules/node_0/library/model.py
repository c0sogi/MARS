import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class GatedEncoder(nn.Module):
    """
    Stage 1: Asymmetrically-Normalized Gated Encoder.
    Fuses multi-modal inputs, applies feature-wise gating, and encodes temporal dynamics via Bi-GRU.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, dropout_prob):
        super(GatedEncoder, self).__init__()

        # Feature-wise Input Gating: x_tilde = x * sigmoid(W * x + b)
        # This allows the model to dynamically suppress noisy feature channels.
        self.gate_fc = nn.Linear(input_dim, input_dim)

        # Bi-Directional GRU
        # config.HIDDEN_DIM represents the total hidden size.
        # Since it's bidirectional, each direction gets hidden_dim // 2.
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, skeleton, audio):
        # skeleton: (Batch, Time, 180)
        # audio: (Batch, Time, 13)

        # Early Fusion via Concatenation
        x = torch.cat([skeleton, audio], dim=2)  # (Batch, Time, 193)

        # Apply Gating
        gate = torch.sigmoid(self.gate_fc(x))
        x_gated = x * gate

        # Temporal Encoding
        gru_out, _ = self.gru(x_gated)  # (Batch, Time, hidden_dim)

        # Regularization
        out = self.dropout(gru_out)

        # Projection to Class Logits
        logits = self.fc(out)  # (Batch, Time, num_classes)

        return logits


class RefinementStage(nn.Module):
    """
    Stage 2 & 3: Monotonic Non-Causal Refinement.
    Uses Gated Dilated TCNs to refine probability sequences.
    """

    def __init__(self, num_classes, hidden_dim, dilations, kernel_size, dropout):
        super(RefinementStage, self).__init__()

        # Project class probabilities to hidden dimension
        self.input_proj = nn.Conv1d(num_classes, hidden_dim, 1)

        self.layers = nn.ModuleList()
        for d in dilations:
            # WaveNet-style Gated Activation requires 2 * hidden_dim output channels
            # (split into filter and gate)
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=hidden_dim,
                        out_channels=2 * hidden_dim,
                        kernel_size=kernel_size,
                        padding=d,  # Centered padding
                        dilation=d,
                    ),
                    nn.Dropout(dropout),
                )
            )

        # Project back to class logits
        self.output_proj = nn.Conv1d(hidden_dim, num_classes, 1)

    def forward(self, x):
        # Input x: (Batch, Time, num_classes) - Probabilities from previous stage

        # Permute for Conv1d: (Batch, num_classes, Time)
        x = x.permute(0, 2, 1)

        # Input Projection
        x = self.input_proj(x)

        # Dilated TCN Blocks
        for layer in self.layers:
            residual = x

            # Convolution + Dropout
            out = layer(x)

            # Split for Gated Activation
            filter_out, gate_out = out.chunk(2, dim=1)

            # Gated Activation: Tanh(filter) * Sigmoid(gate)
            out = torch.tanh(filter_out) * torch.sigmoid(gate_out)

            # Residual Connection
            x = out + residual

        # Output Projection
        out = self.output_proj(x)

        # Permute back: (Batch, Time, num_classes)
        out = out.permute(0, 2, 1)

        return out


class ANG_KN(nn.Module):
    """
    Asymmetrically-Normalized Gated-Kinematic Network (ANG-KN).
    Three-stage cascaded network for robust gesture recognition.
    """

    def __init__(self):
        super(ANG_KN, self).__init__()

        # Calculate Input Dimensions
        # Skeleton: 20 joints * 3 coords * 3 derivatives (Pos, Vel, Acc) = 180
        skel_dim = config.JOINTS_COUNT * 3 * 3
        audio_dim = config.N_MFCC  # 13
        input_dim = skel_dim + audio_dim

        # Stage 1: Encoder
        self.stage1 = GatedEncoder(
            input_dim=input_dim,
            hidden_dim=config.HIDDEN_DIM,
            num_classes=config.NUM_CLASSES,
            dropout_prob=config.ENCODER_DROPOUT,
        )

        # Refinement Internal Dimension
        # 64 is a robust width for refining 21 classes
        tcn_hidden_dim = 64

        # Stage 2: First Refinement
        self.stage2 = RefinementStage(
            num_classes=config.NUM_CLASSES,
            hidden_dim=tcn_hidden_dim,
            dilations=config.DILATIONS,
            kernel_size=config.KERNEL_SIZE,
            dropout=config.REFINEMENT_DROPOUT,
        )

        # Stage 3: Second Refinement (Independent Weights)
        self.stage3 = RefinementStage(
            num_classes=config.NUM_CLASSES,
            hidden_dim=tcn_hidden_dim,
            dilations=config.DILATIONS,
            kernel_size=config.KERNEL_SIZE,
            dropout=config.REFINEMENT_DROPOUT,
        )

    def forward(self, skeleton, audio):
        """
        Forward pass.
        Returns a dictionary of logits for Deep Supervision.
        """
        # --- Stage 1 ---
        s1_logits = self.stage1(skeleton, audio)
        # Convert logits to probabilities for the next stage
        s1_probs = torch.softmax(s1_logits, dim=2)

        # --- Stage 2 ---
        # Input: Probabilities from Stage 1
        s2_logits = self.stage2(s1_probs)
        s2_probs = torch.softmax(s2_logits, dim=2)

        # --- Stage 3 ---
        # Input: Probabilities from Stage 2
        s3_logits = self.stage3(s2_probs)

        return {"stage1": s1_logits, "stage2": s2_logits, "stage3": s3_logits}
