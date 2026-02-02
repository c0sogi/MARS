import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DualStreamEncoder(nn.Module):
    """
    Stage 1: Dual-Stream Encoder.

    Processes static features (Skeleton Position + Audio) and dynamic features
    (Skeleton Velocity + Acceleration) in separate Bi-GRU streams.
    Fuses the representations to produce initial class logits.
    """

    def __init__(self):
        super(DualStreamEncoder, self).__init__()

        # Static Stream Input: 60 (Normalized Skeleton) + 13 (MFCC) = 73
        self.static_input_size = 73
        self.static_gru = nn.GRU(
            input_size=self.static_input_size,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.GRU_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT if Config.GRU_LAYERS > 1 else 0,
        )

        # Dynamic Stream Input: 60 (Velocity) + 60 (Acceleration) = 120
        self.dynamic_input_size = 120
        self.dynamic_gru = nn.GRU(
            input_size=self.dynamic_input_size,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.GRU_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT if Config.GRU_LAYERS > 1 else 0,
        )

        # Fusion Layer
        # Each Bi-GRU outputs 2 * hidden_size
        self.fusion_dim = (Config.HIDDEN_SIZE * 2) * 2

        self.classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT), nn.Linear(self.fusion_dim, Config.NUM_CLASSES)
        )

    def forward(self, static_x, dynamic_x):
        """
        Args:
            static_x (torch.Tensor): (Batch, Time, 73)
            dynamic_x (torch.Tensor): (Batch, Time, 120)

        Returns:
            torch.Tensor: Logits of shape (Batch, Time, NumClasses)
        """
        # Pass through GRUs
        # Output shape: (Batch, Time, Hidden * 2)
        static_out, _ = self.static_gru(static_x)
        dynamic_out, _ = self.dynamic_gru(dynamic_x)

        # Concatenate features along the channel dimension
        fused = torch.cat([static_out, dynamic_out], dim=2)

        # Project to class logits
        logits = self.classifier(fused)

        return logits


class RefinementTCN(nn.Module):
    """
    Stage 2: Bottleneck Refinement Module.

    A Dilated Temporal Convolutional Network that takes class probabilities
    as input and refines them to ensure temporal consistency.
    """

    def __init__(self):
        super(RefinementTCN, self).__init__()

        self.layers = nn.ModuleList()

        # Input Projection: NumClasses -> First Hidden Channel
        # Kernel size 1 acts as a channel-wise dense layer
        self.input_proj = nn.Conv1d(
            in_channels=Config.NUM_CLASSES,
            out_channels=Config.TCN_NUM_CHANNELS[0],
            kernel_size=1,
        )

        # Stack of Dilated Convolutions
        # We iterate through the channel configuration list
        for i, out_channels in enumerate(Config.TCN_NUM_CHANNELS):
            # Determine input channels for this layer
            in_channels = (
                Config.TCN_NUM_CHANNELS[i - 1] if i > 0 else Config.TCN_NUM_CHANNELS[0]
            )

            # Exponential dilation: 1, 2, 4, ...
            dilation = 2**i
            kernel_size = Config.TCN_KERNEL_SIZE

            # Calculate padding to maintain temporal dimension (same padding)
            padding = (kernel_size - 1) * dilation // 2

            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.ReLU(),
                    nn.Dropout(Config.TCN_DROPOUT),
                )
            )

        # Output Projection: Last Hidden Channel -> NumClasses
        self.output_proj = nn.Conv1d(
            in_channels=Config.TCN_NUM_CHANNELS[-1],
            out_channels=Config.NUM_CLASSES,
            kernel_size=1,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input probabilities (Batch, Time, NumClasses)

        Returns:
            torch.Tensor: Refined logits (Batch, Time, NumClasses)
        """
        # Permute to (Batch, Channels, Time) for Conv1d
        x = x.permute(0, 2, 1)

        # Project input
        x = self.input_proj(x)

        # Apply dilated layers with residual connections
        for layer in self.layers:
            residual = x
            out = layer(x)

            # Add residual if shapes match (they should based on config)
            if residual.shape[1] == out.shape[1]:
                x = out + residual
            else:
                x = out

        # Project to output classes
        out = self.output_proj(x)

        # Permute back to (Batch, Time, Channels)
        out = out.permute(0, 2, 1)

        return out


class GestureNet(nn.Module):
    """
    End-to-End Dual-Stream Dynamic-Static Network with Bottleneck Refinement.
    """

    def __init__(self):
        super(GestureNet, self).__init__()
        self.encoder = DualStreamEncoder()
        self.refinement = RefinementTCN()

    def forward(self, static_x, dynamic_x):
        """
        Args:
            static_x (torch.Tensor): (Batch, Time, 73)
            dynamic_x (torch.Tensor): (Batch, Time, 120)

        Returns:
            stage1_logits (torch.Tensor): Output from Encoder (Batch, Time, C)
            stage2_logits (torch.Tensor): Output from Refinement (Batch, Time, C)
        """
        # Stage 1: Encoder
        stage1_logits = self.encoder(static_x, dynamic_x)

        # Information Bottleneck:
        # Convert logits to probabilities using Softmax.
        # This forces the refinement network to work with class confidences
        # rather than high-dimensional latent features.
        stage1_probs = F.softmax(stage1_logits, dim=2)

        # Stage 2: Refinement
        stage2_logits = self.refinement(stage1_probs)

        return stage1_logits, stage2_logits
