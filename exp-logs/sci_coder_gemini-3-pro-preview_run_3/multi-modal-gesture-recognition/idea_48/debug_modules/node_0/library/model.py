import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DecoupledInputGate(nn.Module):
    """
    Decoupled-Norm Input Gating mechanism.
    Splits the input into two paths:
      1. Gating Path: Normalized (LayerNorm) to learn stable attention weights.
      2. Signal Path: Raw input to preserve physical magnitude/dynamics.

    Formula:
        Gate = Sigmoid(Linear(LayerNorm(X)))
        Output = X * Gate
    """

    def __init__(self, input_dim):
        super(DecoupledInputGate, self).__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.gate_fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Time, Channels]
        Returns:
            torch.Tensor: Gated input of shape [Batch, Time, Channels]
        """
        # Path A: Gating (Normalized)
        x_norm = self.norm(x)
        gate = torch.sigmoid(self.gate_fc(x_norm))

        # Path B: Signal (Raw) -> Fused
        return x * gate


class StochasticDepth(nn.Module):
    """
    Implements Stochastic Depth (Drop Path).
    Randomly drops the residual branch during training with probability p.
    """

    def __init__(self, prob=0.2):
        super(StochasticDepth, self).__init__()
        self.prob = prob

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x

        keep_prob = 1.0 - self.prob
        # Compute shape for broadcasting: [Batch, 1, 1] for 1D/2D data
        # Assuming x is [Batch, Channels, Time] or similar, we drop the whole sample's branch
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize

        # Scale x by 1/keep_prob to preserve expectation
        return x.div(keep_prob) * random_tensor


class TCNResidualBlock(nn.Module):
    """
    Temporal Convolutional Network Residual Block with Stochastic Depth.
    Uses centered padding for non-causal convolution (since we have full sequence).
    """

    def __init__(self, channels, kernel_size, dilation, dropout, stochastic_depth_prob):
        super(TCNResidualBlock, self).__init__()

        # Calculate padding for centered convolution
        # padding = (kernel_size - 1) * dilation / 2
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.stochastic_depth = StochasticDepth(prob=stochastic_depth_prob)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [Batch, Channels, Time]
        """
        residual = x

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Apply Stochastic Depth to the residual branch
        out = self.stochastic_depth(out)

        return residual + out


class Stage1_Encoder(nn.Module):
    """
    Stage 1: Decoupled-Norm Gated Encoder.
    Input: Raw Features -> Gating -> Bi-GRU -> Projection -> Logits
    """

    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super(Stage1_Encoder, self).__init__()

        self.gate = DecoupledInputGate(input_dim)

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=Config.GRU_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)
        # Bi-GRU outputs hidden_dim * 2
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [Batch, Channels, Time]
        Returns:
            torch.Tensor: Logits [Batch, Classes, Time]
        """
        # Transpose to [Batch, Time, Channels] for GRU/Linear layers
        x = x.transpose(1, 2)

        # Apply Decoupled Gating
        x = self.gate(x)

        # Bi-GRU
        # self.gru returns (output, h_n)
        x, _ = self.gru(x)

        x = self.dropout(x)

        # Project to classes
        logits = self.classifier(x)

        # Transpose back to [Batch, Classes, Time] for TCN stages
        logits = logits.transpose(1, 2)

        return logits


class Stage_Refinement(nn.Module):
    """
    Refinement Stage (Stage 2 & 3).
    Input: Logits (Probabilities) from previous stage.
    Architecture: Stack of TCN Residual Blocks with Stochastic Depth.
    """

    def __init__(self, num_classes, kernel_size, dilations, dropout, stoch_prob):
        super(Stage_Refinement, self).__init__()

        layers = []
        for dilation in dilations:
            layers.append(
                TCNResidualBlock(
                    channels=num_classes,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    stochastic_depth_prob=stoch_prob,
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Logits from previous stage [Batch, Classes, Time]
        Returns:
            torch.Tensor: Refined Logits [Batch, Classes, Time]
        """
        # Input is logits. We usually apply Softmax before feeding to next stage
        # to represent probabilities, but standard practice in MS-TCN is often
        # feeding raw logits or softmaxed probs.
        # The prompt says "Input: Strictly the class probabilities (P1)".

        probs = F.softmax(x, dim=1)

        # Pass through TCN stack
        out = self.net(probs)

        return out


class SD_DGN(nn.Module):
    """
    Stochastic-Depth Decoupled-Gated Network (SD-DGN).
    Three-Stage Cascaded Network.
    """

    def __init__(self):
        super(SD_DGN, self).__init__()

        # Calculate total input dimension
        input_dim = Config.INPUT_DIM_SKELETON + Config.INPUT_DIM_AUDIO

        # Stage 1: Encoder
        self.stage1 = Stage1_Encoder(
            input_dim=input_dim,
            hidden_dim=Config.HIDDEN_SIZE,
            num_classes=Config.NUM_CLASSES,
            dropout=Config.DROPOUT,
        )

        # Stage 2: Refinement
        self.stage2 = Stage_Refinement(
            num_classes=Config.NUM_CLASSES,
            kernel_size=Config.KERNEL_SIZE,
            dilations=Config.DILATIONS,
            dropout=Config.DROPOUT,
            stoch_prob=Config.STOCHASTIC_DEPTH_PROB,
        )

        # Stage 3: Refinement (Independent weights)
        self.stage3 = Stage_Refinement(
            num_classes=Config.NUM_CLASSES,
            kernel_size=Config.KERNEL_SIZE,
            dilations=Config.DILATIONS,
            dropout=Config.DROPOUT,
            stoch_prob=Config.STOCHASTIC_DEPTH_PROB,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features [Batch, Channels, Time]
        Returns:
            tuple: (logits1, logits2, logits3)
        """
        # Stage 1
        logits1 = self.stage1(x)

        # Stage 2 (Refines probabilities from Stage 1)
        logits2 = self.stage2(logits1)

        # Stage 3 (Refines probabilities from Stage 2)
        logits3 = self.stage3(logits2)

        return logits1, logits2, logits3
