import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DecoupledGatedBlock(nn.Module):
    """
    Implements the Decoupled-Norm Input Gating mechanism.

    Path A (Gate Generation): Norm -> Linear -> Sigmoid -> Gate
    Path B (Signal Retention): Raw Input
    Output: Raw Input * Gate

    This allows the model to suppress noise (via the gate) without destroying
    the magnitude hierarchy of the raw kinematic signals (e.g., Position >> Acc).
    """

    def __init__(self, input_dim):
        super(DecoupledGatedBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.gate_fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x: (Batch, Time, Features)

        # Path A: Normalized Gate Generation
        x_norm = self.layer_norm(x)
        gate = torch.sigmoid(self.gate_fc(x_norm))

        # Path B: Signal Retention & Fusion
        # Element-wise multiplication
        return x * gate


class GatedDilatedConv(nn.Module):
    """
    A single Gated Dilated Convolutional Block.
    Implements: Output = Tanh(Conv_f(x)) * Sigmoid(Conv_g(x)) + Residual
    Uses centered padding to maintain sequence length (Non-causal).
    """

    def __init__(self, channels, kernel_size, dilation, dropout=0.0):
        super(GatedDilatedConv, self).__init__()

        # Calculate padding to keep length constant
        # For k=3, dilation=d, padding = d
        self.padding = dilation * (kernel_size - 1) // 2

        self.conv_f = nn.Conv1d(
            channels, channels, kernel_size, padding=self.padding, dilation=dilation
        )
        self.conv_g = nn.Conv1d(
            channels, channels, kernel_size, padding=self.padding, dilation=dilation
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Channels, Time)

        f = self.conv_f(x)
        g = self.conv_g(x)

        # Gating mechanism
        out = torch.tanh(f) * torch.sigmoid(g)
        out = self.dropout(out)

        # Residual connection
        return x + out


class RefinementStage(nn.Module):
    """
    RF-Aligned Monotonic Refinement Stage.
    Input: Class Probabilities from previous stage (Batch, Time, NumClasses)
    Output: Refined Class Probabilities (Batch, Time, NumClasses)

    Architecture:
    1. Projection: NumClasses -> HiddenDim
    2. Stack of Gated Dilated Convs (Monotonically increasing dilation)
    3. Projection: HiddenDim -> NumClasses
    """

    def __init__(self, num_classes, hidden_dim, kernel_size, dilations, dropout=0.0):
        super(RefinementStage, self).__init__()

        self.input_proj = nn.Conv1d(num_classes, hidden_dim, kernel_size=1)

        self.layers = nn.ModuleList()
        for d in dilations:
            self.layers.append(
                GatedDilatedConv(hidden_dim, kernel_size, dilation=d, dropout=dropout)
            )

        self.output_proj = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (Batch, Time, NumClasses)
        # Permute for Conv1d: (Batch, NumClasses, Time)
        x = x.permute(0, 2, 1)

        out = self.input_proj(x)

        for layer in self.layers:
            out = layer(out)

        out = self.output_proj(out)

        # Permute back: (Batch, Time, NumClasses)
        return out.permute(0, 2, 1)


class DGC_KN(nn.Module):
    """
    Decoupled-Norm Gated Central-Kinematic Network.

    Structure:
    1. DecoupledGatedBlock (Input Preprocessing)
    2. Stage 1: Bi-GRU Encoder -> P1
    3. Stage 2: TCN Refinement (Input P1) -> P2
    4. Stage 3: TCN Refinement (Input P2) -> P3
    """

    def __init__(self):
        super(DGC_KN, self).__init__()

        # Dimensions
        input_dim = Config.TOTAL_INPUT_DIM
        hidden_dim = Config.HIDDEN_DIM
        num_classes = Config.NUM_CLASSES

        # --- Input Gating ---
        self.input_gate = DecoupledGatedBlock(input_dim)

        # --- Stage 1: Bi-GRU Encoder ---
        # Input projection to match hidden dim before GRU (optional but good practice)
        self.stage1_proj = nn.Linear(input_dim, hidden_dim)

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,  # Bidirectional, so output will be hidden_dim
            num_layers=Config.GRU_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT if Config.GRU_LAYERS > 1 else 0.0,
        )
        self.stage1_classifier = nn.Linear(hidden_dim, num_classes)

        # --- Stage 2: Refinement ---
        self.stage2 = RefinementStage(
            num_classes=num_classes,
            hidden_dim=Config.TCN_CHANNELS,
            kernel_size=Config.TCN_KERNEL_SIZE,
            dilations=Config.TCN_DILATIONS,
            dropout=Config.DROPOUT,
        )

        # --- Stage 3: Refinement ---
        self.stage3 = RefinementStage(
            num_classes=num_classes,
            hidden_dim=Config.TCN_CHANNELS,
            kernel_size=Config.TCN_KERNEL_SIZE,
            dilations=Config.TCN_DILATIONS,
            dropout=Config.DROPOUT,
        )

    def forward(self, x):
        # x: (Batch, Time, InputDim)

        # 1. Decoupled Input Gating
        x = self.input_gate(x)

        # 2. Stage 1: Encoder
        x_proj = F.relu(self.stage1_proj(x))
        gru_out, _ = self.gru(x_proj)
        logits_1 = self.stage1_classifier(gru_out)
        probs_1 = F.softmax(logits_1, dim=2)

        # 3. Stage 2: Refinement
        # Input is strictly probabilities from Stage 1
        logits_2 = self.stage2(probs_1)
        probs_2 = F.softmax(logits_2, dim=2)

        # 4. Stage 3: Refinement
        # Input is strictly probabilities from Stage 2
        logits_3 = self.stage3(probs_2)
        # probs_3 calculated outside if needed, usually logits returned for Loss

        return {
            "logits_1": logits_1,
            "logits_2": logits_2,
            "logits_3": logits_3,
            "probs_3": F.softmax(logits_3, dim=2),
        }
