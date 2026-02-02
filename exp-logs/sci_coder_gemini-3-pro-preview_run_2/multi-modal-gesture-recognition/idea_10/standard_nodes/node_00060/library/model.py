import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.utils import make_pad_mask


class GatedMultimodalUnit(nn.Module):
    """
    Gated Multimodal Unit (GMU) for fusing Skeleton and Audio features.

    Mechanism:
    1. Project Skeleton and Audio to a shared hidden dimension.
    2. Compute a gate z based on the concatenation of inputs.
    3. Fuse: h = z * tanh(h_skel) + (1-z) * tanh(h_audio)
    """

    def __init__(self, skeleton_dim, audio_dim, hidden_dim):
        super(GatedMultimodalUnit, self).__init__()

        # Projections for the modalities
        self.skel_proj = nn.Linear(skeleton_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        # Gate computation
        # Input to gate is concatenation of raw features
        self.gate_fc = nn.Linear(skeleton_dim + audio_dim, hidden_dim)

    def forward(self, skeleton, audio):
        """
        Args:
            skeleton: (Batch, Time, Skeleton_Dim)
            audio: (Batch, Time, Audio_Dim)

        Returns:
            fused: (Batch, Time, Hidden_Dim)
        """
        # Feature transformation
        h_skel = torch.tanh(self.skel_proj(skeleton))
        h_audio = torch.tanh(self.audio_proj(audio))

        # Gate calculation
        combined = torch.cat([skeleton, audio], dim=-1)
        z = torch.sigmoid(self.gate_fc(combined))

        # Fusion
        fused = z * h_skel + (1 - z) * h_audio
        return fused


class DilatedResidualLayer(nn.Module):
    """
    A single dilated residual block for the TCN.
    Conv1d (dilated) -> Norm -> ReLU -> Dropout -> Conv1d (1x1) -> Norm -> ReLU -> Dropout
    """

    def __init__(self, channels, kernel_size, dilation, dropout):
        super(DilatedResidualLayer, self).__init__()

        # Padding calculation to keep temporal dimension same
        # For dilation d and kernel k, padding p = (k-1) * d / 2 (if even padding)
        # PyTorch Conv1d padding adds to both sides. We need 'same' padding.
        # We will use explicit padding or trim. Here we use padding = (k-1) * d
        # and Chomp (slice) if implementing causal, but for offline recognition
        # we can use standard padding = dilation * (kernel_size - 1) // 2

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding=padding
        )
        self.norm1 = nn.InstanceNorm1d(channels, affine=True)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.norm2 = nn.InstanceNorm1d(channels, affine=True)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Channels, Time)
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.dropout2(out)

        return out + residual


class SingleStageTCN(nn.Module):
    """
    Single Stage Temporal Convolutional Network.
    Consists of stacked DilatedResidualLayers.
    """

    def __init__(
        self, input_dim, num_classes, num_channels, kernel_size, num_layers, dropout
    ):
        super(SingleStageTCN, self).__init__()

        # Input projection to channel dimension
        self.input_proj = nn.Conv1d(input_dim, num_channels, 1)

        layers = []
        for i in range(num_layers):
            dilation = 2**i
            layers.append(
                DilatedResidualLayer(num_channels, kernel_size, dilation, dropout)
            )

        self.layers = nn.ModuleList(layers)

        # Output projection to classes
        self.output_proj = nn.Conv1d(num_channels, num_classes, 1)

    def forward(self, x, mask):
        """
        Args:
            x: (Batch, Input_Dim, Time) - Probabilities from previous stage
            mask: (Batch, 1, Time) - Binary mask (1 for valid, 0 for pad)

        Returns:
            logits: (Batch, Num_Classes, Time)
        """
        # Apply mask to input to zero out padding noise
        x = x * mask

        out = self.input_proj(x)

        for layer in self.layers:
            out = layer(out)
            # Apply mask after each residual block to keep padding clean
            out = out * mask

        logits = self.output_proj(out)
        return logits


class GMD_CRCN(nn.Module):
    """
    Gated Masked Dual-Stage Cascaded Recurrent-Convolutional Network.

    Structure:
    1. Input: Skeleton (Pos+Vel) + Audio
    2. Fusion: GMU
    3. Stage 1: Bi-LSTM (Generation)
    4. Stage 2: TCN (Refinement)
    5. Stage 3: TCN (Refinement)
    """

    def __init__(self):
        super(GMD_CRCN, self).__init__()

        # --- Dimensions ---
        skel_dim = Config.SKELETON_INPUT_SIZE + Config.VELOCITY_INPUT_SIZE
        audio_dim = Config.AUDIO_INPUT_SIZE
        fusion_hidden = Config.FUSION_HIDDEN_SIZE
        num_classes = Config.NUM_CLASSES

        # --- Stage 1: Generation ---
        self.gmu = GatedMultimodalUnit(skel_dim, audio_dim, fusion_hidden)

        self.lstm = nn.LSTM(
            input_size=fusion_hidden,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.LSTM_DROPOUT,
        )

        # Project LSTM output (2 * hidden) to classes
        self.stage1_fc = nn.Linear(Config.LSTM_HIDDEN_SIZE * 2, num_classes)

        # --- Stage 2: Coarse Refinement ---
        # Input is probabilities (num_classes)
        self.stage2_tcn = SingleStageTCN(
            input_dim=num_classes,
            num_classes=num_classes,
            num_channels=Config.TCN_NUM_CHANNELS,
            kernel_size=Config.TCN_KERNEL_SIZE,
            num_layers=Config.TCN_NUM_LAYERS,
            dropout=Config.TCN_DROPOUT,
        )

        # --- Stage 3: Fine Sharpening ---
        self.stage3_tcn = SingleStageTCN(
            input_dim=num_classes,
            num_classes=num_classes,
            num_channels=Config.TCN_NUM_CHANNELS,
            kernel_size=Config.TCN_KERNEL_SIZE,
            num_layers=Config.TCN_NUM_LAYERS,
            dropout=Config.TCN_DROPOUT,
        )

    def forward(self, pos, vel, audio, lengths):
        """
        Args:
            pos: (Batch, Time, 36)
            vel: (Batch, Time, 36)
            audio: (Batch, Time, 13)
            lengths: (Batch,)

        Returns:
            dict: {
                'stage1': logits (B, C, T),
                'stage2': logits (B, C, T),
                'stage3': logits (B, C, T)
            }
        """
        # 1. Prepare Inputs
        # Concatenate Position and Velocity
        skel_input = torch.cat([pos, vel], dim=-1)  # (B, T, 72)

        # Create Mask for TCN stages: (Batch, 1, Time)
        # make_pad_mask returns (Batch, Time) boolean where True is valid
        mask = make_pad_mask(lengths, max_len=pos.size(1)).to(pos.device)
        mask_expanded = mask.unsqueeze(1).float()  # (B, 1, T)

        # 2. Stage 1: Generation
        # Fusion
        fused = self.gmu(skel_input, audio)  # (B, T, Hidden)

        # LSTM
        # Pack sequence not strictly necessary if we use mask later,
        # but good for efficiency. Here we use standard pass for simplicity
        # as we have a dense tensor.
        lstm_out, _ = self.lstm(fused)  # (B, T, 2*Hidden)

        # Logits Stage 1
        logits_1 = self.stage1_fc(lstm_out)  # (B, T, Classes)

        # Transpose for TCN: (B, Classes, T)
        logits_1_t = logits_1.permute(0, 2, 1)

        # 3. Stage 2: Coarse Refinement
        # Input: Softmax Probabilities from Stage 1
        probs_1 = F.softmax(logits_1_t, dim=1)

        # TCN Forward (Masking applied inside)
        logits_2_t = self.stage2_tcn(probs_1, mask_expanded)  # (B, C, T)

        # 4. Stage 3: Fine Sharpening
        # Input: Softmax Probabilities from Stage 2
        probs_2 = F.softmax(logits_2_t, dim=1)

        # TCN Forward
        logits_3_t = self.stage3_tcn(probs_2, mask_expanded)  # (B, C, T)

        return {
            "stage1": logits_1_t,  # (B, C, T)
            "stage2": logits_2_t,  # (B, C, T)
            "stage3": logits_3_t,  # (B, C, T)
        }
