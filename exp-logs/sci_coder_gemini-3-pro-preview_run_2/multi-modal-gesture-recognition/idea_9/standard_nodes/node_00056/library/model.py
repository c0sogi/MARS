import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class BiLSTMEncoder(nn.Module):
    """
    Stage 1: Generation (Recurrent Encoder)
    Processes raw frame-wise features using a Bi-Directional LSTM.
    """

    def __init__(self):
        super(BiLSTMEncoder, self).__init__()

        self.input_norm = nn.LayerNorm(Config.INPUT_DIM)

        self.lstm = nn.LSTM(
            input_size=Config.INPUT_DIM,
            hidden_size=Config.LSTM_HIDDEN_DIM,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.LSTM_DROPOUT if Config.LSTM_LAYERS > 1 else 0,
        )

        # Project from bidirectional hidden dim (hidden*2) to num_classes
        self.classifier = nn.Linear(Config.LSTM_HIDDEN_DIM * 2, Config.NUM_CLASSES)

    def forward(self, x, mask):
        """
        Args:
            x: (Batch, Frames, Input_Dim)
            mask: (Batch, Frames)
        Returns:
            logits: (Batch, Frames, Num_Classes)
        """
        # Apply LayerNorm
        x = self.input_norm(x)

        # Calculate lengths for packing
        lengths = mask.sum(dim=1).cpu().int()

        # Handle case where a sequence might be all zeros (though unlikely in valid data)
        # by clamping min length to 1 to prevent pack_padded_sequence error.
        lengths = torch.clamp(lengths, min=1)

        # Pack sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        # LSTM Forward
        packed_output, _ = self.lstm(packed_input)

        # Unpack sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=x.size(1)
        )

        # Project to logits
        logits = self.classifier(output)

        return logits


class DilatedResidualLayer(nn.Module):
    """
    A single residual block for the TCN with dilated convolutions.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(DilatedResidualLayer, self).__init__()

        # Calculate padding to maintain sequence length (Same padding)
        # Assuming kernel_size is odd (Config.TCN_KERNEL_SIZE is 3)
        self.padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 1x1 conv for residual connection if dimensions change
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu_out = nn.ReLU()

    def forward(self, x):
        # x is (Batch, Channels, Frames)
        residual = x

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu_out(out)
        return out


class SingleStageTCN(nn.Module):
    """
    Stage 2 & 3: Refinement (Deep TCN)
    Processes probability sequences using dilated temporal convolutions.
    """

    def __init__(self):
        super(SingleStageTCN, self).__init__()

        # Input is probability distribution over classes from previous stage
        input_dim = Config.NUM_CLASSES
        hidden_dim = Config.TCN_NUM_CHANNELS[0]  # Assuming uniform channel list

        # 1. Input Projection
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)

        # 2. Stack of Dilated Residual Layers
        layers = []
        num_layers = len(Config.TCN_NUM_CHANNELS)
        for i in range(num_layers):
            dilation = 2**i
            layers.append(
                DilatedResidualLayer(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=Config.TCN_KERNEL_SIZE,
                    dilation=dilation,
                    dropout=Config.TCN_DROPOUT,
                )
            )
        self.layers = nn.Sequential(*layers)

        # 3. Output Projection
        self.output_proj = nn.Conv1d(hidden_dim, Config.NUM_CLASSES, 1)

    def forward(self, x, mask):
        """
        Args:
            x: Input probabilities (Batch, Frames, Classes)
            mask: (Batch, Frames) - Not used directly in Conv, but passed for consistency
        Returns:
            logits: (Batch, Frames, Classes)
        """
        # Permute to (Batch, Channels, Frames) for Conv1d
        x = x.permute(0, 2, 1)

        # Forward pass
        x = self.input_proj(x)
        x = self.layers(x)
        x = self.output_proj(x)

        # Permute back to (Batch, Frames, Classes)
        logits = x.permute(0, 2, 1)

        return logits


class MDCRCN(nn.Module):
    """
    Masked Dual-Stage Cascaded Recurrent-Convolutional Network.
    Stage 1 (BiLSTM) -> Mask -> Stage 2 (TCN) -> Mask -> Stage 3 (TCN)
    """

    def __init__(self):
        super(MDCRCN, self).__init__()

        self.stage1 = BiLSTMEncoder()
        self.stage2 = SingleStageTCN()
        self.stage3 = SingleStageTCN()

    def forward(self, x, mask):
        """
        Args:
            x: Raw features (Batch, Frames, Input_Dim)
            mask: Binary mask (Batch, Frames)
        Returns:
            outputs: Dictionary containing logits from all stages
                     {'stage1': logits, 'stage2': logits, 'stage3': logits}
        """
        outputs = {}

        # --- Stage 1: Generation ---
        logits_1 = self.stage1(x, mask)
        outputs["stage1"] = logits_1

        # --- Inter-Stage Masking 1 ---
        # Convert logits to probabilities
        probs_1 = F.softmax(logits_1, dim=2)
        # Apply mask to zero out padding (Batch, Frames, 1)
        masked_probs_1 = probs_1 * mask.unsqueeze(-1)

        # --- Stage 2: Coarse Refinement ---
        logits_2 = self.stage2(masked_probs_1, mask)
        outputs["stage2"] = logits_2

        # --- Inter-Stage Masking 2 ---
        probs_2 = F.softmax(logits_2, dim=2)
        masked_probs_2 = probs_2 * mask.unsqueeze(-1)

        # --- Stage 3: Fine Sharpening ---
        logits_3 = self.stage3(masked_probs_2, mask)
        outputs["stage3"] = logits_3

        return outputs
