import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import MODEL_PARAMS


class BiLSTMEncoder(nn.Module):
    """
    Stage 1: Sequence Encoder using Bi-Directional LSTM.
    Extracts temporal features from skeleton and audio data and produces initial frame-wise predictions.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super(BiLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (Batch, Time, Input_Dim)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        out = self.dropout(out)
        logits = self.fc(out)  # (Batch, Time, Num_Classes)
        return logits


class DilatedResidualLayer(nn.Module):
    """
    Building block for the TCN Refinement Stage.
    Consists of Dilated Conv -> ReLU -> 1x1 Conv -> Dropout -> Residual Connection.
    """

    def __init__(self, channels, kernel_size, dilation, dropout):
        super(DilatedResidualLayer, self).__init__()
        # Padding to maintain temporal dimension: (k-1) * d // 2 for odd kernels
        padding = (kernel_size - 1) * dilation // 2

        self.conv_dilated = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (Batch, Channels, Time)
        out = self.conv_dilated(x)
        out = self.relu(out)
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class RefinementStage(nn.Module):
    """
    Stage 2: Temporal Refinement using MS-TCN style architecture.
    Takes class probabilities as input and refines them using a large temporal receptive field.
    """

    def __init__(self, num_layers, num_f_maps, num_classes, kernel_size, dropout):
        super(RefinementStage, self).__init__()
        self.conv_in = nn.Conv1d(num_classes, num_f_maps, 1)

        self.layers = nn.ModuleList(
            [
                DilatedResidualLayer(num_f_maps, kernel_size, 2**i, dropout)
                for i in range(num_layers)
            ]
        )

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask=None):
        # x: (Batch, Time, Num_Classes) -> Transpose to (Batch, Channels, Time) for Conv1d
        out = x.transpose(1, 2)
        out = self.conv_in(out)

        for layer in self.layers:
            out = layer(out)

        out = self.conv_out(out)
        out = out.transpose(1, 2)  # Back to (Batch, Time, Num_Classes)

        # Apply mask to zero out padding regions
        if mask is not None:
            out = out * mask.unsqueeze(-1)

        return out


class CRCN(nn.Module):
    """
    Cascaded Recurrent-Convolutional Network.
    Combines Bi-LSTM Encoder and multiple TCN Refinement Stages.
    Returns a list of outputs from all stages for deep supervision.
    """

    def __init__(self):
        super(CRCN, self).__init__()

        # Configuration
        input_dim = MODEL_PARAMS["input_dim"]
        num_classes = MODEL_PARAMS["num_classes"]

        # LSTM Params
        lstm_hidden = MODEL_PARAMS["lstm_hidden_dim"]
        lstm_layers = MODEL_PARAMS["lstm_layers"]
        lstm_drop = MODEL_PARAMS["lstm_dropout"]

        # TCN Params
        tcn_stages = MODEL_PARAMS["tcn_num_stages"]
        tcn_layers = MODEL_PARAMS["tcn_num_layers"]
        tcn_f_maps = MODEL_PARAMS["tcn_num_f_maps"]
        tcn_kernel = MODEL_PARAMS["tcn_kernel_size"]

        # Stage 1: Encoder
        self.encoder = BiLSTMEncoder(
            input_dim, lstm_hidden, lstm_layers, num_classes, lstm_drop
        )

        # Stage 2: Refinement Stages
        self.refinement_stages = nn.ModuleList(
            [
                RefinementStage(
                    tcn_layers, tcn_f_maps, num_classes, tcn_kernel, lstm_drop
                )
                for _ in range(tcn_stages)
            ]
        )

    def forward(self, x, lengths=None):
        """
        Forward pass.
        Args:
            x: (Batch, Time, Input_Dim)
            lengths: (Batch,) Tensor containing valid sequence lengths.
        Returns:
            outputs: List of tensors [(Batch, Time, Num_Classes), ...].
                     First element is Encoder output, subsequent are Refinement outputs.
        """
        # Create mask based on lengths
        mask = None
        if lengths is not None:
            batch_size, max_len, _ = x.shape
            device = x.device
            # (Batch, Time)
            mask = torch.arange(max_len, device=device).expand(
                batch_size, max_len
            ) < lengths.unsqueeze(1)
            mask = mask.float()

        # --- Stage 1: Prediction ---
        logits_1 = self.encoder(x)
        if mask is not None:
            logits_1 = logits_1 * mask.unsqueeze(-1)

        outputs = [logits_1]

        # Input to refinement stages is the probability distribution (Softmax)
        current_input = F.softmax(logits_1, dim=2)

        # --- Stage 2: Refinement ---
        for stage in self.refinement_stages:
            logits_ref = stage(current_input, mask)
            outputs.append(logits_ref)

            # Prepare input for next stage (if any)
            current_input = F.softmax(logits_ref, dim=2)

        return outputs
