import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class BiLSTMEncoder(nn.Module):
    """
    Generation Stage: Bi-Directional LSTM Encoder.

    Processes raw frame-wise features to generate initial class probabilities.
    """

    def __init__(self):
        super(BiLSTMEncoder, self).__init__()
        self.input_dim = Config.INPUT_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.LSTM_LAYERS
        self.num_classes = Config.NUM_CLASSES
        self.dropout_p = Config.DROPOUT

        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_p if self.num_layers > 1 else 0,
        )

        # Projection to Class Logits
        # Input: hidden_dim * 2 (bidirectional)
        self.fc = nn.Linear(self.hidden_dim * 2, self.num_classes)

        # Dropout before linear layer
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Time, InputDim)

        Returns:
            torch.Tensor: Class logits of shape (Batch, Classes, Time)
        """
        # LSTM Forward
        # out shape: (Batch, Time, HiddenDim * 2)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)

        out = self.dropout(out)

        # Project to classes
        # shape: (Batch, Time, NumClasses)
        logits = self.fc(out)

        # Permute to (Batch, Classes, Time) for TCN compatibility
        logits = logits.permute(0, 2, 1)

        return logits


class DilatedResidualLayer(nn.Module):
    """
    Building block for the TCN Refinement Stage.
    Uses dilated convolutions to capture temporal context.
    """

    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.kernel_size = Config.TCN_KERNEL_SIZE
        self.dropout_p = Config.DROPOUT

        # Padding to maintain temporal dimension
        # padding = (kernel_size - 1) * dilation / 2
        self.padding = (self.kernel_size - 1) * dilation // 2

        # Dilated Convolution
        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

        # 1x1 Convolution for feature mixing
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, kernel_size=1)

        self.dropout = nn.Dropout(self.dropout_p)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features (Batch, Channels, Time)
        """
        out = self.conv_dilated(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv_1x1(out)

        # Residual connection
        return x + out


class SingleStageTCN(nn.Module):
    """
    Refinement Stage: Multi-Stage TCN.

    Takes class probabilities from the previous stage and refines them
    using a stack of dilated residual layers.
    """

    def __init__(self):
        super(SingleStageTCN, self).__init__()
        self.num_classes = Config.NUM_CLASSES
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.TCN_LAYERS

        # Input Projection: Probabilities -> Hidden Dim
        self.conv_in = nn.Conv1d(self.num_classes, self.hidden_dim, kernel_size=1)

        # Stack of Dilated Residual Layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            dilation = 2**i
            self.layers.append(
                DilatedResidualLayer(dilation, self.hidden_dim, self.hidden_dim)
            )

        # Output Projection: Hidden Dim -> Logits
        self.conv_out = nn.Conv1d(self.hidden_dim, self.num_classes, kernel_size=1)

        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input probabilities (Batch, Classes, Time)

        Returns:
            torch.Tensor: Refined logits (Batch, Classes, Time)
        """
        out = self.conv_in(x)
        out = self.dropout(out)

        for layer in self.layers:
            out = layer(out)

        out = self.conv_out(out)
        return out


class ICRCN(nn.Module):
    """
    Iterative Cascaded Recurrent-Convolutional Network (IC-RCN).

    Consists of:
    1. Generation Stage (Bi-LSTM)
    2. Refinement Stage 1 (TCN)
    3. Refinement Stage 2 (TCN)
    """

    def __init__(self):
        super(ICRCN, self).__init__()

        # Stage 0: Generation
        self.gen = BiLSTMEncoder()

        # Stage 1: Refinement
        self.ref1 = SingleStageTCN()

        # Stage 2: Refinement
        self.ref2 = SingleStageTCN()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features (Batch, Time, InputDim)

        Returns:
            dict: Dictionary containing logits from all stages.
                  {'gen': ..., 'ref1': ..., 'ref2': ...}
        """
        # ---------------------------------------------------------
        # 1. Generation Stage
        # ---------------------------------------------------------
        # Input: Features
        # Output: Logits (Batch, Classes, Time)
        out_gen = self.gen(x)

        # ---------------------------------------------------------
        # 2. Refinement Stage 1
        # ---------------------------------------------------------
        # Input: Softmax Probabilities from Gen Stage
        # Detach is NOT used here to allow end-to-end training
        probs_gen = F.softmax(out_gen, dim=1)
        out_ref1 = self.ref1(probs_gen)

        # ---------------------------------------------------------
        # 3. Refinement Stage 2
        # ---------------------------------------------------------
        # Input: Softmax Probabilities from Ref1 Stage
        probs_ref1 = F.softmax(out_ref1, dim=1)
        out_ref2 = self.ref2(probs_ref1)

        return {"gen": out_gen, "ref1": out_ref1, "ref2": out_ref2}
