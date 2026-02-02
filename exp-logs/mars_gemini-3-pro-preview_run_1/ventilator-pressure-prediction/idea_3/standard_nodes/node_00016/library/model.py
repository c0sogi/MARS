import torch
import torch.nn as nn
import math
from library.config import Config


class MultiScaleCnnLSTM(nn.Module):
    """
    Hybrid Multi-Scale CNN-LSTM Architecture (Cite solution_lesson_node_00015).
    Replaces Transformer with Multi-Scale Convolutional Stem.
    """

    def __init__(self):
        super(MultiScaleCnnLSTM, self).__init__()

        self.input_dim = len(Config.CONT_FEATURES)

        # --- Multi-Scale CNN Stem ---
        # Parallel convolutions with kernel sizes 3, 5, 7
        self.conv3 = nn.Conv1d(
            self.input_dim, Config.CNN_FILTERS, kernel_size=3, padding=1
        )
        self.conv5 = nn.Conv1d(
            self.input_dim, Config.CNN_FILTERS, kernel_size=5, padding=2
        )
        self.conv7 = nn.Conv1d(
            self.input_dim, Config.CNN_FILTERS, kernel_size=7, padding=3
        )

        self.bn = nn.BatchNorm1d(Config.CNN_FILTERS * 3)
        self.act = nn.GELU()

        # --- Residual Bi-LSTM ---
        self.lstm_layers = nn.ModuleList()

        # Input to LSTM is the concatenated output of 3 convs
        lstm_input_dim = Config.CNN_FILTERS * 3
        lstm_hidden = Config.LSTM_HIDDEN

        # First LSTM Layer
        self.lstm_layers.append(
            nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=lstm_hidden,
                batch_first=True,
                bidirectional=Config.LSTM_BIDIRECTIONAL,
            )
        )

        # Subsequent Layers
        lstm_output_dim = lstm_hidden * 2 if Config.LSTM_BIDIRECTIONAL else lstm_hidden

        for _ in range(Config.LSTM_LAYERS - 1):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=lstm_output_dim,
                    hidden_size=lstm_hidden,
                    batch_first=True,
                    bidirectional=Config.LSTM_BIDIRECTIONAL,
                )
            )

        self.lstm_dropout = nn.Dropout(Config.LSTM_DROPOUT)

        # --- Head ---
        self.head = nn.Sequential(
            nn.Linear(lstm_output_dim, Config.FC_HIDDEN),
            nn.GELU(),
            nn.Linear(Config.FC_HIDDEN, 1),
        )

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Features]

        # Permute for CNN: [Batch, Features, Seq_Len]
        x = x.permute(0, 2, 1)

        # Multi-Scale Convolutions
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        c7 = self.conv7(x)

        # Concatenate: [Batch, Filters*3, Seq_Len]
        x_cnn = torch.cat([c3, c5, c7], dim=1)
        x_cnn = self.bn(x_cnn)
        x_cnn = self.act(x_cnn)

        # Permute back for LSTM: [Batch, Seq_Len, Filters*3]
        x_curr = x_cnn.permute(0, 2, 1)

        # Residual LSTM
        for lstm in self.lstm_layers:
            lstm_out, _ = lstm(x_curr)
            lstm_out = self.lstm_dropout(lstm_out)

            # Residual connection
            if x_curr.size(-1) == lstm_out.size(-1):
                x_curr = x_curr + lstm_out
            else:
                x_curr = lstm_out

        # Head
        out = self.head(x_curr)
        return out.squeeze(-1)
