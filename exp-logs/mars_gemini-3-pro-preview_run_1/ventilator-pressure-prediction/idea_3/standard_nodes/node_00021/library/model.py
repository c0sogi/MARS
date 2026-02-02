import torch
import torch.nn as nn
import math
from library.config import Config


class MultiScaleCnnLSTM(nn.Module):
    """
    Hybrid Multi-Scale CNN-LSTM Architecture.

    Architecture:
    1. Input Processing: Embeddings for R/C + Continuous Features.
    2. Multi-Scale CNN Stem: Parallel 1D Convs with different kernel sizes.
    3. LSTM: Deep Bidirectional LSTM.
    4. Head: FC layers for regression.
    """

    def __init__(self):
        super(MultiScaleCnnLSTM, self).__init__()

        # --- 1. Input & Embeddings ---
        self.r_emb = nn.Embedding(3, Config.R_EMBED_DIM)
        self.c_emb = nn.Embedding(3, Config.C_EMBED_DIM)

        # Calculate input dimension
        self.input_dim = (
            len(Config.CONT_FEATURES) + Config.R_EMBED_DIM + Config.C_EMBED_DIM
        )

        # --- 2. Multi-Scale CNN Stem ---
        # We use ModuleList for parallel branches
        self.cnn_branches = nn.ModuleList()

        for k in Config.CNN_KERNELS:
            padding = (k - 1) // 2
            self.cnn_branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=self.input_dim,
                        out_channels=Config.CNN_FILTERS,
                        kernel_size=k,
                        padding=padding,
                    ),
                    nn.GELU(),
                    nn.BatchNorm1d(Config.CNN_FILTERS),
                )
            )

        # Output dim of CNN stem is sum of filters from all branches
        self.cnn_out_dim = Config.CNN_FILTERS * len(Config.CNN_KERNELS)

        # --- 3. Residual Bi-LSTM ---
        self.lstm_layers = nn.ModuleList()

        # First LSTM layer: CNN Out Dim -> LSTM Hidden
        self.lstm_layers.append(
            nn.LSTM(
                input_size=self.cnn_out_dim,
                hidden_size=Config.LSTM_HIDDEN,
                batch_first=True,
                bidirectional=Config.LSTM_BIDIRECTIONAL,
            )
        )

        # Subsequent LSTM layers
        lstm_input_dim = (
            Config.LSTM_HIDDEN * 2 if Config.LSTM_BIDIRECTIONAL else Config.LSTM_HIDDEN
        )

        for _ in range(Config.LSTM_LAYERS - 1):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=lstm_input_dim,
                    hidden_size=Config.LSTM_HIDDEN,
                    batch_first=True,
                    bidirectional=Config.LSTM_BIDIRECTIONAL,
                )
            )

        self.lstm_dropout = nn.Dropout(Config.LSTM_DROPOUT)

        # --- 4. Head ---
        head_input_dim = (
            Config.LSTM_HIDDEN * 2 if Config.LSTM_BIDIRECTIONAL else Config.LSTM_HIDDEN
        )

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, Config.FC_HIDDEN),
            nn.GELU(),
            nn.Linear(Config.FC_HIDDEN, 1),
        )

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Num_Features]

        # 1. Feature Separation
        cont_feats = x[:, :, :-2]
        r_idx = x[:, :, -2].long()
        c_idx = x[:, :, -1].long()

        # 2. Embeddings
        r_emb = self.r_emb(r_idx)
        c_emb = self.c_emb(c_idx)

        # Concatenate
        x_cat = torch.cat(
            [cont_feats, r_emb, c_emb], dim=2
        )  # [Batch, Seq_Len, Input_Dim]

        # 3. Multi-Scale CNN Stem
        # Permute for Conv1d: [Batch, Channels, Seq_Len]
        x_cat = x_cat.permute(0, 2, 1)

        cnn_outputs = []
        for branch in self.cnn_branches:
            cnn_outputs.append(branch(x_cat))

        # Concatenate along channel dimension
        x_cnn = torch.cat(cnn_outputs, dim=1)

        # Permute back: [Batch, Seq_Len, Channels]
        x_cnn = x_cnn.permute(0, 2, 1)

        # 4. Residual LSTM
        x_curr = x_cnn

        for i, lstm in enumerate(self.lstm_layers):
            lstm_out, _ = lstm(x_curr)
            lstm_out = self.lstm_dropout(lstm_out)

            # Residual connection
            if x_curr.size(-1) == lstm_out.size(-1):
                x_curr = x_curr + lstm_out
            else:
                x_curr = lstm_out

        # 5. Head
        out = self.head(x_curr)
        return out.squeeze(-1)
