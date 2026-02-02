import torch
import torch.nn as nn
import math
from library.config import Config


class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Dim]
        # pe shape: [1, Max_Len, Dim] -> slice to [1, Seq_Len, Dim]
        x = x + self.pe[:, : x.size(1), :]
        return x


class TransLSTM(nn.Module):
    """
    Hybrid Transformer-LSTM Architecture for Ventilator Pressure Prediction.

    Architecture:
    1. Input Processing: Embeddings for R/C + Continuous Features.
    2. CNN Stem: 1D Conv for local feature extraction and projection.
    3. Transformer: Encoder to capture global context (entire breath).
    4. LSTM: Bidirectional LSTM with residual connections for sequential dynamics.
    5. Head: FC layers for final regression.
    """

    def __init__(self):
        super(TransLSTM, self).__init__()

        # --- 1. Input & Embeddings ---
        self.r_emb = nn.Embedding(3, Config.R_EMBED_DIM)
        self.c_emb = nn.Embedding(3, Config.C_EMBED_DIM)

        # Calculate input dimension after concatenation
        # Input features include continuous features + 2 embeddings
        self.input_dim = (
            len(Config.CONT_FEATURES) + Config.R_EMBED_DIM + Config.C_EMBED_DIM
        )

        # --- 2. CNN Stem ---
        # Projects input_dim to TRANS_D_MODEL
        # Kernel size preserves sequence length if padding is handled,
        # but here we use padding='same' logic manually or via PyTorch defaults
        padding = (Config.CNN_KERNEL_SIZE - 1) // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=self.input_dim,
                out_channels=Config.TRANS_D_MODEL,
                kernel_size=Config.CNN_KERNEL_SIZE,
                padding=padding,
            ),
            nn.GELU(),
            nn.BatchNorm1d(Config.TRANS_D_MODEL),
        )

        # --- 3. Transformer Encoder ---
        self.pos_encoder = PositionalEncoding(
            Config.TRANS_D_MODEL, max_len=Config.SEQ_LEN
        )

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=Config.TRANS_D_MODEL,
            nhead=Config.TRANS_NHEAD,
            dim_feedforward=Config.TRANS_DIM_FEEDFORWARD,
            dropout=Config.TRANS_DROPOUT,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=Config.TRANS_LAYERS
        )

        # --- 4. Residual Bi-LSTM ---
        # We implement layers manually to allow for residual connections
        self.lstm_layers = nn.ModuleList()

        # First LSTM layer: Transformer Dim -> LSTM Hidden
        self.lstm_layers.append(
            nn.LSTM(
                input_size=Config.TRANS_D_MODEL,
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

        # --- 5. Head ---
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
        # The last two columns are R_cat and C_cat

        # 1. Feature Separation
        cont_feats = x[:, :, :-2]
        r_idx = x[:, :, -2].long()
        c_idx = x[:, :, -1].long()

        # 2. Embeddings
        r_emb = self.r_emb(r_idx)  # [Batch, Seq_Len, R_Dim]
        c_emb = self.c_emb(c_idx)  # [Batch, Seq_Len, C_Dim]

        # Concatenate
        x_cat = torch.cat(
            [cont_feats, r_emb, c_emb], dim=2
        )  # [Batch, Seq_Len, Input_Dim]

        # 3. CNN Stem
        # Conv1d expects [Batch, Channels, Seq_Len]
        x_cat = x_cat.permute(0, 2, 1)
        x_cnn = self.cnn(x_cat)
        # Permute back to [Batch, Seq_Len, Channels] for Transformer
        x_cnn = x_cnn.permute(0, 2, 1)

        # 4. Transformer Encoder
        x_pos = self.pos_encoder(x_cnn)
        x_trans = self.transformer_encoder(x_pos)

        # 5. Residual LSTM
        x_curr = x_trans

        for i, lstm in enumerate(self.lstm_layers):
            # LSTM returns (output, (h_n, c_n))
            lstm_out, _ = lstm(x_curr)
            lstm_out = self.lstm_dropout(lstm_out)

            # Apply residual if dimensions match
            if x_curr.size(-1) == lstm_out.size(-1):
                x_curr = x_curr + lstm_out
            else:
                x_curr = lstm_out

        # 6. Head
        # x_curr shape: [Batch, Seq_Len, LSTM_Out_Dim]
        out = self.head(x_curr)  # [Batch, Seq_Len, 1]

        # Squeeze the last dimension to match target shape [Batch, Seq_Len]
        return out.squeeze(-1)
