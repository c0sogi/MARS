import torch
import torch.nn as nn
from library.config import Config


class HybridLSTMTransformer(nn.Module):
    """
    A Hybrid architecture combining Bidirectional LSTMs for sequential dynamics
    and Transformer Encoders for global context, specifically designed for
    ventilator pressure prediction.
    """

    def __init__(self):
        super(HybridLSTMTransformer, self).__init__()

        # ==============================
        # Hyperparameters from Config
        # ==============================
        self.emb_dim = Config.EMBEDDING_DIM
        self.input_proj_dim = Config.INPUT_PROJ_DIM

        self.lstm_hidden = Config.LSTM_HIDDEN_DIM
        self.lstm_layers = Config.LSTM_LAYERS
        self.lstm_bidirectional = Config.LSTM_BIDIRECTIONAL

        self.transformer_heads = Config.TRANSFORMER_HEADS
        self.transformer_layers = Config.TRANSFORMER_LAYERS
        self.transformer_ff = Config.TRANSFORMER_FF_DIM

        self.dropout_p = Config.DROPOUT

        # Feature Dimensions
        self.num_cont = Config.NUM_CONT_FEATURES
        self.r_card = Config.R_CARDINALITY
        self.c_card = Config.C_CARDINALITY

        # ==============================
        # 1. Input Embeddings & Projection
        # ==============================
        # Learnable embeddings for lung attributes R and C
        self.r_embedding = nn.Embedding(self.r_card, self.emb_dim)
        self.c_embedding = nn.Embedding(self.c_card, self.emb_dim)

        # Projection for continuous features
        self.cont_proj = nn.Linear(self.num_cont, self.input_proj_dim)

        # Calculate combined dimension after concatenation
        # Dim = Proj(Cont) + Emb(R) + Emb(C)
        self.combined_dim = self.input_proj_dim + self.emb_dim + self.emb_dim

        self.ln_input = nn.LayerNorm(self.combined_dim)
        self.dropout = nn.Dropout(self.dropout_p)

        # ==============================
        # 2. Recurrent Block (Bi-LSTM)
        # ==============================
        self.lstm = nn.LSTM(
            input_size=self.combined_dim,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            dropout=self.dropout_p if self.lstm_layers > 1 else 0,
            bidirectional=self.lstm_bidirectional,
            batch_first=True,
        )

        # Calculate LSTM output dimension
        self.lstm_output_dim = (
            self.lstm_hidden * 2 if self.lstm_bidirectional else self.lstm_hidden
        )

        # Residual Connection for LSTM:
        # Since Input Dim != Output Dim, we need a projection for the skip connection
        self.lstm_residual_proj = nn.Linear(self.combined_dim, self.lstm_output_dim)
        self.ln_lstm = nn.LayerNorm(self.lstm_output_dim)

        # ==============================
        # 3. Transformer Block
        # ==============================
        # Encoder Layer with Multi-Head Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.lstm_output_dim,
            nhead=self.transformer_heads,
            dim_feedforward=self.transformer_ff,
            dropout=self.dropout_p,
            activation="gelu",
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.transformer_layers
        )

        # Residual Connection for Transformer:
        # Input Dim == Output Dim, so we can use identity addition, but we add Norm
        self.ln_transformer = nn.LayerNorm(self.lstm_output_dim)

        # ==============================
        # 4. Regression Head
        # ==============================
        self.head = nn.Sequential(
            nn.Linear(self.lstm_output_dim, 128), nn.GELU(), nn.Linear(128, 1)
        )

    def forward(self, cont_x, cat_x):
        """
        Forward pass of the model.

        Args:
            cont_x (torch.Tensor): Continuous features (Batch, Seq_Len, Num_Cont)
            cat_x (torch.Tensor): Categorical features (Batch, Seq_Len, 2) where
                                  cat_x[:,:,0] is R and cat_x[:,:,1] is C.

        Returns:
            torch.Tensor: Pressure predictions (Batch, Seq_Len)
        """
        # --- 1. Embeddings ---
        # Extract R and C indices
        r_idx = cat_x[:, :, 0]
        c_idx = cat_x[:, :, 1]

        r_emb = self.r_embedding(r_idx)
        c_emb = self.c_embedding(c_idx)

        # Project continuous features
        cont_emb = self.cont_proj(cont_x)

        # Concatenate all features
        x = torch.cat([cont_emb, r_emb, c_emb], dim=-1)
        x = self.ln_input(x)
        x = self.dropout(x)

        # --- 2. LSTM Block with Residual ---
        # Main Path
        lstm_out, _ = self.lstm(x)

        # Skip Connection (Projected input)
        shortcut_lstm = self.lstm_residual_proj(x)

        # Add & Norm
        x_lstm = lstm_out + shortcut_lstm
        x_lstm = self.ln_lstm(x_lstm)

        # --- 3. Transformer Block with Residual ---
        # Main Path
        # No mask needed as we want bidirectional attention over the full breath
        trans_out = self.transformer_encoder(x_lstm)

        # Skip Connection (Identity)
        x_trans = trans_out + x_lstm
        x_trans = self.ln_transformer(x_trans)

        # --- 4. Head ---
        out = self.head(x_trans)

        # Remove last dimension (Batch, Seq, 1) -> (Batch, Seq)
        return out.squeeze(-1)
