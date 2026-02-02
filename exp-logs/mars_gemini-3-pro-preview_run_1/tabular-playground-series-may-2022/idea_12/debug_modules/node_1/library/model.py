import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class GEGLU(nn.Module):
    """
    GEGLU activation function: x * GELU(gate)
    Splits the input tensor into two halves along the last dimension.
    """

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class TransformerEncoderLayer(nn.Module):
    """
    Custom Transformer Encoder Layer using GEGLU in the Feed-Forward Network.
    Follows the standard Post-Norm architecture.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Feed-forward network with GEGLU
        # Project to 2 * dim_feedforward to allow splitting in GEGLU
        self.linear1 = nn.Linear(d_model, dim_feedforward * 2)
        self.act = GEGLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self Attention block
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = self.norm1(src + self.dropout(src2))

        # Feed Forward block
        src2 = self.linear2(self.act(self.linear1(src)))
        src = self.norm2(src + self.dropout(src2))
        return src


class DSDN(nn.Module):
    """
    Dual-Stream Denoising Network (DSDN).
    Combines a Deep Transformer stream with masking/reconstruction objectives and
    a Wide MLP stream for positional rigidity.
    """

    def __init__(self, num_features, vocab_size, seq_len):
        super().__init__()

        self.d_model = Config.EMBED_DIM
        self.num_features = num_features
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        # ---------------------------------------------------------------------
        # Embeddings & Tokenization
        # ---------------------------------------------------------------------

        # Linear Feature Tokenizer for Numerical Features
        # Implemented as learnable weights/biases for element-wise projection
        # Shape: (Num_Features, Embed_Dim)
        self.num_w = nn.Parameter(torch.empty(num_features, self.d_model))
        self.num_b = nn.Parameter(torch.empty(num_features, self.d_model))

        # Sequence Embedding
        self.seq_emb = nn.Embedding(vocab_size, self.d_model)

        # Special Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        # Positional Embedding
        # Length = 1 (CLS) + Num_Features + Seq_Len
        total_len = 1 + num_features + seq_len
        self.pos_emb = nn.Parameter(torch.zeros(1, total_len, self.d_model))

        # ---------------------------------------------------------------------
        # Stream 1: Deep Denoising Expert (Transformer)
        # ---------------------------------------------------------------------
        self.transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=Config.NUM_HEADS,
                    dim_feedforward=Config.DIM_FEEDFORWARD,
                    dropout=Config.DROPOUT,
                )
                for _ in range(Config.NUM_TRANSFORMER_LAYERS)
            ]
        )

        # Auxiliary Reconstruction Heads
        self.num_recon_head = nn.Linear(self.d_model, 1)
        self.seq_recon_head = nn.Linear(self.d_model, vocab_size)

        # ---------------------------------------------------------------------
        # Stream 2: Wide Anchoring Expert (MLP)
        # ---------------------------------------------------------------------
        # Input: Raw Num (num_features) + Flattened Seq Emb (seq_len * d_model)
        wide_input_dim = num_features + (seq_len * self.d_model)

        # Projection to hidden dim
        self.wide_proj = nn.Linear(wide_input_dim, Config.MLP_HIDDEN_DIM)

        # Residual Block: Linear -> BN -> GELU -> Linear
        self.wide_res_block = nn.Sequential(
            nn.Linear(Config.MLP_HIDDEN_DIM, Config.MLP_HIDDEN_DIM),
            nn.BatchNorm1d(Config.MLP_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(Config.MLP_HIDDEN_DIM, Config.MLP_HIDDEN_DIM),
        )

        # ---------------------------------------------------------------------
        # Fusion & Classification
        # ---------------------------------------------------------------------
        # Input: CLS (d_model) + Wide (hidden_dim)
        fusion_dim = self.d_model + Config.MLP_HIDDEN_DIM

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(fusion_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize tokenizers and embeddings
        nn.init.kaiming_normal_(self.num_w)
        nn.init.zeros_(self.num_b)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)

        # Initialize Transformer weights
        for p in self.transformer_layers.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Initialize Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Avoid re-initializing if already done (e.g. inside Transformer)
                # But explicit initialization for heads is good
                pass

    def forward(self, x_num, x_seq):
        """
        Args:
            x_num: (Batch, Num_Features) - Standardized numerical features
            x_seq: (Batch, Seq_Len) - Integer encoded sequence features

        Returns:
            dict containing logits, reconstruction predictions, and mask info.
        """
        batch_size = x_num.shape[0]
        device = x_num.device

        # -----------------------------------------------------------------
        # 1. Feature Tokenization
        # -----------------------------------------------------------------

        # Numerical: (B, N) -> (B, N, D)
        # Element-wise linear projection: x * w + b
        num_tokens = x_num.unsqueeze(-1) * self.num_w + self.num_b

        # Sequence: (B, L) -> (B, L, D)
        seq_tokens = self.seq_emb(x_seq)

        # Concatenate features for Transformer: [Num_Tokens, Seq_Tokens]
        # Shape: (B, N+L, D)
        features = torch.cat([num_tokens, seq_tokens], dim=1)

        # -----------------------------------------------------------------
        # 2. Stream 2: Wide Branch (Uses Clean Data)
        # -----------------------------------------------------------------
        # Flatten sequence embeddings: (B, L, D) -> (B, L*D)
        seq_flat = seq_tokens.view(batch_size, -1)

        # Concatenate raw numerical and flattened sequence
        wide_in = torch.cat([x_num, seq_flat], dim=1)

        # Project and Apply Residual Block
        wide_hidden = self.wide_proj(wide_in)
        wide_out = wide_hidden + self.wide_res_block(wide_hidden)

        # -----------------------------------------------------------------
        # 3. Stream 1: Deep Branch (Masking & Transformer)
        # -----------------------------------------------------------------

        mask_indices = None

        if self.training:
            # Create mask probability matrix: (B, N+L)
            prob_matrix = torch.full(
                features.shape[:2], Config.MASK_PROB, device=device
            )
            mask_indices = torch.bernoulli(prob_matrix).bool()

            # Replace masked positions with mask_token
            features_masked = features.clone()
            features_masked[mask_indices] = self.mask_token.view(1, -1).expand_as(
                features[mask_indices]
            )
        else:
            features_masked = features

        # Prepend CLS token: (B, 1, D)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Input to transformer: [CLS, Num, Seq]
        transformer_input = torch.cat([cls_tokens, features_masked], dim=1)

        # Add Positional Embeddings
        transformer_input = transformer_input + self.pos_emb

        # Pass through Transformer Layers
        x = transformer_input
        for layer in self.transformer_layers:
            x = layer(x)

        # Extract CLS output: (B, D)
        cls_out = x[:, 0, :]

        # -----------------------------------------------------------------
        # 4. Reconstruction Heads (Auxiliary)
        # -----------------------------------------------------------------
        # Extract features (excluding CLS)
        feature_out = x[:, 1:, :]

        # Split back into Num and Seq parts for specific heads
        n_num = num_tokens.shape[1]
        num_out = feature_out[:, :n_num, :]
        seq_out = feature_out[:, n_num:, :]

        # Predict original values
        num_pred = self.num_recon_head(num_out).squeeze(-1)  # (B, N)
        seq_pred = self.seq_recon_head(seq_out)  # (B, L, Vocab)

        # -----------------------------------------------------------------
        # 5. Fusion & Output
        # -----------------------------------------------------------------
        fusion_in = torch.cat([cls_out, wide_out], dim=1)
        logits = self.classifier(fusion_in)

        return {
            "logits": logits,
            "num_pred": num_pred,
            "seq_pred": seq_pred,
            "mask_indices": mask_indices,  # (B, N+L) boolean matrix
            "num_orig": x_num,
            "seq_orig": x_seq,
        }
