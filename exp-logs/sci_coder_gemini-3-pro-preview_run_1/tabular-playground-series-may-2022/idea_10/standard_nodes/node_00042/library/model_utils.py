import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class GEGLU(nn.Module):
    """
    Gated Linear Unit with GELU activation.
    Splits the input projection into two parts: a gate and a value.
    Output is (x * W_gate + b_gate).gelu() * (x * W_val + b_val).
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class GranularInputLayer(nn.Module):
    """
    Handles embedding of mixed numerical and sequence data.
    - Numerical features: Projected via Linear Feature Tokenization (x*w + b).
    - Sequence features: Mapped via Entity Embeddings.
    - Adds learnable positional encodings and handles [CLS] token.
    - Implements stochastic masking for the denoising objective.
    """

    def __init__(self, num_numerical, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.num_numerical = num_numerical

        # Linear Feature Tokenization for Numerical Features
        # We learn a separate weight and bias vector for each numerical feature
        # Shape: (num_numerical, d_model)
        self.num_proj_w = nn.Parameter(torch.randn(num_numerical, d_model) * 0.02)
        self.num_proj_b = nn.Parameter(torch.zeros(num_numerical, d_model))

        # Entity Embeddings for Sequence Features (f_27 characters)
        self.seq_emb = nn.Embedding(vocab_size, d_model)

        # Special Tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional Embedding
        # Covers [CLS] + Num_Feats + Seq_Len
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

    def forward(self, x_num, x_seq, mask_ratio=0.0):
        """
        Args:
            x_num: (B, N_num) standardized numerical features
            x_seq: (B, N_seq) integer encoded sequence features
            mask_ratio: Float, probability of masking a token (0.0 during inference)
        Returns:
            x: (B, T, D) Combined embedded sequence
            mask_bool: (B, T) Boolean mask indicating which tokens were masked (True=Masked)
        """
        B = x_num.shape[0]

        # 1. Embed Numerical: (B, N_num) -> (B, N_num, D)
        # x * w + b broadcasted over batch
        # x_num.unsqueeze(-1): (B, N_num, 1)
        # self.num_proj_w: (N_num, D)
        x_num_emb = x_num.unsqueeze(-1) * self.num_proj_w + self.num_proj_b

        # 2. Embed Sequence: (B, N_seq) -> (B, N_seq, D)
        x_seq_emb = self.seq_emb(x_seq)

        # 3. Concatenate: [CLS] + Num + Seq
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x_num_emb, x_seq_emb), dim=1)

        # 4. Add Positional Embeddings
        seq_len = x.shape[1]
        x = x + self.pos_emb[:, :seq_len, :]

        # 5. Apply Masking (if training/denoising)
        mask_bool = None
        if mask_ratio > 0.0:
            # Create mask for all tokens EXCEPT [CLS]
            # Probabilities
            probs = torch.full((B, seq_len - 1), mask_ratio, device=x.device)
            mask_indices = torch.bernoulli(probs).bool()

            # Prepend False for CLS token (never mask CLS)
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
            mask_bool = torch.cat((cls_mask, mask_indices), dim=1)

            # Replace masked tokens with [MASK] embedding
            mask_emb = self.mask_token.expand(B, seq_len, -1)
            x = torch.where(mask_bool.unsqueeze(-1), mask_emb, x)

        return x, mask_bool


class TransformerEncoderLayerGEGLU(nn.Module):
    """
    Standard Transformer Encoder Layer but with GEGLU Feed-Forward.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        # GEGLU block: Linear -> GEGLU -> Linear
        # Note: GEGLU internal projection is d_model -> d_ff * 2
        self.geglu = GEGLU(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Self Attention
        res = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x)
        x = res + self.dropout1(x)

        # Feed Forward with GEGLU
        res = x
        x = self.ln2(x)
        x = self.geglu(x)
        x = self.fc2(x)
        x = res + self.dropout2(x)
        return x


class SSDeGUT(nn.Module):
    """
    Semi-Supervised Denoising Granular Unified Transformer.
    """

    def __init__(self, config: Config, num_numerical_features: int):
        super().__init__()
        self.config = config

        # Input Layer
        self.input_layer = GranularInputLayer(
            num_numerical=num_numerical_features,
            vocab_size=config.VOCAB_SIZE,
            d_model=config.D_MODEL,
            max_seq_len=config.MAX_SEQ_LEN,
        )

        # Encoder Layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerGEGLU(
                    d_model=config.D_MODEL,
                    n_heads=config.N_HEADS,
                    d_ff=config.D_FF // 2,  # Adjust for GEGLU 2x expansion
                    dropout=config.DROPOUT,
                )
                for _ in range(config.N_LAYERS)
            ]
        )

        # Heads
        self.ln_f = nn.LayerNorm(config.D_MODEL)

        # 1. Classification Head (Target)
        self.head_cls = nn.Linear(config.D_MODEL, 1)

        # 2. Reconstruction Heads (Denoising)
        # Reconstruct numerical values
        self.head_recon_num = nn.Linear(config.D_MODEL, 1)
        # Reconstruct sequence characters (classification over vocab)
        self.head_recon_seq = nn.Linear(config.D_MODEL, config.VOCAB_SIZE)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, std=0.02)

    def forward(self, x_num, x_seq, mask_ratio=0.0):
        """
        Args:
            x_num: (B, N_num)
            x_seq: (B, N_seq)
            mask_ratio: Probability of masking tokens.
        Returns:
            dict containing:
                - 'logits': Classification logits (B, 1)
                - 'recon_num': Reconstructed numerical values (B, N_num)
                - 'recon_seq': Reconstructed sequence logits (B, N_seq, Vocab)
                - 'mask': Boolean mask used (B, T) or None
        """
        # Embed and Mask
        x, mask = self.input_layer(x_num, x_seq, mask_ratio=mask_ratio)

        # Transformer Encoder
        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)

        # Extract components
        # x is [CLS, Num_1, ..., Num_N, Seq_1, ..., Seq_M]
        cls_token = x[:, 0, :]
        num_tokens = x[:, 1 : 1 + self.input_layer.num_numerical, :]
        seq_tokens = x[:, 1 + self.input_layer.num_numerical :, :]

        # Heads
        logits = self.head_cls(cls_token)
        recon_num = self.head_recon_num(num_tokens).squeeze(-1)
        recon_seq = self.head_recon_seq(seq_tokens)

        return {
            "logits": logits,
            "recon_num": recon_num,
            "recon_seq": recon_seq,
            "mask": mask,
        }

    def compute_loss(self, batch, device):
        """
        Computes the composite loss:
        L = BCE(Labeled) + lambda * (MSE(Recon_Num) + CE(Recon_Seq))
        """
        x_num = batch["x_num"].to(device)
        x_seq = batch["x_seq"].to(device)
        targets = batch.get("target")  # May be None for unlabeled data

        # Forward pass with masking
        outputs = self.forward(x_num, x_seq, mask_ratio=self.config.MASK_RATIO)

        mask = outputs["mask"]  # (B, T)
        # T = 1 + N_num + N_seq
        # Split mask
        num_feats = self.input_layer.num_numerical
        mask_num = mask[:, 1 : 1 + num_feats]
        mask_seq = mask[:, 1 + num_feats :]

        total_loss = 0.0
        metrics = {}

        # 1. Classification Loss (Only on labeled data)
        if targets is not None:
            # Filter rows where target is not NaN (if mixing labeled/unlabeled in one tensor)
            # Assuming batch['target'] contains -1 or NaN for unlabeled if mixed.
            # But usually we might pass separate batches.
            # Here we assume standard supervised batch or mixed.
            # If targets are present, compute BCE.
            targets = targets.to(device).float()
            # Check for valid targets (not NaN)
            valid_idx = ~torch.isnan(targets)
            if valid_idx.sum() > 0:
                y_true = targets[valid_idx].unsqueeze(-1)
                y_pred = outputs["logits"][valid_idx]

                # Label Smoothing
                loss_cls = F.binary_cross_entropy_with_logits(
                    y_pred,
                    y_true * (1 - self.config.LABEL_SMOOTHING)
                    + 0.5 * self.config.LABEL_SMOOTHING,
                )
                total_loss += loss_cls
                metrics["loss_cls"] = loss_cls.item()

        # 2. Reconstruction Loss (On MASKED tokens only)
        # Numerical Reconstruction (MSE)
        if mask_num.sum() > 0:
            pred_num = outputs["recon_num"][mask_num]
            true_num = x_num[mask_num]
            loss_recon_num = F.mse_loss(pred_num, true_num)
            total_loss += self.config.RECON_LOSS_WEIGHT * loss_recon_num
            metrics["loss_recon_num"] = loss_recon_num.item()

        # Sequence Reconstruction (Cross Entropy)
        if mask_seq.sum() > 0:
            pred_seq = outputs["recon_seq"][mask_seq]  # (N_masked, Vocab)
            true_seq = x_seq[mask_seq]  # (N_masked)
            loss_recon_seq = F.cross_entropy(pred_seq, true_seq)
            total_loss += self.config.RECON_LOSS_WEIGHT * loss_recon_seq
            metrics["loss_recon_seq"] = loss_recon_seq.item()

        metrics["loss_total"] = (
            total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        )

        return total_loss, metrics
