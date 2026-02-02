import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math

from library.config import Config
from library.modules import AdaLN, AdaLNDecoderLayer


class AttributeHead(nn.Module):
    """
    MLP head for regressing molecular attributes from global image features.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class AMViT(nn.Module):
    """
    Attribute-Modulated Visual Transformer (AM-ViT).

    Architecture:
    1. EfficientNet-B0 Encoder -> Spatial Features + Global Features
    2. Attribute Head -> Regresses chemical attributes (C, H, O, etc.)
    3. Transformer Decoder -> Generates InChI sequence
       - Uses Adaptive Layer Norm (AdaLN) conditioned on predicted attributes.
    """

    def __init__(self, vocab_size, pad_idx=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        # ----------------------------------------------------------------------
        # 1. Image Encoder (EfficientNet-B0)
        # ----------------------------------------------------------------------
        # We load the model pretrained on ImageNet.
        # forward_features returns (B, 1280, H, W) for B0
        self.encoder = timm.create_model(Config.ENCODER_NAME, pretrained=True)

        # ----------------------------------------------------------------------
        # 2. Attribute Branch
        # ----------------------------------------------------------------------
        self.attribute_head = AttributeHead(
            input_dim=Config.ENCODER_DIM, output_dim=Config.NUM_ATTRIBUTES
        )

        # ----------------------------------------------------------------------
        # 3. Sequence Decoder (Transformer with AdaLN)
        # ----------------------------------------------------------------------
        self.embedding = nn.Embedding(vocab_size, Config.EMBED_DIM, padding_idx=pad_idx)

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, Config.MAX_LEN, Config.EMBED_DIM) * 0.02
        )

        self.dropout = nn.Dropout(Config.DROPOUT)

        # Decoder Layers
        self.decoder_layers = nn.ModuleList(
            [
                AdaLNDecoderLayer(
                    embed_dim=Config.EMBED_DIM,
                    cond_dim=Config.NUM_ATTRIBUTES,
                    num_heads=Config.DECODER_HEADS,
                    ff_dim=Config.DECODER_FF_DIM,
                    dropout=Config.DROPOUT,
                    encoder_dim=Config.ENCODER_DIM,
                )
                for _ in range(Config.DECODER_LAYERS)
            ]
        )

        # Final Normalization (also adaptive to maintain conditioning context)
        self.final_norm = AdaLN(Config.EMBED_DIM, Config.NUM_ATTRIBUTES)

        # Output Projection
        self.output_proj = nn.Linear(Config.EMBED_DIM, vocab_size)

        # Weight initialization for projection
        nn.init.normal_(self.output_proj.weight, mean=0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, images, text_seq):
        """
        Forward pass for training.

        Args:
            images (torch.Tensor): (B, 3, H, W)
            text_seq (torch.Tensor): (B, SeqLen) - Input sequence (e.g., <SOS> ... )

        Returns:
            logits (torch.Tensor): (B, SeqLen, VocabSize)
            pred_attrs (torch.Tensor): (B, NumAttributes)
        """
        batch_size = images.size(0)
        seq_len = text_seq.size(1)

        # --- 1. Encode Image ---
        # features: (B, 1280, 8, 8) for 256x256 input
        features = self.encoder.forward_features(images)

        # Global features for attribute head: (B, 1280)
        # Adaptive pooling ensures shape regardless of input size
        global_features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)

        # Spatial features for cross-attention: (B, 64, 1280)
        # Permute to (B, H*W, C)
        spatial_features = features.permute(0, 2, 3, 1).flatten(1, 2)

        # --- 2. Predict Attributes ---
        pred_attrs = self.attribute_head(global_features)

        # --- 3. Prepare Decoder Inputs ---
        # Embeddings: (B, L, D)
        x = self.embedding(text_seq)

        # Add Positional Encoding (broadcast over batch)
        # We slice pos_encoding to the current sequence length
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)

        # Masks
        # Causal Mask: (L, L) - prevents attending to future tokens
        tgt_mask = self._generate_square_subsequent_mask(seq_len).to(images.device)

        # Padding Mask: (B, L) - True where padding exists
        tgt_key_padding_mask = text_seq == self.pad_idx

        # --- 4. Decode with AdaLN ---
        for layer in self.decoder_layers:
            x = layer(
                x,
                enc_out=spatial_features,
                condition=pred_attrs,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )

        # Final Norm
        x = self.final_norm(x, pred_attrs)

        # Projection
        logits = self.output_proj(x)

        return logits, pred_attrs

    def generate(self, images, max_len=None, sos_idx=1, eos_idx=2):
        """
        Greedy decoding for inference.

        Args:
            images (torch.Tensor): (B, 3, H, W)
            max_len (int): Maximum generation length.
            sos_idx (int): Start of Sequence token ID.
            eos_idx (int): End of Sequence token ID.

        Returns:
            predictions (torch.Tensor): (B, L)
        """
        if max_len is None:
            max_len = Config.MAX_LEN

        batch_size = images.size(0)
        device = images.device

        # --- Encode ---
        features = self.encoder.forward_features(images)
        global_features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        spatial_features = features.permute(0, 2, 3, 1).flatten(1, 2)

        # --- Predict Attributes ---
        pred_attrs = self.attribute_head(global_features)

        # --- Greedy Decoding ---
        # Initialize sequence with SOS
        generated_seq = torch.full(
            (batch_size, 1), sos_idx, dtype=torch.long, device=device
        )

        # Keep track of finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            seq_len = generated_seq.size(1)

            # Embed
            x = self.embedding(generated_seq)
            x = x + self.pos_encoding[:, :seq_len, :]

            # Causal Mask
            tgt_mask = self._generate_square_subsequent_mask(seq_len).to(device)

            # Pass through layers
            # Note: We re-process the whole sequence each time (no KV caching in this impl)
            for layer in self.decoder_layers:
                x = layer(
                    x, enc_out=spatial_features, condition=pred_attrs, tgt_mask=tgt_mask
                )

            x = self.final_norm(x, pred_attrs)
            logits = self.output_proj(x)

            # Get last token predictions
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)

            # Update finished status
            finished |= next_token == eos_idx

            # Append to sequence
            generated_seq = torch.cat([generated_seq, next_token.unsqueeze(1)], dim=1)

            # Stop if all finished
            if finished.all():
                break

        return generated_seq

    def _generate_square_subsequent_mask(self, sz):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
