import torch
import torch.nn as nn
import timm
import math
import numpy as np
from library.config import Config


class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder wrapping timm models.
    Extracts patch embeddings to serve as memory for the decoder.
    """

    def __init__(self, model_name, pretrained=True):
        super(ViTEncoder, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.embed_dim = self.model.embed_dim

    def forward(self, x):
        """
        Args:
            x: Image tensor (B, C, H, W)
        Returns:
            features: Sequence of patch embeddings (B, N_patches + 1, E)
        """
        # forward_features returns the sequence of patch embeddings + CLS token
        features = self.model.forward_features(x)
        return features


class TransformerDecoderWrapper(nn.Module):
    """
    Standard Transformer Decoder for sequence generation.
    """

    def __init__(
        self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, dropout, max_len
    ):
        super(TransformerDecoderWrapper, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=Config.PAD_IDX)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None):
        """
        Args:
            tgt: Target sequence indices (B, L)
            memory: Encoder output (B, S, E)
            tgt_mask: Causal mask for autoregressive property
            tgt_padding_mask: Mask to ignore pad tokens
        """
        # Embed and add position encoding
        x = self.embedding(tgt)  # (B, L, E)
        x = self.pos_encoder(x)

        # Decode
        output = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        logits = self.fc_out(output)  # (B, L, Vocab)
        return logits


class ViT2InChI(nn.Module):
    """
    End-to-end model combining ViT Encoder and Transformer Decoder.
    """

    def __init__(self):
        super(ViT2InChI, self).__init__()

        # Encoder
        self.encoder = ViTEncoder(Config.ENCODER_MODEL_NAME, pretrained=True)

        # Projection Layer (if encoder/decoder dims mismatch)
        self.enc_dim = self.encoder.embed_dim
        self.dec_dim = Config.DECODER_EMBED_DIM

        if self.enc_dim != self.dec_dim:
            self.enc_project = nn.Linear(self.enc_dim, self.dec_dim)
        else:
            self.enc_project = nn.Identity()

        # Decoder
        self.decoder = TransformerDecoderWrapper(
            vocab_size=Config.VOCAB_SIZE,
            embed_dim=Config.DECODER_EMBED_DIM,
            num_layers=Config.DECODER_LAYERS,
            num_heads=Config.DECODER_HEADS,
            ff_dim=Config.DECODER_FF_DIM,
            dropout=Config.DROPOUT,
            max_len=Config.MAX_TEXT_LEN,
        )

    def generate_square_subsequent_mask(self, sz):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, images, targets=None):
        """
        Forward pass for training or feature extraction.

        Args:
            images: Input images (B, 3, H, W)
            targets: Target token indices (B, L). If provided, decoder runs with teacher forcing.

        Returns:
            logits (B, L, Vocab) if targets provided, else memory (B, S, E)
        """
        # 1. Encode images
        memory = self.encoder(images)  # (B, S, E_enc)
        memory = self.enc_project(memory)  # (B, S, E_dec)

        # 2. Decode (Training Mode)
        if targets is not None:
            tgt_len = targets.size(1)

            # Causal mask to prevent attending to future tokens
            tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(targets.device)

            # Padding mask to ignore PAD tokens in attention
            tgt_padding_mask = targets == Config.PAD_IDX

            logits = self.decoder(
                targets, memory, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask
            )
            return logits

        # Inference Mode (return memory for external loop)
        return memory

    def predict(self, images, max_len=None, device=Config.DEVICE):
        """
        Performs greedy decoding inference on a batch of images.

        Args:
            images: Input images (B, 3, H, W)
            max_len: Maximum generation length
            device: Torch device

        Returns:
            input_ids: Generated token indices (B, seq_len)
        """
        if max_len is None:
            max_len = Config.MAX_TEXT_LEN

        self.eval()
        batch_size = images.size(0)

        with torch.no_grad():
            # Encode
            memory = self.encoder(images)
            memory = self.enc_project(memory)

            # Initialize sequence with SOS token
            input_ids = torch.full(
                (batch_size, 1), Config.SOS_IDX, dtype=torch.long, device=device
            )

            # Track finished sequences (hit EOS)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_len):
                # Create mask for current sequence length
                tgt_mask = self.generate_square_subsequent_mask(input_ids.size(1)).to(
                    device
                )

                # Forward pass through decoder
                # Note: This is inefficient (re-computing past) but standard for basic TransformerDecoder usage
                logits = self.decoder(input_ids, memory, tgt_mask=tgt_mask)

                # Get logits for the last token
                next_token_logits = logits[:, -1, :]

                # Greedy selection
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Update finished state
                is_eos = next_token.squeeze(1) == Config.EOS_IDX
                finished = finished | is_eos

                if finished.all():
                    break

        return input_ids
