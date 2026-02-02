import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (Batch, Seq_Len, Dim)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class CharTransformer(nn.Module):
    """
    Transformer Encoder-Decoder for Character-Level Sequence to Sequence tasks.
    Conditioned on the token Class ID.
    """

    def __init__(self, vocab_chars_size, vocab_classes_size):
        """
        Args:
            vocab_chars_size (int): Size of the character vocabulary.
            vocab_classes_size (int): Size of the class vocabulary.
        """
        super(CharTransformer, self).__init__()
        self.config = Config()

        # Hyperparameters
        self.d_model = self.config.SEQ2SEQ_EMBED_DIM
        self.nhead = self.config.SEQ2SEQ_NUM_HEADS
        self.num_encoder_layers = self.config.SEQ2SEQ_NUM_LAYERS
        self.num_decoder_layers = self.config.SEQ2SEQ_NUM_LAYERS
        self.dim_feedforward = self.config.SEQ2SEQ_HIDDEN_DIM
        self.dropout = self.config.SEQ2SEQ_DROPOUT

        # Embeddings
        self.char_embedding = nn.Embedding(vocab_chars_size, self.d_model)
        self.class_embedding = nn.Embedding(vocab_classes_size, self.d_model)

        # Positional Encoding
        # Max len + buffer for safety
        self.pos_encoder = PositionalEncoding(
            self.d_model, self.dropout, max_len=self.config.MAX_TOKEN_CHAR_LEN + 10
        )

        # Transformer Backbone
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
        )

        # Output Head
        self.fc_out = nn.Linear(self.d_model, vocab_chars_size)

        # Padding ID (Assumed to be 0 based on data_processing.py)
        self.pad_idx = 0

    def generate_square_subsequent_mask(self, sz):
        """Generates a causal mask for the decoder."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, src_ids, tgt_in, class_id):
        """
        Forward pass for training.

        Args:
            src_ids (torch.Tensor): Source char IDs (Batch, Src_Len).
            tgt_in (torch.Tensor): Target char IDs input (Batch, Tgt_Len).
            class_id (torch.Tensor): Class IDs (Batch,).

        Returns:
            logits (torch.Tensor): (Batch, Tgt_Len, Vocab_Size).
        """
        device = src_ids.device
        batch_size = src_ids.size(0)

        # 1. Embed Source
        src_emb = self.char_embedding(src_ids)  # (B, S, D)

        # 2. Condition on Class: Prepend Class Embedding to Source
        cls_emb = self.class_embedding(class_id).unsqueeze(1)  # (B, 1, D)
        src_emb = torch.cat([cls_emb, src_emb], dim=1)  # (B, S+1, D)

        # 3. Embed Target
        tgt_emb = self.char_embedding(tgt_in)  # (B, T, D)

        # 4. Positional Encoding
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)

        # 5. Create Masks
        # Source Padding Mask: Account for prepended class token (never padding)
        src_pad_mask_orig = src_ids == self.pad_idx  # (B, S)
        cls_mask = torch.zeros((batch_size, 1), device=device, dtype=torch.bool)
        src_key_padding_mask = torch.cat(
            [cls_mask, src_pad_mask_orig], dim=1
        )  # (B, S+1)

        # Target Padding Mask
        tgt_key_padding_mask = tgt_in == self.pad_idx  # (B, T)

        # Causal Mask for Decoder
        tgt_mask = self.generate_square_subsequent_mask(tgt_in.size(1)).to(device)

        # 6. Transformer Forward
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # 7. Project to Vocab
        logits = self.fc_out(output)  # (B, T, V)

        return logits

    def predict(self, src_ids, class_id, max_len, sos_id, eos_id):
        """
        Greedy decoding for inference.

        Args:
            src_ids (torch.Tensor): Source char IDs (Batch, Src_Len).
            class_id (torch.Tensor): Class IDs (Batch,).
            max_len (int): Maximum generation length.
            sos_id (int): Start-of-Sequence token ID.
            eos_id (int): End-of-Sequence token ID.

        Returns:
            torch.Tensor: Generated token IDs (Batch, Gen_Len).
        """
        self.eval()
        device = src_ids.device
        batch_size = src_ids.size(0)

        # 1. Encode Source (Once)
        with torch.no_grad():
            src_emb = self.char_embedding(src_ids)
            cls_emb = self.class_embedding(class_id).unsqueeze(1)
            src_emb = torch.cat([cls_emb, src_emb], dim=1)
            src_emb = self.pos_encoder(src_emb)

            # Masking
            src_pad_mask_orig = src_ids == self.pad_idx
            cls_mask = torch.zeros((batch_size, 1), device=device, dtype=torch.bool)
            src_key_padding_mask = torch.cat([cls_mask, src_pad_mask_orig], dim=1)

            # Get Encoder Memory
            memory = self.transformer.encoder(
                src_emb, src_key_padding_mask=src_key_padding_mask
            )

            # 2. Decode Loop
            # Initialize with SOS
            tgt_ids = torch.full(
                (batch_size, 1), sos_id, dtype=torch.long, device=device
            )
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_len):
                tgt_emb = self.char_embedding(tgt_ids)
                tgt_emb = self.pos_encoder(tgt_emb)

                tgt_mask = self.generate_square_subsequent_mask(tgt_ids.size(1)).to(
                    device
                )

                # Decoder step
                output = self.transformer.decoder(
                    tgt=tgt_emb,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_key_padding_mask,
                )

                # Predict next token from last position
                last_output = output[:, -1, :]  # (B, D)
                logits = self.fc_out(last_output)  # (B, V)
                next_token = torch.argmax(logits, dim=-1)  # (B,)

                # Update finished status
                finished = finished | (next_token == eos_id)

                # Append prediction
                tgt_ids = torch.cat([tgt_ids, next_token.unsqueeze(1)], dim=1)

                if finished.all():
                    break

        return tgt_ids
