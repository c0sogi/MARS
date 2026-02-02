import torch
import torch.nn as nn
import math
from library.config import Config


class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding.
    Injects information about the relative or absolute position of the tokens in the sequence.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim]
               depending on how it's sliced. Here we assume (Batch, Seq, Dim) due to batch_first=True logic
               but PE is usually (Seq, 1, Dim).
        """
        # x is (Batch, Seq_Len, Dim)
        x = x + self.pe[: x.size(1)].transpose(0, 1)
        return self.dropout(x)


class CharTransformer(nn.Module):
    """
    Character-level Transformer Seq2Seq model.
    Encodes raw character sequence + class token -> Decodes normalized character sequence.
    """

    def __init__(
        self,
        num_chars: int,
        num_classes: int,
        d_model: int = Config.SEQ2SEQ_EMBEDDING_DIM,
        nhead: int = Config.SEQ2SEQ_NUM_HEADS,
        num_layers: int = Config.SEQ2SEQ_NUM_LAYERS,
        dim_feedforward: int = Config.SEQ2SEQ_HIDDEN_DIM,
        dropout: float = Config.SEQ2SEQ_DROPOUT,
    ):
        super(CharTransformer, self).__init__()
        self.d_model = d_model
        self.model_type = "Transformer"

        # Embeddings
        self.char_embedding = nn.Embedding(num_chars, d_model, padding_idx=0)
        self.class_embedding = nn.Embedding(num_classes, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer Backbone
        # batch_first=True means input is (Batch, Seq, Feature)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Output Projection
        self.out = nn.Linear(d_model, num_chars)

        self.init_weights()

        # Token IDs for masking
        # Assuming 0 is PAD based on config
        self.pad_token_id = 0

    def init_weights(self):
        initrange = 0.1
        self.char_embedding.weight.data.uniform_(-initrange, initrange)
        self.class_embedding.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.zero_()
        self.out.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generates a causal mask (upper triangular matrix of -inf)."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def _create_masks(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Creates masks for Source Pad, Target Pad, and Target Look-ahead.
        src: (Batch, Src_Len)
        tgt: (Batch, Tgt_Len)
        """
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        # Target causal mask
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len).to(src.device)

        # Padding masks (True indicates value should be ignored)
        # src_padding_mask: (Batch, Src_Len)
        src_padding_mask = src == self.pad_token_id
        tgt_padding_mask = tgt == self.pad_token_id

        return tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(
        self, src_ids: torch.Tensor, tgt_ids: torch.Tensor, class_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            src_ids: (Batch, Src_Len) - Character IDs of input token
            tgt_ids: (Batch, Tgt_Len) - Character IDs of target token (input to decoder)
            class_ids: (Batch,) - Class ID of the token

        Returns:
            logits: (Batch, Tgt_Len, Num_Chars)
        """
        # 1. Prepare Source Embedding
        # Embed chars: (Batch, Src_Len, D_Model)
        src_emb = self.char_embedding(src_ids) * math.sqrt(self.d_model)

        # Embed class: (Batch, D_Model) -> (Batch, 1, D_Model)
        cls_emb = self.class_embedding(class_ids).unsqueeze(1) * math.sqrt(self.d_model)

        # Prepend class embedding to source sequence
        # New Src Shape: (Batch, Src_Len + 1, D_Model)
        src_emb = torch.cat([cls_emb, src_emb], dim=1)
        src_emb = self.pos_encoder(src_emb)

        # Update src padding mask to account for the prepended class token
        # The class token is never padding (False)
        batch_size = src_ids.size(0)
        cls_mask = torch.zeros(
            (batch_size, 1), dtype=torch.bool, device=src_ids.device
        )  # False
        src_padding_mask = src_ids == self.pad_token_id  # (Batch, Src_Len)
        src_key_padding_mask = torch.cat([cls_mask, src_padding_mask], dim=1)

        # 2. Prepare Target Embedding
        tgt_emb = self.char_embedding(tgt_ids) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        # 3. Create Masks
        tgt_mask = self._generate_square_subsequent_mask(tgt_ids.size(1)).to(
            src_ids.device
        )
        tgt_key_padding_mask = tgt_ids == self.pad_token_id

        # 4. Transformer Pass
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # 5. Output Projection
        logits = self.out(output)
        return logits

    def predict(
        self,
        src_ids: torch.Tensor,
        class_ids: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = Config.SEQ2SEQ_MAX_OUTPUT_LEN,
    ) -> torch.Tensor:
        """
        Greedy decoding for inference.

        Args:
            src_ids: (Batch, Src_Len)
            class_ids: (Batch,)
            sos_idx: Index of <SOS> token
            eos_idx: Index of <EOS> token
            max_len: Maximum generation length

        Returns:
            generated_ids: (Batch, Max_Len) - Padded with EOS/PAD
        """
        batch_size = src_ids.size(0)
        device = src_ids.device

        # 1. Encode Source
        src_emb = self.char_embedding(src_ids) * math.sqrt(self.d_model)
        cls_emb = self.class_embedding(class_ids).unsqueeze(1) * math.sqrt(self.d_model)
        src_emb = torch.cat([cls_emb, src_emb], dim=1)
        src_emb = self.pos_encoder(src_emb)

        # Source Mask
        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        src_padding_mask = src_ids == self.pad_token_id
        src_key_padding_mask = torch.cat([cls_mask, src_padding_mask], dim=1)

        # Encode
        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )

        # 2. Decode Loop
        # Initialize input with SOS: (Batch, 1)
        ys = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)

        # Keep track of finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for i in range(max_len):
            # Embed current target sequence
            tgt_emb = self.char_embedding(ys) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb)

            # Causal Mask
            tgt_mask = self._generate_square_subsequent_mask(ys.size(1)).to(device)

            # Decode step
            out = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )

            # Project last token
            # out: (Batch, Seq, Dim) -> Take last: (Batch, Dim)
            last_token_logits = self.out(out[:, -1, :])

            # Greedy choice
            _, next_word = torch.max(last_token_logits, dim=1)
            next_word = next_word.unsqueeze(1)

            # Update finished status
            is_eos = next_word.squeeze(1) == eos_idx
            finished = finished | is_eos

            # Append to sequence
            ys = torch.cat([ys, next_word], dim=1)

            # If all finished, break
            if finished.all():
                break

        # Remove SOS token from start
        return ys[:, 1:]
