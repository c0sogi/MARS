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

        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerNumNorm(nn.Module):
    """
    Transformer Encoder-Decoder model for Text Normalization.
    Includes an auxiliary classification head on the encoder output.
    """

    def __init__(
        self,
        vocab_size,
        num_classes,
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        num_encoder_layers=Config.NUM_ENCODER_LAYERS,
        num_decoder_layers=Config.NUM_DECODER_LAYERS,
        dim_feedforward=Config.DIM_FEEDFORWARD,
        dropout=Config.DROPOUT,
        max_seq_len=Config.MAX_SEQ_LEN,
    ):
        super(TransformerNumNorm, self).__init__()

        self.d_model = d_model

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(
            d_model, max_len=max_seq_len, dropout=dropout
        )

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Heads
        self.text_head = nn.Linear(d_model, vocab_size)
        self.class_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.text_head.bias.data.zero_()
        self.text_head.weight.data.uniform_(-initrange, initrange)
        # Class head initialization
        for m in self.class_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def generate_square_subsequent_mask(self, sz):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def make_src_mask(self, src):
        # src: (batch, seq_len)
        # mask: (batch, seq_len) -> True where padding
        return src == Config.PAD_IDX

    def make_tgt_mask(self, tgt):
        # tgt: (batch, seq_len)
        # padding mask: (batch, seq_len)
        tgt_pad_mask = tgt == Config.PAD_IDX
        # look ahead mask: (seq_len, seq_len)
        tgt_len = tgt.size(1)
        tgt_sub_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        return tgt_pad_mask, tgt_sub_mask

    def encode(self, src):
        """
        Runs the encoder part of the model.
        """
        # Create mask
        src_key_padding_mask = self.make_src_mask(src)  # (batch, seq_len)

        # Embed and PosEnc
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        # Encoder
        memory = self.transformer_encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )

        return memory, src_key_padding_mask

    def decode(self, tgt, memory, memory_key_padding_mask=None):
        """
        Runs the decoder part of the model.
        """
        tgt_pad_mask, tgt_sub_mask = self.make_tgt_mask(tgt)

        # Embed and PosEnc
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Decoder
        output = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_sub_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return output

    def forward(self, src, tgt):
        """
        Args:
            src: (batch_size, src_len)
            tgt: (batch_size, tgt_len)
        Returns:
            text_logits: (batch_size, tgt_len, vocab_size)
            class_logits: (batch_size, num_classes)
        """
        # 1. Encode
        memory, src_key_padding_mask = self.encode(src)

        # 2. Auxiliary Classification
        # Pooling: Mean over non-padding tokens
        # Create a mask for division (batch, 1)
        mask_float = (~src_key_padding_mask).float().unsqueeze(-1)  # (batch, seq, 1)

        # Zero out padding in memory just in case
        memory_masked = memory * mask_float

        # Sum and divide by length
        sum_pooled = torch.sum(memory_masked, dim=1)  # (batch, d_model)
        lengths = torch.sum(mask_float, dim=1)  # (batch, 1)
        # Avoid division by zero
        lengths = torch.clamp(lengths, min=1.0)
        mean_pooled = sum_pooled / lengths

        class_logits = self.class_head(mean_pooled)

        # 3. Decode
        output = self.decode(tgt, memory, memory_key_padding_mask=src_key_padding_mask)

        # 4. Text Generation Head
        text_logits = self.text_head(output)

        return text_logits, class_logits
