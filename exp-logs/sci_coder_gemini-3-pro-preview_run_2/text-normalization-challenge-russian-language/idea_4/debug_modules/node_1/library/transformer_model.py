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

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape for batch_first=True: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Slice the positional encoding to the current sequence length
        # x.size(1) is the sequence length
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    """
    A standard Transformer architecture for Sequence-to-Sequence tasks.
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
        pad_token_id=0,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional Encoding
        # We add a small buffer to MAX_SEQ_LEN to be safe
        self.pos_encoder = PositionalEncoding(
            d_model, dropout, max_len=Config.MAX_SEQ_LEN + 50
        )

        # Transformer
        # batch_first=True means input/output tensors are [batch, seq, feature]
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Final Output Layer
        self.fc_out = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize parameters with Glorot / fan_avg."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag.
        Used for masking future tokens in the decoder.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def create_mask(self, src, tgt):
        """
        Creates the necessary masks for the Transformer.

        Args:
            src: Source tensor [batch, src_len]
            tgt: Target tensor [batch, tgt_len]

        Returns:
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
        """
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        # Target causal mask (prevent looking ahead)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(src.device)

        # Source mask (usually empty for encoder, as it can see everything)
        # However, nn.Transformer expects a tensor if provided.
        # We can pass None or a zero matrix.
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(
            torch.bool
        )

        # Padding masks (True where value is padding)
        src_padding_mask = src == self.pad_token_id
        tgt_padding_mask = tgt == self.pad_token_id

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src, tgt):
        """
        Forward pass for training.

        Args:
            src: Source sequence indices [batch, src_len]
            tgt: Target sequence indices [batch, tgt_len]
        """
        # Generate masks
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(
            src, tgt
        )

        # Embed and add positional encoding
        # Multiply by sqrt(d_model) as per Attention is All You Need paper
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))

        # Transformer Pass
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        # Project to vocabulary size
        return self.fc_out(outs)

    def encode(self, src):
        """
        Helper for inference: Runs the encoder part.

        Args:
            src: Source sequence indices [batch, src_len]

        Returns:
            memory: Encoder output [batch, src_len, d_model]
            src_padding_mask: Mask for padding tokens
        """
        src_padding_mask = src == self.pad_token_id
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_padding_mask
        )
        return memory, src_padding_mask

    def decode(self, tgt, memory, memory_key_padding_mask):
        """
        Helper for inference: Runs the decoder part.

        Args:
            tgt: Target sequence indices so far [batch, tgt_len]
            memory: Encoder output
            memory_key_padding_mask: Mask from encoder

        Returns:
            logits: Output logits [batch, tgt_len, vocab_size]
        """
        tgt_seq_len = tgt.shape[1]
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
        tgt_padding_mask = tgt == self.pad_token_id

        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))

        output = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.fc_out(output)
