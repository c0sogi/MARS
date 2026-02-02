import math
import torch
import torch.nn as nn
from library.config import Config
from library.backbone import AnisotropicBackbone


class PositionalEncoding(nn.Module):
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


class HybridCTCAttentionModel(nn.Module):
    def __init__(self):
        super(HybridCTCAttentionModel, self).__init__()

        # 1. Visual Backbone
        self.backbone = AnisotropicBackbone(
            model_name=Config.ENCODER_NAME, pretrained=True
        )

        # 2. Adapter (Backbone -> Transformer)
        self.feature_projection = nn.Linear(Config.ENCODER_DIM, Config.HIDDEN_DIM)
        self.pos_encoder = PositionalEncoding(
            Config.HIDDEN_DIM, Config.DROPOUT, max_len=Config.MAX_WIDTH // 2
        )

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.HIDDEN_DIM,
            nhead=Config.NUM_HEADS,
            dim_feedforward=Config.HIDDEN_DIM * 4,
            dropout=Config.DROPOUT,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.NUM_ENCODER_LAYERS
        )

        # 4. CTC Head (Auxiliary / Alignment)
        self.ctc_head = nn.Linear(Config.HIDDEN_DIM, Config.VOCAB_SIZE)

        # 5. Attention Decoder (Autoregressive)
        self.embedding = nn.Embedding(Config.VOCAB_SIZE, Config.HIDDEN_DIM)
        self.pos_decoder = PositionalEncoding(
            Config.HIDDEN_DIM, Config.DROPOUT, max_len=Config.MAX_SEQ_LEN
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=Config.HIDDEN_DIM,
            nhead=Config.NUM_HEADS,
            dim_feedforward=Config.HIDDEN_DIM * 4,
            dropout=Config.DROPOUT,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=Config.NUM_DECODER_LAYERS
        )

        self.attention_head = nn.Linear(Config.HIDDEN_DIM, Config.VOCAB_SIZE)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_causal_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, images, target_seqs=None):
        """
        Args:
            images: (B, C, H, W)
            target_seqs: (B, Seq_Len) - Optional, for training decoder
        Returns:
            ctc_logits: (B, T_enc, Vocab)
            attn_logits: (B, Seq_Len, Vocab) or None
        """
        # --- Encoder Path ---
        # 1. Extract Features
        features = self.backbone(images)  # (B, T, 512)

        # 2. Project and Add Position
        features = self.feature_projection(features)  # (B, T, 256)
        features = self.pos_encoder(features)

        # 3. Transformer Encoder
        memory = self.transformer_encoder(features)  # (B, T, 256)

        # 4. CTC Head Output
        ctc_logits = self.ctc_head(memory)  # (B, T, Vocab)

        # --- Decoder Path ---
        attn_logits = None
        if target_seqs is not None:
            # target_seqs usually includes SOS.
            # For teacher forcing, we input target_seqs and predict next token.
            # Standard practice: Input is target_seqs (with SOS, without EOS ideally for length match,
            # but usually we pass the whole thing and shift labels in loss).

            # Embed targets
            tgt_emb = self.embedding(target_seqs)  # (B, L, 256)
            tgt_emb = self.pos_decoder(tgt_emb)

            # Create masks
            seq_len = target_seqs.size(1)
            causal_mask = self.make_causal_mask(seq_len).to(images.device)
            # Padding mask: True where value is PAD
            tgt_padding_mask = target_seqs == Config.PAD_IDX

            # Transformer Decoder
            decoder_output = self.transformer_decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                # memory_key_padding_mask could be added if we had variable length images in a batch padded
            )

            attn_logits = self.attention_head(decoder_output)

        return ctc_logits, attn_logits

    def predict(self, images, max_len=Config.MAX_SEQ_LEN):
        """
        Greedy decoding for inference.
        Args:
            images: (B, C, H, W)
            max_len: Maximum sequence length
        Returns:
            predictions: (B, max_len) tensor of indices
        """
        self.eval()
        device = images.device
        batch_size = images.size(0)

        with torch.no_grad():
            # --- Encoder ---
            features = self.backbone(images)
            features = self.feature_projection(features)
            features = self.pos_encoder(features)
            memory = self.transformer_encoder(features)

            # --- Decoder (Greedy) ---
            # Start with SOS
            input_seq = torch.full(
                (batch_size, 1), Config.SOS_IDX, dtype=torch.long
            ).to(device)

            # Keep track of finished sequences
            finished = torch.zeros(batch_size, dtype=torch.bool).to(device)

            for _ in range(max_len):
                tgt_emb = self.embedding(input_seq)
                tgt_emb = self.pos_decoder(tgt_emb)

                # No causal mask needed for greedy step-by-step, but PyTorch decoder expects sequence
                # We can just pass the growing sequence. It's inefficient but simple.
                # A causal mask is strictly required if we pass the whole sequence so far.
                seq_len = input_seq.size(1)
                causal_mask = self.make_causal_mask(seq_len).to(device)

                decoder_output = self.transformer_decoder(
                    tgt=tgt_emb, memory=memory, tgt_mask=causal_mask
                )

                # Get logits for the last token only
                last_token_logits = self.attention_head(decoder_output[:, -1, :])

                # Greedy choice
                _, next_token = torch.max(last_token_logits, dim=1)

                # Update input sequence
                input_seq = torch.cat([input_seq, next_token.unsqueeze(1)], dim=1)

                # Check EOS
                is_eos = next_token == Config.EOS_IDX
                finished = finished | is_eos

                if finished.all():
                    break

        return input_seq
