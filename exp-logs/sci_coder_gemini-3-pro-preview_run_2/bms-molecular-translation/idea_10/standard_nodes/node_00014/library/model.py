import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from library.config import Config
from library.backbone import AnisotropicResNet
from library.tokenizer import InChiTokenizer


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for Transformer.
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
        # x: (B, Seq_Len, D)
        # Slice pe to the current sequence length
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class HybridResNetTransformer(nn.Module):
    """
    Hybrid architecture combining an Anisotropic ResNet backbone with a Transformer.

    Structure:
    Image -> AnisotropicResNet -> Feature Sequence -> Transformer Encoder -> Memory
    Memory -> CTC Head -> CTC Loss (Auxiliary)
    Memory + Targets -> Transformer Decoder -> Prediction Head -> CE Loss (Primary)
    """

    def __init__(self, config: Config, tokenizer: InChiTokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        vocab_size = len(tokenizer)

        # 1. Visual Backbone
        self.backbone = AnisotropicResNet(config)

        # 2. Sequence Encoder
        # Models contextual dependencies in the flattened image feature sequence
        self.pos_encoder = PositionalEncoding(
            config.encoder_dim, config.dropout, max_len=2048
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_dim,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.seq_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_encoder_layers
        )

        # 3. CTC Head (Auxiliary)
        # Projects encoder output directly to vocab for structural alignment
        self.ctc_head = nn.Linear(config.encoder_dim, vocab_size)

        # 4. Transformer Decoder (Primary)
        # Autoregressive generation of InChI string
        self.embedding = nn.Embedding(
            vocab_size, config.decoder_dim, padding_idx=tokenizer.PAD_ID
        )
        self.pos_decoder = PositionalEncoding(
            config.decoder_dim, config.dropout, max_len=config.max_len
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_dim,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.num_decoder_layers
        )

        self.prediction_head = nn.Linear(config.decoder_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize parameters with Xavier Uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Explicitly zero out padding embedding
        self.embedding.weight.data[self.tokenizer.PAD_ID] = 0

    def encode_image(self, images):
        """
        Pass images through backbone and encoder.
        Returns: memory (B, Seq_Len, D)
        """
        # Extract features: (B, Seq_Len, Enc_Dim)
        features = self.backbone(images)

        # Add positional encoding
        features = self.pos_encoder(features)

        # Pass through Transformer Encoder
        memory = self.seq_encoder(features)
        return memory

    def make_len_mask(self, inp):
        """
        Create padding mask for Transformer.
        True indicates the position should be ignored.
        """
        return inp == self.tokenizer.PAD_ID

    def generate_square_subsequent_mask(self, sz):
        """
        Generate causal mask for decoder self-attention.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, images, targets=None):
        """
        Forward pass.

        Args:
            images: (B, C, H, W)
            targets: (B, L) - padded target sequences (optional, for training)

        Returns:
            dict containing 'ctc_logits' and 'decoder_logits'
        """
        # 1. Encode
        # memory: (B, Src_Len, D)
        memory = self.encode_image(images)

        # 2. CTC Head Output
        # ctc_logits: (B, Src_Len, V)
        ctc_logits = self.ctc_head(memory)

        decoder_logits = None

        # 3. Decoder (Training Mode)
        if targets is not None:
            # Teacher Forcing: Input is targets excluding the last token (EOS/PAD)
            # targets shape: (B, L)
            # dec_input: (B, L-1)
            dec_input = targets[:, :-1]

            # Embeddings
            tgt_emb = self.embedding(dec_input)
            tgt_emb = self.pos_decoder(tgt_emb)

            # Masks
            # Causal mask for self-attention
            tgt_seq_len = dec_input.size(1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(
                images.device
            )

            # Padding mask for self-attention (ignore pads in target input)
            tgt_key_padding_mask = self.make_len_mask(dec_input).to(images.device)

            # Decode
            dec_output = self.transformer_decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )

            # Project to vocabulary
            decoder_logits = self.prediction_head(dec_output)

        return {"ctc_logits": ctc_logits, "decoder_logits": decoder_logits}

    def calc_loss(self, outputs, targets, lengths, ctc_weight=0.5):
        """
        Compute Joint Loss: CTC + CrossEntropy.

        Args:
            outputs: dict from forward()
            targets: (B, L) raw targets with SOS/EOS
            lengths: (B,) lengths of targets
            ctc_weight: float, weight for CTC loss
        """
        ctc_logits = outputs["ctc_logits"]  # (B, T, V)
        decoder_logits = outputs["decoder_logits"]  # (B, L-1, V)

        losses = {}

        # --- CTC Loss ---
        # Permute to (T, B, V) for PyTorch CTCLoss
        ctc_input = ctc_logits.permute(1, 0, 2).log_softmax(2)

        # Input lengths for CTC: Full sequence length of encoder output
        B = ctc_logits.size(0)
        T = ctc_logits.size(1)
        input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long).to(
            ctc_logits.device
        )

        # CTC Loss
        # We use the full target sequence (including SOS/EOS) for CTC alignment
        ctc_loss_fn = nn.CTCLoss(blank=self.tokenizer.PAD_ID, zero_infinity=True)
        loss_ctc = ctc_loss_fn(ctc_input, targets, input_lengths, lengths)
        losses["ctc"] = loss_ctc

        # --- Cross Entropy Loss ---
        if decoder_logits is not None:
            # Targets for CE are shifted by 1 (predict next token)
            # Input was [SOS, A, B], Target is [A, B, EOS]
            ce_targets = targets[:, 1:]

            # Flatten for CrossEntropyLoss
            # decoder_logits: (B, L-1, V) -> (B*(L-1), V)
            # ce_targets: (B, L-1) -> (B*(L-1))
            loss_ce = F.cross_entropy(
                decoder_logits.reshape(-1, decoder_logits.size(-1)),
                ce_targets.reshape(-1),
                ignore_index=self.tokenizer.PAD_ID,
            )
            losses["ce"] = loss_ce
        else:
            loss_ce = torch.tensor(0.0, device=ctc_logits.device)

        # --- Total Loss ---
        loss = ctc_weight * loss_ctc + (1 - ctc_weight) * loss_ce
        losses["total"] = loss

        return losses

    @torch.no_grad()
    def predict(self, images, max_len=None):
        """
        Greedy decoding for inference.

        Args:
            images: (B, C, H, W)
            max_len: Maximum generation length

        Returns:
            list of predicted strings
        """
        self.eval()
        if max_len is None:
            max_len = self.config.max_len

        B = images.size(0)
        device = images.device

        # Encode
        memory = self.encode_image(images)

        # Initialize input with SOS
        sos_id = self.tokenizer.SOS_ID
        eos_id = self.tokenizer.EOS_ID

        # Generated sequence: (B, 1)
        generated = torch.full((B, 1), sos_id, dtype=torch.long).to(device)

        # Active beams mask (True = still generating)
        active_mask = torch.ones(B, dtype=torch.bool).to(device)

        for _ in range(max_len):
            # Embed current sequence
            tgt_emb = self.embedding(generated)
            tgt_emb = self.pos_decoder(tgt_emb)

            # Causal mask
            tgt_mask = self.generate_square_subsequent_mask(generated.size(1)).to(
                device
            )

            # Decode
            # We assume no padding mask needed for generated sequence as it has no pads yet
            dec_output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)

            # Get logits for the last generated token
            # (B, 1, V)
            last_logits = self.prediction_head(dec_output[:, -1:, :])

            # Greedy choice
            next_token = torch.argmax(last_logits, dim=-1)  # (B, 1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Update active mask: if EOS generated, mark as finished
            is_eos = next_token.squeeze(-1) == eos_id
            active_mask = active_mask & (~is_eos)

            # If all finished, stop
            if not active_mask.any():
                break

        # Decode indices to strings
        predictions = []
        for i in range(B):
            # Exclude SOS (index 0)
            indices = generated[i, 1:]
            pred_str = self.tokenizer.decode(indices)
            predictions.append(pred_str)

        return predictions
