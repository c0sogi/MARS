import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import os
from library.config import Config


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for Transformer models.
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
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerSeq2Seq(nn.Module):
    """
    Character-level Encoder-Decoder Transformer for Text Normalization.
    """

    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        pad_token_id=0,
        sos_token_id=1,
        eos_token_id=2,
        max_len=128,
    ):
        super(TransformerSeq2Seq, self).__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.max_len = max_len

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len)

        # Core Transformer
        # batch_first=True expects inputs of shape (batch, seq, feature)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)

    def create_mask(self, src, tgt):
        """
        Creates masks for padding and causal attention.
        """
        tgt_seq_len = tgt.shape[1]

        # Target causal mask (prevent attending to future tokens)
        # Shape: (tgt_len, tgt_len)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(
            src.device
        )

        # Padding masks (True where padding exists)
        # Shape: (batch, seq_len)
        src_padding_mask = src == self.pad_token_id
        tgt_padding_mask = tgt == self.pad_token_id

        return tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src, tgt):
        """
        Forward pass for training.
        Args:
            src: (batch, src_len) Source token indices.
            tgt: (batch, tgt_len) Target token indices (decoder input).
        """
        # Generate masks
        tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt)

        # Apply embeddings and positional encoding
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Transformer pass
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        # Project to vocabulary size
        return self.fc_out(outs)

    def encode(self, src):
        """
        Encodes the source sequence. Used during inference.
        """
        src_padding_mask = src == self.pad_token_id
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_padding_mask
        )
        return memory, src_padding_mask

    @torch.no_grad()
    def generate(self, src, max_len=None):
        """
        Greedy decoding for inference.
        Args:
            src: (batch, src_len) Source token indices.
            max_len: Maximum length of generated sequence.
        Returns:
            Tensor: (batch, gen_len) Generated token indices.
        """
        if max_len is None:
            max_len = self.max_len

        self.eval()
        device = src.device
        batch_size = src.size(0)

        # Encode source
        memory, memory_mask = self.encode(src)

        # Initialize decoder input with SOS token
        ys = (
            torch.ones(batch_size, 1)
            .fill_(self.sos_token_id)
            .type(torch.long)
            .to(device)
        )

        # Track finished sequences (those that hit EOS)
        finished = torch.zeros(batch_size, dtype=torch.bool).to(device)

        for i in range(max_len - 1):
            # Create causal mask for current sequence length
            tgt_mask = self.transformer.generate_square_subsequent_mask(ys.size(1)).to(
                device
            )

            # Embed decoder input
            tgt_emb = self.embedding(ys) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb)

            # Decode
            out = self.transformer.decoder(
                tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask
            )

            # Project last token output to vocab
            prob = self.fc_out(out[:, -1])
            _, next_word = torch.max(prob, dim=1)

            # Check for EOS
            is_eos = next_word == self.eos_token_id
            finished = finished | is_eos

            # Append prediction
            next_word = next_word.unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)

            # Stop if all sequences in batch are finished
            if finished.all():
                break

        return ys


class Trainer:
    """
    Handles training, validation, and checkpointing.
    """

    def __init__(
        self, model, train_loader, val_loader, optimizer, device, save_path, patience=3
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.patience = patience

        # CrossEntropyLoss expects class indices. Ignore padding index.
        self.criterion = nn.CrossEntropyLoss(ignore_index=model.pad_token_id)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        start_time = time.time()

        for batch in self.train_loader:
            src = batch["src"].to(self.device)
            tgt = batch["tgt"].to(self.device)

            # Prepare inputs and targets for Teacher Forcing
            # Decoder Input: <sos> ... token_n-1
            tgt_input = tgt[:, :-1]
            # Labels: token_1 ... <eos>
            tgt_output = tgt[:, 1:]

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(src, tgt_input)

            # Reshape for loss calculation: (batch * seq_len, vocab_size)
            loss = self.criterion(
                logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1)
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.CLIP_GRAD)

            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = (
            total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        )
        duration = time.time() - start_time
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.6f} | Time: {duration:.2f}s")
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                src = batch["src"].to(self.device)
                tgt = batch["tgt"].to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                logits = self.model(src, tgt_input)
                loss = self.criterion(
                    logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1)
                )
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        print(f"Epoch {epoch} | Val Loss: {avg_loss:.16f}")
        return avg_loss

    def fit(self, epochs):
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training on {self.device}...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            # Checkpointing and Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f"New best model saved to {self.save_path}")
            else:
                patience_counter += 1
                print(
                    f"EarlyStopping counter: {patience_counter} out of {self.patience}"
                )

            if patience_counter >= self.patience:
                print("Early stopping triggered.")
                break

        print("Training complete.")
