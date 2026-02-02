import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from library.config import Config
from library.utils import set_seed, MetricTracker, get_device
from library.text_processing import CharTokenizer, TargetTokenizer

# ==========================================
# Dataset
# ==========================================


class ResidualDataset(Dataset):
    """
    PyTorch Dataset for the Curriculum-Enriched Residuals.
    Handles tokenization of input (Char) and target (BPE).
    """

    def __init__(self, df, char_tokenizer, target_tokenizer, max_len=128):
        self.df = df.reset_index(drop=True)
        self.char_tokenizer = char_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_len = max_len
        self.sep_str = "<SEP>"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_text = row["input_text"]
        target_text = row["target_text"]

        # --- Source Processing (Char Level) ---
        # The input string contains literal "<SEP>". We need to split it
        # and insert the actual SEP token ID.
        parts = input_text.split(self.sep_str)
        src_indices = []

        # Add SOS
        src_indices.append(self.char_tokenizer.sos_token_id)

        for i, part in enumerate(parts):
            # Encode the text segment (chars -> ids)
            # We don't use add_special_tokens=True here because we handle them manually
            part_indices = self.char_tokenizer.encode(part, add_special_tokens=False)
            src_indices.extend(part_indices)

            # Add SEP if not the last part
            if i < len(parts) - 1:
                src_indices.append(self.char_tokenizer.sep_token_id)

        # Add EOS
        src_indices.append(self.char_tokenizer.eos_token_id)

        # Truncate if necessary
        if len(src_indices) > self.max_len:
            src_indices = src_indices[: self.max_len]
            # Ensure EOS is at the end if truncated
            src_indices[-1] = self.char_tokenizer.eos_token_id

        # --- Target Processing (BPE Level) ---
        # Encode with SentencePiece
        # We need BOS for input to decoder, and EOS for target label
        tgt_indices_raw = self.target_tokenizer.encode(target_text)

        # Construct sequence: [BOS, t1, t2, ..., tn, EOS]
        tgt_indices = (
            [self.target_tokenizer.bos_id]
            + tgt_indices_raw
            + [self.target_tokenizer.eos_id]
        )

        if len(tgt_indices) > self.max_len:
            tgt_indices = tgt_indices[: self.max_len]
            tgt_indices[-1] = self.target_tokenizer.eos_id

        return {
            "src": torch.tensor(src_indices, dtype=torch.long),
            "tgt": torch.tensor(tgt_indices, dtype=torch.long),
        }


def collate_fn(batch):
    """
    Custom collate function to pad sequences in the batch.
    """
    src_batch = [item["src"] for item in batch]
    tgt_batch = [item["tgt"] for item in batch]

    # Pad with 0 (assuming 0 is PAD for BPE, check CharTokenizer for its PAD)
    # CharTokenizer PAD is usually 0 if defined first in special_tokens
    # TargetTokenizer (SentencePiece) PAD is usually 0 (pad_id)

    # We need to get PAD IDs dynamically ideally, but usually 0 is safe for SP
    # For CharTokenizer, we need to check the instance passed to dataset,
    # but here we assume standard padding.

    # Note: pad_sequence pads with 0 by default.
    # We need to ensure we use the correct padding values.
    # Ideally we'd access tokenizer.pad_token_id, but collate_fn is static here.
    # We will assume 0 is PAD. In text_processing.py:
    # CharTokenizer: PAD_TOKEN is first in special_tokens -> index 0.
    # TargetTokenizer: pad_id is 0.

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    return src_padded, tgt_padded


# ==========================================
# Model Architecture
# ==========================================


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
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        emb_size,
        nhead,
        src_vocab_size,
        tgt_vocab_size,
        dim_feedforward,
        dropout=0.1,
    ):
        super(Seq2SeqTransformer, self).__init__()

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(
        self,
        src,
        tgt,
        src_mask,
        tgt_mask,
        src_padding_mask,
        tgt_padding_mask,
        memory_key_padding_mask,
    ):
        """
        Forward pass for training.
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src, src_mask):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt, pad_idx_src, pad_idx_tgt, device):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = src == pad_idx_src
    tgt_padding_mask = tgt == pad_idx_tgt

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# ==========================================
# Trainer
# ==========================================


class TransformerTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler,
        device,
        config,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.config = config

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self, epoch):
        self.model.train()
        losses = MetricTracker()

        for i, (src, tgt) in enumerate(self.train_loader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            # Create masks
            # tgt_input is tgt excluding the last token (EOS)
            tgt_input = tgt[:, :-1]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src,
                tgt_input,
                pad_idx_src=0,
                pad_idx_tgt=0,  # Assuming 0 is PAD
                device=self.device,
            )

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                logits = self.model(
                    src,
                    tgt_input,
                    src_mask,
                    tgt_mask,
                    src_padding_mask,
                    tgt_padding_mask,
                    src_padding_mask,
                )

                # Calculate loss
                # tgt_out is tgt excluding the first token (BOS)
                tgt_out = tgt[:, 1:]

                # Reshape for loss
                loss = self.criterion(
                    logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)
                )

            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            losses.update(loss.item(), src.size(0))

            # Warmup step if applicable
            if hasattr(self.scheduler, "step_batch"):
                self.scheduler.step_batch()

        return losses.avg

    def evaluate(self):
        self.model.eval()
        losses = MetricTracker()

        with torch.no_grad():
            for src, tgt in self.val_loader:
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_out = tgt[:, 1:]

                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                    src, tgt_input, pad_idx_src=0, pad_idx_tgt=0, device=self.device
                )

                with torch.cuda.amp.autocast():
                    logits = self.model(
                        src,
                        tgt_input,
                        src_mask,
                        tgt_mask,
                        src_padding_mask,
                        tgt_padding_mask,
                        src_padding_mask,
                    )

                    loss = self.criterion(
                        logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)
                    )
                losses.update(loss.item(), src.size(0))

        return losses.avg

    def fit(self):
        print(f"Starting training for {self.config.EPOCHS} epochs...")

        for epoch in range(1, self.config.EPOCHS + 1):
            start_time = time.time()

            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()

            # Scheduler step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch} | Time: {elapsed:.2f}s | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            # Early Stopping and Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.config.MODEL_BEST_PATH)
                print(f"  -> Model saved (improved loss).")
            else:
                self.patience_counter += 1
                print(
                    f"  -> No improvement. Patience: {self.patience_counter}/{self.config.EARLY_STOPPING_PATIENCE}"
                )

            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Val Loss: {self.best_val_loss:.6f}")


def get_model(src_vocab_size, tgt_vocab_size, device):
    """
    Factory function to create the model based on Config.
    """
    model = Seq2SeqTransformer(
        num_encoder_layers=Config.NUM_LAYERS,
        num_decoder_layers=Config.NUM_LAYERS,
        emb_size=Config.D_MODEL,
        nhead=Config.NHEAD,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        dim_feedforward=Config.DIM_FEEDFORWARD,
        dropout=Config.DROPOUT,
    )

    # Initialize parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model.to(device)
