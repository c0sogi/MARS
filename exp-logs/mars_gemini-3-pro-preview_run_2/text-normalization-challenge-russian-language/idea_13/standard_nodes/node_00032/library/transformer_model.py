import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import time
from typing import Tuple

from library.config import Config
from library.utils import get_device, ensure_dir


class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [Seq_Len, Batch_Size, Embedding_Dim]
        """
        # x is [Seq_Len, Batch, Dim]
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    """
    Tier 2: Anchored Heterogeneous Transformer.

    Encoder: Character-Level (High granularity for parsing numbers/symbols)
    Decoder: Subword-Level (BPE for valid morphology)
    """

    def __init__(
        self,
        config: Config,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
    ):
        super().__init__()
        self.config = config
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.d_model = config.d_model

        # Embeddings
        self.src_tok_emb = nn.Embedding(src_vocab_size, config.d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, config.d_model)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(
            config.d_model,
            dropout=config.dropout,
            max_len=max(config.max_enc_len, config.max_dec_len) + 50,
        )

        # Transformer
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=False,  # We will transpose inputs to (Seq, Batch) for Transformer
        )

        # Output Projection
        self.generator = nn.Linear(config.d_model, tgt_vocab_size)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        Args:
            src: [Batch, Src_Seq_Len]
            tgt: [Batch, Tgt_Seq_Len]

        Returns:
            logits: [Batch, Tgt_Seq_Len, Tgt_Vocab_Size]
        """
        # Transpose to [Seq, Batch] for nn.Transformer (default behavior is usually faster)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        # Generate Masks
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(src.device)
        src_key_padding_mask = (src == self.src_pad_idx).transpose(0, 1)  # [Batch, Seq]
        tgt_key_padding_mask = (tgt == self.tgt_pad_idx).transpose(0, 1)  # [Batch, Seq]

        # Embeddings + Positional
        src_emb = self.positional_encoding(
            self.src_tok_emb(src) * math.sqrt(self.d_model)
        )
        tgt_emb = self.positional_encoding(
            self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)
        )

        # Transformer Pass
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # Project and Transpose back to [Batch, Seq, Vocab]
        return self.generator(outs).transpose(0, 1)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None):
        """
        Encodes the source sequence. Used during inference.
        src: [Batch, Seq]
        """
        src = src.transpose(0, 1)  # [Seq, Batch]
        src_key_padding_mask = (src == self.src_pad_idx).transpose(0, 1)

        src_emb = self.positional_encoding(
            self.src_tok_emb(src) * math.sqrt(self.d_model)
        )
        return self.transformer.encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )

    def decode(
        self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor = None
    ):
        """
        Decodes one step. Used during inference.
        tgt: [Batch, Seq]
        memory: [Seq, Batch] (Output of encoder)
        """
        tgt = tgt.transpose(0, 1)  # [Seq, Batch]
        tgt_key_padding_mask = (tgt == self.tgt_pad_idx).transpose(0, 1)

        tgt_emb = self.positional_encoding(
            self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)
        )

        # memory_key_padding_mask is not strictly needed here if memory is already masked/processed,
        # but nn.TransformerDecoder usually doesn't take it if not passed.
        # Ideally, we should pass the src_padding_mask if we had access to it here.
        # For greedy decoding, we usually assume the encoder output 'memory' is sufficient.

        return self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask


class TransformerTrainer:
    """
    Manages the training lifecycle of the Seq2SeqTransformer.
    """

    def __init__(
        self, config: Config, model: Seq2SeqTransformer, train_loader, val_loader
    ):
        self.config = config
        self.device = get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=config.weight_decay,
        )

        # Loss with Label Smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.model.tgt_pad_idx, label_smoothing=config.label_smoothing
        )

        # Scheduler: Linear Warmup + Cosine Decay
        self.scheduler = self._get_scheduler()

        # Checkpointing
        self.checkpoint_dir = os.path.join(config.base_working_dir, "checkpoints")
        ensure_dir(os.path.join(self.checkpoint_dir, "placeholder"))
        self.best_model_path = os.path.join(self.checkpoint_dir, "transformer_best.pth")

    def _get_scheduler(self):
        """Creates a LambdaLR scheduler for linear warmup and cosine decay."""

        def lr_lambda(step):
            # Total steps estimation
            total_steps = len(self.train_loader) * self.config.epochs
            warmup_steps = self.config.warmup_steps

            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))

            # Cosine decay
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(self):
        """Main training loop with early stopping."""
        print(f"Starting training on {self.device}...")
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            start_time = time.time()

            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._evaluate()

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch:02d} | Time: {elapsed:.2f}s | "
                f"Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.5f} | "
                f"Val Loss: {val_loss:.5f} | Val Acc: {val_acc:.5f}"
            )

            # Checkpointing & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  -> New best model saved to {self.best_model_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"  -> Early stopping triggered after {epoch} epochs.")
                    break

        # Load best model
        print("Loading best model for inference...")
        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )

    def _train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        for batch in self.train_loader:
            src = batch[0].to(self.device)
            tgt = batch[1].to(self.device)

            # tgt input for model is <BOS> ... <Last_Token>
            # tgt output for loss is <First_Token> ... <EOS>
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            self.optimizer.zero_grad()

            logits = self.model(src, tgt_input)

            # Reshape for loss: [Batch * Seq, Vocab]
            loss = self.criterion(
                logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1)
            )

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            # Accuracy calculation (ignoring padding)
            preds = torch.argmax(logits, dim=-1)
            mask = tgt_output != self.model.tgt_pad_idx
            correct = (preds == tgt_output) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

        return total_loss / len(self.train_loader), (
            total_correct / total_tokens if total_tokens > 0 else 0.0
        )

    def _evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in self.val_loader:
                src = batch[0].to(self.device)
                tgt = batch[1].to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                logits = self.model(src, tgt_input)

                loss = self.criterion(
                    logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1)
                )
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                mask = tgt_output != self.model.tgt_pad_idx
                correct = (preds == tgt_output) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()

        return total_loss / len(self.val_loader), (
            total_correct / total_tokens if total_tokens > 0 else 0.0
        )
