import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List

from library.config import Config
from library.neural_architecture import RAGTransformer
from library.text_utils import CharTokenizer


class ModelTrainer:
    """
    Manages the training, evaluation, and checkpointing of the RAGTransformer model.
    """

    def __init__(self, tokenizer: CharTokenizer):
        """
        Initialize the trainer with model, optimizer, and loss function.

        Args:
            tokenizer: The fitted CharTokenizer instance containing vocabulary info.
        """
        self.tokenizer = tokenizer
        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        print(f"Initializing RAGTransformer on {self.device}...")
        self.model = RAGTransformer(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            d_model=Config.EMBED_DIM,
            nhead=Config.N_HEADS,
            num_encoder_layers=Config.N_ENCODER_LAYERS,
            num_decoder_layers=Config.N_DECODER_LAYERS,
            dim_feedforward=Config.HIDDEN_DIM,
            dropout=Config.DROPOUT,
        ).to(self.device)

        # Loss Function
        # We use CrossEntropyLoss with label smoothing for better generalization
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id,
            label_smoothing=Config.LABEL_SMOOTHING,
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

    def train_epoch(self, dataloader) -> Tuple[float, float]:
        """
        Runs one epoch of training.

        Returns:
            Tuple of (Average Loss, Average Character Accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            src = batch["input_ids"].to(self.device)
            tgt = batch["target_ids"].to(self.device)

            # Teacher Forcing Preparation
            # Decoder Input: <sos> ... x_n-1 (Everything except last token)
            tgt_input = tgt[:, :-1]
            # Target Output: x_1 ... <eos> (Everything except first token)
            tgt_output = tgt[:, 1:]

            self.optimizer.zero_grad()

            # Forward Pass
            # logits shape: (Batch, Seq_Len, Vocab)
            logits = self.model(src, tgt_input)

            # Reshape for Loss Calculation
            # Flatten batch and sequence dimensions
            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_output_flat = tgt_output.reshape(-1)

            loss = self.criterion(logits_flat, tgt_output_flat)

            # Backward Pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.GRADIENT_CLIP_VAL
            )

            self.optimizer.step()

            # Metrics
            batch_loss = loss.item()
            total_loss += batch_loss

            # Calculate Accuracy (ignoring padding)
            preds = torch.argmax(logits, dim=-1)
            mask = tgt_output != self.tokenizer.pad_token_id
            correct = (preds == tgt_output) & mask

            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_correct / total_tokens if total_tokens > 0 else 0.0

        return avg_loss, avg_acc

    def evaluate(self, dataloader) -> Tuple[float, float]:
        """
        Runs evaluation on the validation set.

        Returns:
            Tuple of (Average Loss, Average Character Accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                src = batch["input_ids"].to(self.device)
                tgt = batch["target_ids"].to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                logits = self.model(src, tgt_input)

                logits_flat = logits.reshape(-1, logits.size(-1))
                tgt_output_flat = tgt_output.reshape(-1)

                loss = self.criterion(logits_flat, tgt_output_flat)

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                mask = tgt_output != self.tokenizer.pad_token_id
                correct = (preds == tgt_output) & mask

                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_correct / total_tokens if total_tokens > 0 else 0.0

        return avg_loss, avg_acc

    def fit(self, train_loader, val_loader, epochs: int = Config.EPOCHS):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {epochs} epochs...")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.evaluate(val_loader)

            duration = time.time() - epoch_start

            # Print metrics with full precision
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Time: {duration:.2f}s | "
                f"Train Loss: {train_loss} | "
                f"Train Char Acc: {train_acc} | "
                f"Val Loss: {val_loss} | "
                f"Val Char Acc: {val_acc}"
            )

            # Checkpoint & Early Stopping
            if val_loss < best_val_loss:
                print(
                    f"Validation loss improved from {best_val_loss} to {val_loss}. Saving model..."
                )
                best_val_loss = val_loss
                patience_counter = 0

                # Save Best Model
                torch.save(self.model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            else:
                patience_counter += 1
                print(
                    f"Validation loss did not improve. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

        print("Training complete.")
