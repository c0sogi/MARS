import torch
import torch.nn as nn
import os
from library.config import Config
from library.utils import set_seed
from library.neural_net import CharToSubwordTransformer
from library.data_utils import get_dataloaders


class Trainer:
    """
    Encapsulates the training loop for the Residual Transformer model.
    """

    def __init__(self, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG):
        """
        Initializes the Trainer with data, model, optimizer, and criterion.
        """
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)

        # Load Data
        # get_dataloaders returns (train_loader, val_loader, tokenizer)
        self.train_loader, self.val_loader, self.tokenizer = get_dataloaders(
            batch_size=batch_size, debug=debug
        )

        # Check if data exists
        if len(self.train_loader) == 0:
            print("Warning: Train loader is empty. No residuals found for training.")
            self.model = None
            return

        # Initialize Model
        # Note: CharTokenizer uses .vocab_size attribute, TargetBPETokenizer uses .vocab_size() method
        self.model = CharToSubwordTransformer(
            src_vocab_size=self.tokenizer.char.vocab_size,
            tgt_vocab_size=self.tokenizer.bpe.vocab_size(),
            pad_idx=self.tokenizer.char.pad_token_id,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.bpe.pad_id,
            label_smoothing=Config.LABEL_SMOOTHING,
        )

    def train_epoch(self) -> float:
        """
        Runs one epoch of training.
        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for src, tgt in self.train_loader:
            src, tgt = src.to(self.device), tgt.to(self.device)

            # Transformer target setup:
            # Input to decoder: tgt[:, :-1] (exclude last token)
            # Target for loss: tgt[:, 1:] (exclude first token/BOS)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(src, tgt_input)

            # Reshape logits and targets for CrossEntropyLoss
            # Logits: (batch * seq_len, vocab_size)
            # Targets: (batch * seq_len)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.GRADIENT_CLIP_VAL
            )
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def evaluate(self) -> float:
        """
        Runs evaluation on the validation set.
        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for src, tgt in self.val_loader:
                src, tgt = src.to(self.device), tgt.to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                logits = self.model(src, tgt_input)

                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
                )
                total_loss += loss.item()

        if len(self.val_loader) == 0:
            return 0.0

        return total_loss / len(self.val_loader)

    def fit(self):
        """
        Executes the full training process with early stopping.
        """
        if self.model is None:
            print("Model not initialized (likely due to empty dataset). Aborting fit.")
            return

        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training on {len(self.train_loader.dataset)} samples...")

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = self.train_epoch()
            val_loss = self.evaluate()

            # Print full precision as requested
            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss}"
            )

            # Checkpoint and Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print("Training complete.")

    def save_checkpoint(self):
        """
        Saves the model state and vocabulary to disk.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "char_vocab": self.tokenizer.char.char2idx,
            "config": {
                "src_vocab_size": self.tokenizer.char.vocab_size,
                "tgt_vocab_size": self.tokenizer.bpe.vocab_size(),
            },
        }
        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.TRANSFORMER_CHECKPOINT), exist_ok=True)
        torch.save(checkpoint, Config.TRANSFORMER_CHECKPOINT)
