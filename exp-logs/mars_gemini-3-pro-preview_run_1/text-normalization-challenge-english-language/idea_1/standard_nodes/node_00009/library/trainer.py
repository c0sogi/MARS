import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from library.config import Config, set_seed
from library.utils import get_logger, timer
from library.model import BiLSTMTagger

logger = get_logger("trainer")


class Trainer:
    """
    Trainer class for the Bi-LSTM Tagger.
    Handles training, validation, early stopping, and model saving.
    """

    def __init__(self, model: BiLSTMTagger, train_loader, val_loader, vocab):
        """
        Args:
            model (BiLSTMTagger): The neural network model.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            vocab (Vocabulary): Vocabulary object containing token/class mappings and counts.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.device = torch.device(Config.DEVICE)

        # Move model to device
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

        # Loss Function with Class Weights
        self.criterion = self._get_loss_function()

        # Early Stopping parameters
        self.patience = Config.EARLY_STOPPING_PATIENCE
        self.best_val_loss = float("inf")
        self.counter = 0

        set_seed(Config.SEED)

    def _get_loss_function(self):
        """
        Calculates class weights and returns the CrossEntropyLoss function.
        """
        if Config.USE_CLASS_WEIGHTS:
            logger.info("Calculating class weights for loss function...")
            # Map class counts to IDs
            # vocab.class_counts is a Counter with class names
            # vocab.class2id maps class names to IDs

            num_classes = len(self.vocab.class2id)
            # Initialize counts array. Index 0 is PAD.
            counts = np.zeros(num_classes)

            total_samples = 0
            for cls_name, count in self.vocab.class_counts.items():
                if cls_name in self.vocab.class2id:
                    idx = self.vocab.class2id[cls_name]
                    counts[idx] = count
                    total_samples += count

            # Avoid division by zero for classes not in training set (if any) or PAD
            # We set PAD weight to 0 manually via ignore_index, but for calculation safety:
            counts = np.maximum(counts, 1)

            # Compute weights: Total / (Num_Classes * Class_Count)
            # This is a standard heuristic (inverse frequency)
            weights = total_samples / (num_classes * counts)

            # Apply smoothing (square root) to prevent extreme weights for rare classes
            weights = np.sqrt(weights)

            # Convert to tensor
            weight_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)

            logger.info(f"Class weights calculated. Shape: {weight_tensor.shape}")

            return nn.CrossEntropyLoss(
                weight=weight_tensor, ignore_index=Config.PAD_TOKEN_ID
            )
        else:
            return nn.CrossEntropyLoss(ignore_index=Config.PAD_TOKEN_ID)

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        total_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            seq_len = batch[
                "seq_len"
            ]  # CPU tensor usually fine for pack_padded_sequence length arg

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            # logits shape: (batch_size, seq_len, num_classes)
            logits = self.model(input_ids, seq_len, attention_mask)

            # Flatten for Loss Calculation
            # logits: (batch_size * seq_len, num_classes)
            # labels: (batch_size * seq_len)
            loss = self.criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / total_batches
        logger.info(f"Epoch {epoch} | Train Loss: {avg_loss}")
        return avg_loss

    def validate(self, epoch):
        """
        Runs validation on the validation set.
        Returns validation loss and accuracy.
        """
        self.model.eval()
        running_loss = 0.0
        correct_tokens = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                seq_len = batch["seq_len"]

                logits = self.model(input_ids, seq_len, attention_mask)

                # Loss
                loss = self.criterion(
                    logits.view(-1, logits.shape[-1]), labels.view(-1)
                )
                running_loss += loss.item()

                # Accuracy Calculation
                # Get predictions
                preds = torch.argmax(logits, dim=-1)

                # Create mask for non-padding tokens
                # labels == Config.PAD_TOKEN_ID are ignored
                mask = labels != Config.PAD_TOKEN_ID

                # Compare
                matches = (preds == labels) & mask

                correct_tokens += matches.sum().item()
                total_tokens += mask.sum().item()

        avg_loss = running_loss / len(self.val_loader)
        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

        logger.info(f"Epoch {epoch} | Val Loss: {avg_loss}")
        logger.info(f"Epoch {epoch} | Val Accuracy: {accuracy}")

        return avg_loss, accuracy

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        logger.info(f"Starting training on device: {self.device}")

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=1
        )

        for epoch in range(1, Config.EPOCHS + 1):
            with timer(f"Epoch {epoch}", logger):
                train_loss = self.train_epoch(epoch)
                val_loss, val_acc = self.validate(epoch)

                # Step Scheduler
                scheduler.step(val_loss)

                # Check for NaN
                if np.isnan(train_loss) or np.isnan(val_loss):
                    logger.error(
                        "Loss is NaN. Stopping training to prevent saving invalid model."
                    )
                    return False

                # Early Stopping Check
                if val_loss < self.best_val_loss:
                    logger.info(
                        f"Validation loss improved from {self.best_val_loss} to {val_loss}. Saving model..."
                    )
                    self.best_val_loss = val_loss
                    self.counter = 0

                    # Save Model
                    torch.save(self.model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
                else:
                    self.counter += 1
                    logger.info(
                        f"Validation loss did not improve. Counter: {self.counter}/{self.patience}"
                    )

                    if self.counter >= self.patience:
                        logger.info("Early stopping triggered. Training finished.")
                        break

        logger.info(f"Training complete. Best Val Loss: {self.best_val_loss}")
        return True
