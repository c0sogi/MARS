import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
import numpy as np

from library.config import Config
from library.utils import set_seed, Timer
from library.model import DistilRobertaForTagging


class Trainer:
    """
    Trainer class for Fine-Tuned DistilRoBERTa.
    Handles training loop, validation, mixed precision, and model saving.
    """

    def __init__(self, model, train_loader, val_loader, config=Config):
        """
        Args:
            model: The PyTorch model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            config: Configuration class.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.device

        # Move model to device
        self.model.to(self.device)

        # Loss function for Multi-Label Classification
        self.criterion = nn.BCEWithLogitsLoss()

    def train_one_epoch(self, epoch_idx, optimizer, scheduler, scaler):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        running_loss = 0.0
        total_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            # Zero gradients
            optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with autocast(enabled=self.config.use_fp16):
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

            # Backward Pass with Scaler
            scaler.scale(loss).backward()

            # Unscale and Clip Gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

            # Optimizer Step
            scaler.step(optimizer)
            scaler.update()

            # Scheduler Step
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / total_batches
        return avg_loss

    def validate(self):
        """
        Evaluates the model on the validation set.
        Returns:
            avg_val_loss: Average validation loss.
        """
        self.model.eval()
        running_loss = 0.0
        total_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(
                    self.device, non_blocking=True
                )
                labels = batch["labels"].to(self.device, non_blocking=True)

                # Forward pass (no autocast needed for eval usually, but consistent behavior is good)
                # Using float32 for validation stability
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                running_loss += loss.item()

        avg_val_loss = running_loss / total_batches
        return avg_val_loss

    def fit(self):
        """
        Main training loop.
        """
        print(f"Starting training on device: {self.device}")

        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler
        num_training_steps = len(self.train_loader) * self.config.epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Mixed Precision Scaler
        scaler = GradScaler(enabled=self.config.use_fp16)

        # Early Stopping & Model Saving variables
        best_val_loss = float("inf")
        patience = 3  # Not strictly used if epochs=1, but good practice
        patience_counter = 0

        for epoch in range(self.config.epochs):
            with Timer(f"Epoch {epoch + 1}"):
                # Train
                train_loss = self.train_one_epoch(epoch, optimizer, scheduler, scaler)

                # Validate
                val_loss = self.validate()

                print(f"Epoch {epoch + 1}/{self.config.epochs}")
                print(f"Train Loss: {train_loss}")
                print(f"Val Loss: {val_loss}")

                # Checkpoint
                if val_loss < best_val_loss:
                    print(
                        f"Validation loss improved from {best_val_loss} to {val_loss}. Saving model..."
                    )
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save model
                    torch.save(self.model.state_dict(), self.config.model_save_path)
                else:
                    patience_counter += 1
                    print(
                        f"Validation loss did not improve. Patience: {patience_counter}/{patience}"
                    )

                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        print("Training complete.")
