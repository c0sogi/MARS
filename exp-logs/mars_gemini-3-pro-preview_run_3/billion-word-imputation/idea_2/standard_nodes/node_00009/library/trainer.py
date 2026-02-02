import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import setup_logger
from library.models import PointerLocator, get_filler_model


class Trainer:
    """
    Manages the training lifecycle for the Locator and Filler models.
    """

    def __init__(self, train_loader, val_loader):
        """
        Initialize the Trainer with data loaders and models.

        Args:
            train_loader (DataLoader): Loader for training data.
            val_loader (DataLoader): Loader for validation data.
        """
        self.logger = setup_logger(
            "Trainer", os.path.join(Config.OUTPUT_DIR, "execution.log")
        )
        self.device = Config.DEVICE
        self.train_loader = train_loader
        self.val_loader = val_loader

        # -------------------------------------------------------
        # Initialize Models
        # -------------------------------------------------------
        self.logger.info(f"Initializing models on device: {self.device}")
        self.locator = PointerLocator(model_name=Config.MODEL_NAME).to(self.device)
        self.filler = get_filler_model(model_name=Config.MODEL_NAME).to(self.device)

        # -------------------------------------------------------
        # Optimization Setup
        # -------------------------------------------------------
        # We define optimizers and schedulers in the train method or lazily
        # to ensure they are reset for each phase if needed.
        # However, initializing them here is fine for a sequential run.

        # Locator Optimizer
        self.locator_optimizer = AdamW(
            self.locator.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Filler Optimizer
        self.filler_optimizer = AdamW(
            self.filler.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss function for Locator (Filler uses internal loss)
        self.locator_criterion = nn.CrossEntropyLoss()

    def get_scheduler(self, optimizer, num_training_steps):
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * Config.WARMUP_RATIO),
            num_training_steps=num_training_steps,
        )

    # =========================================================================
    # LOCATOR TRAINING
    # =========================================================================

    def train_locator_epoch(self, epoch_idx, scheduler):
        self.locator.train()
        total_loss = 0.0
        start_time = time.time()

        self.logger.info(f"Starting Locator training epoch {epoch_idx + 1}")

        for step, batch in enumerate(self.train_loader):
            # Move inputs to device
            input_ids = batch["locator_input_ids"].to(self.device)
            attention_mask = batch["locator_attention_mask"].to(self.device)
            labels = batch["locator_labels"].to(self.device)

            # Forward pass
            self.locator_optimizer.zero_grad()
            logits = self.locator(input_ids, attention_mask)

            # Compute loss
            loss = self.locator_criterion(logits, labels)

            # Backward pass
            loss.backward()
            self.locator_optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % Config.LOG_INTERVAL == 0:
                avg_loss = total_loss / (step + 1)
                elapsed = time.time() - start_time
                self.logger.info(
                    f"Locator Epoch {epoch_idx + 1} | Step {step + 1}/{len(self.train_loader)} | "
                    f"Loss: {loss.item()} | Avg Loss: {avg_loss} | Time: {elapsed:.2f}s"
                )

        return total_loss / len(self.train_loader)

    def validate_locator(self):
        self.locator.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        self.logger.info("Starting Locator validation...")

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["locator_input_ids"].to(self.device)
                attention_mask = batch["locator_attention_mask"].to(self.device)
                labels = batch["locator_labels"].to(self.device)

                logits = self.locator(input_ids, attention_mask)
                loss = self.locator_criterion(logits, labels)

                total_loss += loss.item()

                # Calculate accuracy
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_samples

        self.logger.info(
            f"Locator Validation | Loss: {avg_loss} | Accuracy: {accuracy}"
        )
        return avg_loss, accuracy

    # =========================================================================
    # FILLER TRAINING
    # =========================================================================

    def train_filler_epoch(self, epoch_idx, scheduler):
        self.filler.train()
        total_loss = 0.0
        start_time = time.time()

        self.logger.info(f"Starting Filler training epoch {epoch_idx + 1}")

        for step, batch in enumerate(self.train_loader):
            # Move inputs to device
            input_ids = batch["filler_input_ids"].to(self.device)
            attention_mask = batch["filler_attention_mask"].to(self.device)
            labels = batch["filler_labels"].to(self.device)

            # Forward pass
            self.filler_optimizer.zero_grad()
            outputs = self.filler(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss

            # Backward pass
            loss.backward()
            self.filler_optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % Config.LOG_INTERVAL == 0:
                avg_loss = total_loss / (step + 1)
                elapsed = time.time() - start_time
                self.logger.info(
                    f"Filler Epoch {epoch_idx + 1} | Step {step + 1}/{len(self.train_loader)} | "
                    f"Loss: {loss.item()} | Avg Loss: {avg_loss} | Time: {elapsed:.2f}s"
                )

        return total_loss / len(self.train_loader)

    def validate_filler(self):
        self.filler.eval()
        total_loss = 0.0

        self.logger.info("Starting Filler validation...")

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["filler_input_ids"].to(self.device)
                attention_mask = batch["filler_attention_mask"].to(self.device)
                labels = batch["filler_labels"].to(self.device)

                outputs = self.filler(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        self.logger.info(f"Filler Validation | Loss: {avg_loss}")
        return avg_loss

    # =========================================================================
    # MAIN TRAINING LOOP
    # =========================================================================

    def train(self):
        """
        Executes the two-stage training process:
        1. Train Locator
        2. Train Filler
        """

        # -------------------------------------------------------
        # Phase 1: Train Locator
        # -------------------------------------------------------
        self.logger.info("=== Phase 1: Training Locator ===")

        num_training_steps = len(self.train_loader) * Config.EPOCHS
        locator_scheduler = self.get_scheduler(
            self.locator_optimizer, num_training_steps
        )

        best_locator_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_locator_epoch(epoch, locator_scheduler)
            val_loss, val_acc = self.validate_locator()

            self.logger.info(
                f"Epoch {epoch + 1} Summary (Locator) | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Acc: {val_acc}"
            )

            # Checkpoint & Early Stopping
            if val_loss < best_locator_val_loss:
                best_locator_val_loss = val_loss
                patience_counter = 0
                self.logger.info(
                    f"Validation loss improved. Saving Locator model to {Config.BEST_LOCATOR_PATH}"
                )
                torch.save(self.locator.state_dict(), Config.BEST_LOCATOR_PATH)
            else:
                patience_counter += 1
                self.logger.info(
                    f"Validation loss did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
                )
                if patience_counter >= Config.PATIENCE:
                    self.logger.info("Early stopping triggered for Locator.")
                    break

        # Free up memory before next phase
        torch.cuda.empty_cache()

        # -------------------------------------------------------
        # Phase 2: Train Filler
        # -------------------------------------------------------
        self.logger.info("=== Phase 2: Training Filler ===")

        filler_scheduler = self.get_scheduler(self.filler_optimizer, num_training_steps)

        best_filler_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_filler_epoch(epoch, filler_scheduler)
            val_loss = self.validate_filler()

            self.logger.info(
                f"Epoch {epoch + 1} Summary (Filler) | Train Loss: {train_loss} | Val Loss: {val_loss}"
            )

            # Checkpoint & Early Stopping
            if val_loss < best_filler_val_loss:
                best_filler_val_loss = val_loss
                patience_counter = 0
                self.logger.info(
                    f"Validation loss improved. Saving Filler model to {Config.BEST_FILLER_PATH}"
                )
                torch.save(self.filler.state_dict(), Config.BEST_FILLER_PATH)
            else:
                patience_counter += 1
                self.logger.info(
                    f"Validation loss did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
                )
                if patience_counter >= Config.PATIENCE:
                    self.logger.info("Early stopping triggered for Filler.")
                    break

        self.logger.info("Training complete.")
