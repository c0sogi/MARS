import os
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from library.config import Config
from library.models import LocatorModel, InfillerModel
from library.utils import setup_logger


class Trainer:
    """
    Manages the training lifecycle for the Locator and Infiller models.
    Includes optimization, logging, evaluation, and checkpointing.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.logger = setup_logger(
            "Trainer", os.path.join(Config.WORKING_DIR, "training.log")
        )

        # Early stopping configuration
        self.patience = 2

    def train_locator(self, train_loader, val_loader):
        """
        Trains the Locator model (Stage 1).

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
        """
        self.logger.info("Starting Locator Training...")

        model = LocatorModel().to(self.device)

        # Optimizer and Scheduler
        optimizer = AdamW(
            model.parameters(), lr=Config.LR_LOCATOR, weight_decay=Config.WEIGHT_DECAY
        )

        total_steps = len(train_loader) * Config.EPOCHS
        warmup_steps = int(total_steps * Config.WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            # --- Training Phase ---
            model.train()
            total_train_loss = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Apply Label Smoothing for Binary Classification
                # y_ls = y * (1 - alpha) + 0.5 * alpha
                if Config.LABEL_SMOOTHING > 0:
                    labels = (
                        labels * (1.0 - Config.LABEL_SMOOTHING)
                        + 0.5 * Config.LABEL_SMOOTHING
                    )

                optimizer.zero_grad()

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # --- Validation Phase ---
            model.eval()
            total_val_loss = 0
            correct_preds = 0
            total_samples = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)  # Original binary labels

                    logits = model(input_ids, attention_mask)

                    # Calculate loss (using smoothed labels if configured, or raw?
                    # Usually val loss uses raw, but consistency helps. We'll use raw for metric, smoothed for loss if desired.
                    # Here we use raw labels for loss to measure true performance,
                    # but BCE requires float targets.
                    loss = criterion(logits, labels)
                    total_val_loss += loss.item()

                    # Calculate Top-1 Accuracy
                    # We want to see if the argmax of logits matches the index of the '1' in labels.
                    # Note: Labels are 0s and 1s.
                    pred_indices = torch.argmax(logits, dim=1)

                    # Find the target index (argmax works because there's usually one 1)
                    # We must handle rows where no label is set (truncation edge case)
                    target_indices = torch.argmax(labels, dim=1)
                    has_label = labels.sum(dim=1) > 0

                    matches = (pred_indices == target_indices) & has_label
                    correct_preds += matches.sum().item()
                    total_samples += has_label.sum().item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = correct_preds / total_samples if total_samples > 0 else 0.0

            self.logger.info(f"Epoch {epoch+1}/{Config.EPOCHS}")
            self.logger.info(f"Train Loss: {avg_train_loss}")
            self.logger.info(f"Val Loss: {avg_val_loss}")
            self.logger.info(f"Val Accuracy: {val_accuracy}")

            # --- Checkpointing & Early Stopping ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), Config.BEST_LOCATOR_PATH)
                self.logger.info(
                    f"New best Locator model saved to {Config.BEST_LOCATOR_PATH}"
                )
                patience_counter = 0
            else:
                patience_counter += 1
                self.logger.info(
                    f"No improvement. Patience: {patience_counter}/{self.patience}"
                )
                if patience_counter >= self.patience:
                    self.logger.info("Early stopping triggered.")
                    break

    def train_infiller(self, train_loader, val_loader):
        """
        Trains the In-Filler model (Stage 2).

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
        """
        self.logger.info("Starting Infiller Training...")

        model = InfillerModel().to(self.device)

        # Optimizer and Scheduler
        optimizer = AdamW(
            model.parameters(), lr=Config.LR_INFILLER, weight_decay=Config.WEIGHT_DECAY
        )

        total_steps = len(train_loader) * Config.EPOCHS
        warmup_steps = int(total_steps * Config.WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            # --- Training Phase ---
            model.train()
            total_train_loss = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()

                # Model outputs MaskedLMOutput, which contains loss
                outputs = model(input_ids, attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # --- Validation Phase ---
            model.eval()
            total_val_loss = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = model(input_ids, attention_mask, labels=labels)
                    loss = outputs.loss
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            perplexity = torch.exp(torch.tensor(avg_val_loss)).item()

            self.logger.info(f"Epoch {epoch+1}/{Config.EPOCHS}")
            self.logger.info(f"Train Loss: {avg_train_loss}")
            self.logger.info(f"Val Loss: {avg_val_loss}")
            self.logger.info(f"Val Perplexity: {perplexity}")

            # --- Checkpointing & Early Stopping ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), Config.BEST_INFILLER_PATH)
                self.logger.info(
                    f"New best Infiller model saved to {Config.BEST_INFILLER_PATH}"
                )
                patience_counter = 0
            else:
                patience_counter += 1
                self.logger.info(
                    f"No improvement. Patience: {patience_counter}/{self.patience}"
                )
                if patience_counter >= self.patience:
                    self.logger.info("Early stopping triggered.")
                    break
