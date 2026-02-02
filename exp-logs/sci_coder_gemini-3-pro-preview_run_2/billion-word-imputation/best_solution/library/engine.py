import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm
import numpy as np

from library.config import Config
from library.utils import get_logger
from library.models import ModelFactory

logger = get_logger("engine")


class Trainer:
    """
    Manages the training of the Locator and In-Filler models.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.locator_checkpoint = Config.LOCATOR_MODEL_PATH
        self.infiller_checkpoint = Config.INFILLER_MODEL_PATH

        # Early stopping parameters
        self.patience = 2
        self.min_delta = 1e-4

    def train_locator(self, train_loader, val_loader):
        """
        Trains the Syntactic Locator model (DeBERTa-v3).
        Objective: Predict the probability of a gap following each token.
        Loss: BCE with Label Smoothing.
        """
        logger.info("Starting Locator training...")

        model = ModelFactory.get_locator_model()
        model.to(self.device)

        # Optimizer
        optimizer = AdamW(model.parameters(), lr=Config.LOCATOR_LR)

        # Scheduler
        total_steps = len(train_loader) * Config.LOCATOR_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        # Loss function: BCEWithLogitsLoss
        # We handle label smoothing manually on the targets
        criterion = nn.BCEWithLogitsLoss()
        smoothing = Config.LOCATOR_LABEL_SMOOTHING

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.LOCATOR_EPOCHS):
            # --- Training ---
            model.train()
            train_loss_sum = 0.0

            # Use tqdm for progress tracking if desired, but keeping logs clean as requested
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)  # Shape: (batch, seq_len)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)  # Shape: (batch, seq_len)

                # Apply Label Smoothing to targets
                # Target 1 -> 1 - 0.5*eps
                # Target 0 -> 0.5*eps
                # Formula: y_ls = y * (1 - eps) + 0.5 * eps
                smoothed_labels = labels * (1.0 - smoothing) + 0.5 * smoothing

                # Compute loss (masking padding tokens if necessary, but attention_mask handles input)
                # We only care about loss on active tokens.
                # Flatten for loss computation
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1)
                active_labels = smoothed_labels.view(-1)

                loss = criterion(active_logits[active_loss], active_labels[active_loss])

                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss_sum += loss.item()

            avg_train_loss = train_loss_sum / len(train_loader)

            # --- Validation ---
            avg_val_loss = self._validate_locator(model, val_loader, criterion)

            logger.info(
                f"Epoch {epoch + 1}/{Config.LOCATOR_EPOCHS} | "
                f"Train Loss: {avg_train_loss} | "
                f"Val Loss: {avg_val_loss}"
            )

            # --- Checkpointing & Early Stopping ---
            if avg_val_loss < best_val_loss - self.min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), self.locator_checkpoint)
                logger.info(
                    f"New best Locator model saved to {self.locator_checkpoint}"
                )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping triggered for Locator.")
                    break

        # Free memory
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    def _validate_locator(self, model, val_loader, criterion):
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)

                # For validation, we can compute loss against raw labels or smoothed.
                # Usually standard BCE against raw labels is better for true metric,
                # but to be consistent with training objective, we use raw labels here
                # or just the same criterion without smoothing if we want pure accuracy proxy.
                # Let's stick to the training objective (smoothed) or raw.
                # Using raw labels for validation metric is cleaner.

                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1)
                active_labels = labels.view(-1)  # Raw labels

                loss = criterion(active_logits[active_loss], active_labels[active_loss])
                val_loss_sum += loss.item()

        return val_loss_sum / len(val_loader)

    def train_infiller(self, train_loader, val_loader):
        """
        Trains the Semantic In-Filler model (RoBERTa-Large).
        Objective: Predict the masked word.
        Loss: Cross Entropy on the vocabulary.
        """
        logger.info("Starting In-Filler training...")

        model = ModelFactory.get_infiller_model()
        model.to(self.device)

        # We need the tokenizer to identify the mask token ID to extract logits
        tokenizer = AutoTokenizer.from_pretrained(Config.INFILLER_MODEL_NAME)
        mask_token_id = tokenizer.mask_token_id

        optimizer = AdamW(model.parameters(), lr=Config.INFILLER_LR)

        total_steps = len(train_loader) * Config.INFILLER_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.INFILLER_EPOCHS):
            # --- Training ---
            model.train()
            train_loss_sum = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                target_ids = batch["labels"].to(
                    self.device
                )  # Scalar ID of the missing word

                optimizer.zero_grad()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # (batch, seq_len, vocab_size)

                # Extract logits at the mask position
                # We assume exactly one mask token per sequence based on dataset logic
                mask_mask = input_ids == mask_token_id

                # Check if we found masks (safety)
                if not mask_mask.any():
                    continue

                # Select logits corresponding to the mask
                # mask_mask is (batch, seq_len). logits is (batch, seq_len, vocab).
                # We want (batch, vocab)

                # Identify which rows actually have a mask token (it might be truncated)
                rows_with_mask = mask_mask.any(dim=1)

                if not rows_with_mask.any():
                    continue

                # Filter logits: this flattens the batch dimension to only valid masks
                selected_logits = logits[
                    mask_mask
                ]  # Shape: (num_valid_masks, vocab_size)

                # Filter targets to match the valid rows
                selected_targets = target_ids[rows_with_mask]

                # Calculate loss
                loss = criterion(selected_logits, selected_targets)

                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss_sum += loss.item()

            avg_train_loss = train_loss_sum / len(train_loader)

            # --- Validation ---
            avg_val_loss = self._validate_infiller(
                model, val_loader, criterion, mask_token_id
            )

            logger.info(
                f"Epoch {epoch + 1}/{Config.INFILLER_EPOCHS} | "
                f"Train Loss: {avg_train_loss} | "
                f"Val Loss: {avg_val_loss}"
            )

            # --- Checkpointing & Early Stopping ---
            if avg_val_loss < best_val_loss - self.min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), self.infiller_checkpoint)
                logger.info(
                    f"New best In-Filler model saved to {self.infiller_checkpoint}"
                )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping triggered for In-Filler.")
                    break

        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    def _validate_infiller(self, model, val_loader, criterion, mask_token_id):
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                target_ids = batch["labels"].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                mask_mask = input_ids == mask_token_id
                rows_with_mask = mask_mask.any(dim=1)

                if not rows_with_mask.any():
                    continue

                selected_logits = logits[mask_mask]
                selected_targets = target_ids[rows_with_mask]

                loss = criterion(selected_logits, selected_targets)

                val_loss_sum += loss.item()

        return val_loss_sum / len(val_loader)
