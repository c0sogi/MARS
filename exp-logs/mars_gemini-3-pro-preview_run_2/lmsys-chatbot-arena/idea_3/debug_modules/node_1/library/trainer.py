import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np

from library.config import Config
from library.utils import save_checkpoint, compute_score


class Trainer:
    """
    Manages the training, validation, and prediction loops for the Siamese DeBERTa model.
    """

    def __init__(self, model, device=None, patience=3):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            device (torch.device, optional): Device to run training on. Defaults to Config.DEVICE.
            patience (int): Number of epochs to wait for improvement before early stopping.
        """
        self.model = model
        self.device = device if device else Config.DEVICE
        self.model.to(self.device)
        self.patience = patience
        self.best_val_loss = float("inf")
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_loader, val_loader, epochs=Config.EPOCHS):
        """
        Executes the training pipeline with Early Stopping.

        Args:
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            epochs (int): Maximum number of epochs to train.
        """
        # Initialize Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize Scheduler
        num_training_steps = len(train_loader) * epochs
        num_warmup_steps = int(num_training_steps * Config.NUM_WARMUP_STEPS_RATIO)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        print(f"Starting training for {epochs} epochs...")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Training Phase
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)

            # Validation Phase
            val_loss, val_score = self._validate(val_loader)

            # Print metrics (Full precision)
            print(
                f"Epoch {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Log Loss: {val_score}"
            )

            # Early Stopping and Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(self.model, optimizer, scheduler, epoch, val_loss)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

    def _train_epoch(self, loader, optimizer, scheduler):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            # Move inputs to device
            input_ids_a = batch["input_ids_a"].to(self.device)
            attention_mask_a = batch["attention_mask_a"].to(self.device)
            input_ids_b = batch["input_ids_b"].to(self.device)
            attention_mask_b = batch["attention_mask_b"].to(self.device)
            meta_features = batch["meta_features"].to(self.device)
            targets = batch["target"].to(self.device)

            optimizer.zero_grad()

            # Forward pass
            logits = self.model(
                input_ids_a,
                attention_mask_a,
                input_ids_b,
                attention_mask_b,
                meta_features,
            )

            loss = self.criterion(logits, targets)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _validate(self, loader):
        """
        Evaluates the model on the validation set.

        Returns:
            tuple: (average_loss, log_loss_score)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                input_ids_a = batch["input_ids_a"].to(self.device)
                attention_mask_a = batch["attention_mask_a"].to(self.device)
                input_ids_b = batch["input_ids_b"].to(self.device)
                attention_mask_b = batch["attention_mask_b"].to(self.device)
                meta_features = batch["meta_features"].to(self.device)
                targets = batch["target"].to(self.device)

                logits = self.model(
                    input_ids_a,
                    attention_mask_a,
                    input_ids_b,
                    attention_mask_b,
                    meta_features,
                )

                loss = self.criterion(logits, targets)
                total_loss += loss.item()

                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=1)

                all_preds.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        avg_loss = total_loss / len(loader)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        score = compute_score(all_targets, all_preds)
        return avg_loss, score

    def predict(self, loader):
        """
        Generates predictions for a dataset (e.g., test set).

        Args:
            loader (DataLoader): DataLoader for the test set.

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, 3).
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                input_ids_a = batch["input_ids_a"].to(self.device)
                attention_mask_a = batch["attention_mask_a"].to(self.device)
                input_ids_b = batch["input_ids_b"].to(self.device)
                attention_mask_b = batch["attention_mask_b"].to(self.device)
                meta_features = batch["meta_features"].to(self.device)

                logits = self.model(
                    input_ids_a,
                    attention_mask_a,
                    input_ids_b,
                    attention_mask_b,
                    meta_features,
                )

                probs = torch.softmax(logits, dim=1)
                all_preds.append(probs.cpu().numpy())

        return np.concatenate(all_preds)
