import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os
from library.config import Config
from library.utils import seed_everything, compute_spearman_metric


class Trainer:
    """
    Manages the training, validation, and prediction processes for the SiameseRoBERTa model.
    """

    def __init__(self, model, train_loader, val_loader, test_loader=None):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The PyTorch model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            test_loader (DataLoader, optional): DataLoader for test data.
        """
        seed_everything(Config.SEED)

        self.device = torch.device(Config.DEVICE)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Differential Learning Rates
        optimizer_parameters = [
            {"params": self.model.backbone.parameters(), "lr": Config.BACKBONE_LR},
            {"params": self.model.classifier.parameters(), "lr": Config.HEAD_LR},
        ]

        self.optimizer = optim.AdamW(
            optimizer_parameters, weight_decay=Config.WEIGHT_DECAY
        )

        # Model outputs Sigmoid, so we use BCELoss
        self.criterion = nn.BCELoss()

        # Early Stopping attributes
        self.best_score = -float("inf")
        self.patience_counter = 0
        self.best_model_state = None

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.

        Args:
            epoch_idx (int): Current epoch index.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            # Move batch to device
            q_input_ids = batch["q_input_ids"].to(self.device)
            q_attention_mask = batch["q_attention_mask"].to(self.device)
            a_input_ids = batch["a_input_ids"].to(self.device)
            a_attention_mask = batch["a_attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(
                q_input_ids, q_attention_mask, a_input_ids, a_attention_mask
            )

            # Compute loss
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            # Optimizer step
            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self):
        """
        Runs validation on the validation set.

        Returns:
            float: Mean Column-wise Spearman's Correlation Coefficient.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                q_input_ids = batch["q_input_ids"].to(self.device)
                q_attention_mask = batch["q_attention_mask"].to(self.device)
                a_input_ids = batch["a_input_ids"].to(self.device)
                a_attention_mask = batch["a_attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    q_input_ids, q_attention_mask, a_input_ids, a_attention_mask
                )

                all_preds.append(outputs.cpu())
                all_targets.append(labels.cpu())

        # Concatenate all batches
        if not all_preds:
            return 0.0

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute metric
        score = compute_spearman_metric(all_preds, all_targets)
        return score

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_score = self.validate()

            print(
                f"Epoch {epoch + 1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Spearman: {val_score}"
            )

            # Early Stopping Logic
            if val_score > self.best_score:
                print(
                    f"Validation score improved ({self.best_score} --> {val_score}). Saving model..."
                )
                self.best_score = val_score
                self.patience_counter = 0

                # Deep copy state dict to ensure integrity
                self.best_model_state = copy.deepcopy(self.model.state_dict())

                # Save to disk
                torch.save(self.best_model_state, Config.MODEL_SAVE_PATH)
            else:
                self.patience_counter += 1
                print(
                    f"No improvement. Patience: {self.patience_counter}/{Config.PATIENCE}"
                )

            if self.patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        # Load best model for future use (e.g., prediction)
        if self.best_model_state is not None:
            print("Loading best model weights...")
            self.model.load_state_dict(self.best_model_state)

    def predict(self, loader=None):
        """
        Generates predictions for a given loader (defaults to test_loader).

        Args:
            loader (DataLoader, optional): DataLoader to predict on. Defaults to self.test_loader.

        Returns:
            np.ndarray: Predictions of shape (N, 30).
        """
        if loader is None:
            loader = self.test_loader

        if loader is None:
            raise ValueError("No loader provided for prediction.")

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                q_input_ids = batch["q_input_ids"].to(self.device)
                q_attention_mask = batch["q_attention_mask"].to(self.device)
                a_input_ids = batch["a_input_ids"].to(self.device)
                a_attention_mask = batch["a_attention_mask"].to(self.device)

                outputs = self.model(
                    q_input_ids, q_attention_mask, a_input_ids, a_attention_mask
                )
                all_preds.append(outputs.cpu().numpy())

        if not all_preds:
            return np.array([])

        return np.concatenate(all_preds, axis=0)
