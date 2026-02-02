import os
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from library.config import Config
from library.utils import compute_metrics, get_device


class Trainer:
    def __init__(self, model, train_loader, val_loader):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The PyTorch model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = get_device()
        self.model.to(self.device)

        # Optimization components
        self.optimizer = self.get_optimizer()
        self.scaler = GradScaler()

        # Loss function (CrossEntropyLoss supports soft targets in recent PyTorch versions)
        self.criterion = nn.CrossEntropyLoss()

    def get_optimizer(self):
        """
        Configures the AdamW optimizer with differential learning rates.
        """
        # Separate backbone parameters from head parameters
        backbone_params = list(self.model.backbone.named_parameters())

        # Head parameters include the pooler and classifier
        head_params = list(self.model.pooler.named_parameters()) + list(
            self.model.classifier.named_parameters()
        )

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            # Backbone with weight decay
            {
                "params": [
                    p for n, p in backbone_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": Config.WEIGHT_DECAY,
                "lr": Config.LR_BACKBONE,
            },
            # Backbone without weight decay
            {
                "params": [
                    p for n, p in backbone_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": Config.LR_BACKBONE,
            },
            # Head with weight decay
            {
                "params": [
                    p for n, p in head_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": Config.WEIGHT_DECAY,
                "lr": Config.LR_HEAD,
            },
            # Head without weight decay
            {
                "params": [
                    p for n, p in head_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": Config.LR_HEAD,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        return optimizer

    def train_one_epoch(self, epoch_index):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        # Zero gradients at start of epoch
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            # Move batch to device
            input_ids_a = batch["input_ids_a"].to(self.device)
            attention_mask_a = batch["attention_mask_a"].to(self.device)
            input_ids_b = batch["input_ids_b"].to(self.device)
            attention_mask_b = batch["attention_mask_b"].to(self.device)
            scalar_features = batch["scalar_features"].to(self.device)
            labels = batch["labels"].to(self.device)

            batch_size = labels.size(0)

            # Mixed Precision Forward Pass
            with autocast():
                logits = self.model(
                    input_ids_a=input_ids_a,
                    attention_mask_a=attention_mask_a,
                    input_ids_b=input_ids_b,
                    attention_mask_b=attention_mask_b,
                    scalar_features=scalar_features,
                )
                loss = self.criterion(logits, labels)
                # Scale loss for gradient accumulation
                loss = loss / Config.ACCUMULATION_STEPS

            # Backward Pass
            self.scaler.scale(loss).backward()

            if (step + 1) % Config.ACCUMULATION_STEPS == 0:
                # Unscale gradients
                self.scaler.unscale_(self.optimizer)

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), Config.MAX_GRAD_NORM
                )

                # Optimizer Step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Track loss (multiply back by accumulation steps to get actual batch loss)
            running_loss += loss.item() * Config.ACCUMULATION_STEPS * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self):
        """
        Evaluates the model on the validation set.
        Returns:
            avg_loss (float): The average Log Loss on the validation set.
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids_a = batch["input_ids_a"].to(self.device)
                attention_mask_a = batch["attention_mask_a"].to(self.device)
                input_ids_b = batch["input_ids_b"].to(self.device)
                attention_mask_b = batch["attention_mask_b"].to(self.device)
                scalar_features = batch["scalar_features"].to(self.device)
                labels = batch["labels"].to(self.device)

                with autocast():
                    logits = self.model(
                        input_ids_a=input_ids_a,
                        attention_mask_a=attention_mask_a,
                        input_ids_b=input_ids_b,
                        attention_mask_b=attention_mask_b,
                        scalar_features=scalar_features,
                    )

                # Convert logits to probabilities
                probs = torch.softmax(logits, dim=1)

                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Compute metric using the library function
        score = compute_metrics(all_labels, all_preds)
        return score

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {Config.NUM_EPOCHS} epochs on {self.device}...")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss = self.validate()

            print(
                f"Epoch {epoch + 1}/{Config.NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.15f}"
            )

            # Early Stopping and Model Saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"  -> New best model saved to {Config.MODEL_SAVE_PATH}")
            else:
                patience_counter += 1
                print(
                    f"  -> Validation loss did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
                )
                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Validation Loss: {best_val_loss:.15f}")
