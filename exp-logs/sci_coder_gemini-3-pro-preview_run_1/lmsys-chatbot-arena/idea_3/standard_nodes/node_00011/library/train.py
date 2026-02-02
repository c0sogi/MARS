import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np

from library.config import Config
from library.utils import seed_everything, get_device, AverageMeter, ensure_directories
from library.dataset import get_dataloaders
from library.model import SiameseDebertaWithScalars


class Trainer:
    """
    Manages the training lifecycle of the Siamese DeBERTa model.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize the Trainer.

        Args:
            debug (bool): If True, runs with a smaller dataset for debugging.
        """
        self.debug = debug
        self.device = get_device()

        # Ensure reproducibility
        seed_everything(Config.SEED)

        # Ensure output directories exist
        ensure_directories()

        # Load Data
        print(f"Initializing DataLoaders (Debug={self.debug})...")
        self.train_loader, self.val_loader, _ = get_dataloaders(debug=self.debug)

        # Initialize Model
        print(f"Initializing Model: {Config.MODEL_NAME}...")
        self.model = SiameseDebertaWithScalars()
        self.model.to(self.device)

        # Loss Function
        # CrossEntropyLoss supports soft targets (probabilities) directly in PyTorch
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer and Scheduler
        self.optimizer = self._configure_optimizer()

        # Total training steps
        num_training_steps = len(self.train_loader) * Config.NUM_EPOCHS
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps,
        )

    def _configure_optimizer(self):
        """
        Configures AdamW with differential learning rates.
        - Lower LR for the pre-trained backbone.
        - Higher LR for the new classification head.
        """
        # Separate parameters into backbone and head
        backbone_params = []
        head_params = []

        # We also separate weight decay (no decay for bias/LayerNorm)
        no_decay = ["bias", "LayerNorm.weight"]

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "backbone" in name:
                backbone_params.append((name, param))
            else:
                head_params.append((name, param))

        optimizer_grouped_parameters = [
            # Backbone with Weight Decay
            {
                "params": [
                    p for n, p in backbone_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": Config.WEIGHT_DECAY,
                "lr": Config.LR_BACKBONE,
            },
            # Backbone without Weight Decay
            {
                "params": [
                    p for n, p in backbone_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": Config.LR_BACKBONE,
            },
            # Head with Weight Decay
            {
                "params": [
                    p for n, p in head_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": Config.WEIGHT_DECAY,
                "lr": Config.LR_HEAD,
            },
            # Head without Weight Decay
            {
                "params": [
                    p for n, p in head_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": Config.LR_HEAD,
            },
        ]

        return AdamW(optimizer_grouped_parameters)

    def train_epoch(self, epoch_idx):
        """
        Runs one training epoch.
        """
        self.model.train()
        losses = AverageMeter()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            input_ids_a = batch["input_ids_a"].to(self.device)
            attention_mask_a = batch["attention_mask_a"].to(self.device)
            input_ids_b = batch["input_ids_b"].to(self.device)
            attention_mask_b = batch["attention_mask_b"].to(self.device)
            scalar_features = batch["scalar_features"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            logits = self.model(
                input_ids_a=input_ids_a,
                attention_mask_a=attention_mask_a,
                input_ids_b=input_ids_b,
                attention_mask_b=attention_mask_b,
                scalar_features=scalar_features,
            )

            # Calculate Loss
            loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            losses.update(loss.item(), labels.size(0))

        return losses.avg

    def validate(self):
        """
        Runs validation loop.
        """
        self.model.eval()
        losses = AverageMeter()

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids_a = batch["input_ids_a"].to(self.device)
                attention_mask_a = batch["attention_mask_a"].to(self.device)
                input_ids_b = batch["input_ids_b"].to(self.device)
                attention_mask_b = batch["attention_mask_b"].to(self.device)
                scalar_features = batch["scalar_features"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(
                    input_ids_a=input_ids_a,
                    attention_mask_a=attention_mask_a,
                    input_ids_b=input_ids_b,
                    attention_mask_b=attention_mask_b,
                    scalar_features=scalar_features,
                )

                loss = self.criterion(logits, labels)
                losses.update(loss.item(), labels.size(0))

        return losses.avg

    def train(self):
        """
        Main training loop with Early Stopping.
        """
        print("Starting training...")
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")

            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss}")

            # Validate
            val_loss = self.validate()
            # Print full precision as requested
            print(f"Validation Loss: {val_loss}")

            # Early Stopping and Model Saving
            if val_loss < best_val_loss:
                print(
                    f"Validation loss improved from {best_val_loss} to {val_loss}. Saving model..."
                )
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"Validation loss did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
                )
                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Validation Loss: {best_val_loss}")


def run_training(debug: bool = False):
    """
    Entry point function to initialize Trainer and start training.

    Args:
        debug (bool): Whether to run in debug mode (fewer samples).
    """
    trainer = Trainer(debug=debug)
    trainer.train()
