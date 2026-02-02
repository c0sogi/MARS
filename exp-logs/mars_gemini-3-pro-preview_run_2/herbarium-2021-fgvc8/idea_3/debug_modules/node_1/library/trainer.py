import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

from library.config import Config
from library.utils import MetricMonitor, save_checkpoint, load_checkpoint, set_seed
from library.model import HierarchicalConvNeXt


class Trainer:
    """
    Trainer class for the Hierarchical ConvNeXt model.
    Encapsulates training loops, validation, and inference.
    Supports the two-stage training strategy defined in the Config.
    """

    def __init__(self, model, device=Config.DEVICE):
        self.model = model.to(device)
        self.device = device
        self.best_score = 0.0
        # Criterion with label smoothing as per Config
        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    def freeze_backbone(self, freeze=True):
        """Wrapper to freeze/unfreeze the backbone."""
        self.model.freeze_backbone(freeze)

    def unfreeze_all(self):
        """Unfreezes all parameters in the model."""
        self.model.freeze_backbone(False)
        self.model.freeze_auxiliary_heads(False)

    def freeze_auxiliary_heads(self, freeze=True):
        """Wrapper to freeze/unfreeze auxiliary heads."""
        self.model.freeze_auxiliary_heads(freeze)

    def train_one_epoch(self, train_loader, optimizer, epoch, stage):
        """
        Trains the model for one epoch.

        Args:
            stage (int): 1 for Representation Learning (all losses),
                         2 for Classifier Re-balancing (species loss only).
        """
        self.model.train()
        metric_monitor = MetricMonitor()

        for batch in train_loader:
            # Move data to device
            images = batch[0].to(self.device, non_blocking=True)
            species_targets = batch[1].to(self.device, non_blocking=True)
            family_targets = batch[2].to(self.device, non_blocking=True)
            order_targets = batch[3].to(self.device, non_blocking=True)

            # Forward pass
            outputs = self.model(images)

            # Calculate Loss
            loss_species = self.criterion(outputs["species"], species_targets)

            if stage == 1:
                # Stage 1: Multi-task loss (Species + Family + Order)
                loss_family = self.criterion(outputs["family"], family_targets)
                loss_order = self.criterion(outputs["order"], order_targets)
                loss = loss_species + loss_family + loss_order
            else:
                # Stage 2: Only fine-tune species head
                loss = loss_species

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            metric_monitor.update("Loss", loss.item())

        print(f"Epoch {epoch} (Stage {stage}) Training Results: {metric_monitor}")

    def validate(self, val_loader):
        """
        Validates the model on the validation set.
        Returns the Macro F1 score.
        """
        self.model.eval()
        metric_monitor = MetricMonitor()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch[0].to(self.device, non_blocking=True)
                species_targets = batch[1].to(self.device, non_blocking=True)

                outputs = self.model(images)

                # Calculate validation loss (Species only for metric tracking)
                loss = self.criterion(outputs["species"], species_targets)
                metric_monitor.update("Loss", loss.item())

                # Collect predictions for F1 score
                preds = torch.argmax(outputs["species"], dim=1).cpu().numpy()
                targets = species_targets.cpu().numpy()

                all_preds.extend(preds)
                all_targets.extend(targets)

        # Calculate Macro F1
        f1 = f1_score(all_targets, all_preds, average="macro")
        metric_monitor.update("Macro F1", f1)

        print(f"Validation Results: {metric_monitor}")
        return f1

    def predict(self, test_loader, output_path=Config.SUBMISSION_PATH):
        """
        Generates predictions for the test set and saves to CSV.
        """
        self.model.eval()
        ids = []
        predictions = []

        print("Starting inference on test set...")
        with torch.no_grad():
            for batch in test_loader:
                images = batch[0].to(self.device, non_blocking=True)
                image_ids = batch[1]  # Keep as CPU tensor/numpy

                outputs = self.model(images)
                preds = torch.argmax(outputs["species"], dim=1).cpu().numpy()

                ids.extend(image_ids.numpy())
                predictions.extend(preds)

        # Create submission DataFrame
        df = pd.DataFrame({"Id": ids, "Predicted": predictions})

        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")

    def fit_stage1(self, train_loader, val_loader):
        """
        Executes Stage 1: Representation Learning (Instance-Balanced).
        Trains backbone and all heads.
        """
        print("\n=== Starting Stage 1: Representation Learning ===")

        # Configuration
        self.unfreeze_all()
        optimizer = optim.AdamW(self.model.parameters(), lr=Config.STAGE1_LR)
        # Optional: Scheduler could be added here (e.g., OneCycleLR)

        for epoch in range(1, Config.STAGE1_EPOCHS + 1):
            self.train_one_epoch(train_loader, optimizer, epoch, stage=1)

            score = self.validate(val_loader)

            # Save checkpoint if best
            if score > self.best_score:
                self.best_score = score
                save_checkpoint(
                    self.model, optimizer, epoch, score, Config.STAGE1_CHECKPOINT
                )
                # Also save as best model for now
                save_checkpoint(
                    self.model, optimizer, epoch, score, Config.BEST_MODEL_PATH
                )
                print(f"New best score in Stage 1: {score}")

    def fit_stage2(self, train_loader, val_loader):
        """
        Executes Stage 2: Classifier Re-balancing (Class-Balanced).
        Freezes backbone and aux heads, fine-tunes species head.
        """
        print("\n=== Starting Stage 2: Classifier Re-balancing ===")

        # Load best model from Stage 1
        print(f"Loading best model from Stage 1: {Config.BEST_MODEL_PATH}")
        epoch_start, best_score = load_checkpoint(
            self.model, Config.BEST_MODEL_PATH, device=self.device
        )
        self.best_score = best_score

        # Freeze backbone and auxiliary heads
        self.freeze_backbone(True)
        self.freeze_auxiliary_heads(True)

        # Optimizer for trainable parameters only
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = optim.AdamW(trainable_params, lr=Config.STAGE2_LR)

        for epoch in range(1, Config.STAGE2_EPOCHS + 1):
            # Pass stage=2 to train_one_epoch
            self.train_one_epoch(train_loader, optimizer, epoch, stage=2)

            score = self.validate(val_loader)

            if score > self.best_score:
                self.best_score = score
                save_checkpoint(
                    self.model, optimizer, epoch, score, Config.BEST_MODEL_PATH
                )
                print(f"New best score in Stage 2: {score}")
