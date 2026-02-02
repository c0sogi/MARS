import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything, AverageMeter, LaplaceLogLikelihoodLoss
from library.data import get_dataloaders
from library.model import HiFiDACR


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Handles the training of one epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch in loader:
        # Unpack batch and move to device
        image_axial = batch["image_axial"].to(device)
        image_coronal = batch["image_coronal"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["target"].to(device)
        baseline_fvc = batch["baseline_fvc"].to(device)
        relative_week = batch["relative_week"].to(device)

        # Forward pass
        # The model expects: image_axial, image_coronal, tabular, baseline_fvc, relative_week
        outputs = model(
            image_axial=image_axial,
            image_coronal=image_coronal,
            tabular=tabular,
            baseline_fvc=baseline_fvc,
            relative_week=relative_week,
        )

        # Compute loss
        loss = criterion(outputs, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), image_axial.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns the average loss (negative metric) and the actual metric score.
    """
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            image_axial = batch["image_axial"].to(device)
            image_coronal = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            relative_week = batch["relative_week"].to(device)

            outputs = model(
                image_axial=image_axial,
                image_coronal=image_coronal,
                tabular=tabular,
                baseline_fvc=baseline_fvc,
                relative_week=relative_week,
            )

            loss = criterion(outputs, target)
            losses.update(loss.item(), image_axial.size(0))

    # The loss function is defined as -Metric.
    # So Metric = -Loss.
    metric_score = -losses.avg
    return losses.avg, metric_score


class Runner:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = LaplaceLogLikelihoodLoss().to(device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
        )

        self.best_loss = float("inf")
        self.patience_counter = 0

    def train(self, num_epochs):
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
            )

            val_loss, val_metric = validate(
                self.model, self.val_loader, self.criterion, self.device
            )

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Metric: {val_metric}"
            )

            # Early Stopping and Checkpointing
            if val_loss < self.best_loss:
                print(
                    f"Validation loss improved from {self.best_loss:.6f} to {val_loss:.6f}. Saving model..."
                )
                self.best_loss = val_loss
                self.patience_counter = 0

                # Save best model
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
            else:
                self.patience_counter += 1
                print(
                    f"Validation loss did not improve. Patience: {self.patience_counter}/{Config.PATIENCE}"
                )

                if self.patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Validation Loss: {self.best_loss:.6f}")
        print(f"Best model saved to: {Config.BEST_MODEL_PATH}")


def run_training(debug=False, num_epochs=None):
    """
    Main entry point to run the training pipeline.
    """
    # 1. Reproducibility
    seed_everything(Config.SEED)

    # 2. Data Preparation
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(debug=debug)

    # Calculate tabular input dimension dynamically
    # Get one batch to inspect tabular shape
    # tabular tensor shape is (Batch, Features)
    sample_batch = next(iter(train_loader))
    tab_dim = sample_batch["tabular"].shape[1]
    print(f"Detected Tabular Input Dimension: {tab_dim}")

    # 3. Model Initialization
    print("Initializing HiFi-DACR Model...")
    device = Config.DEVICE
    model = HiFiDACR(tab_input_dim=tab_dim)
    model = model.to(device)

    # 4. Training
    epochs_to_run = num_epochs if num_epochs is not None else Config.NUM_EPOCHS
    runner = Runner(model, train_loader, val_loader, device)
    runner.train(num_epochs=epochs_to_run)
