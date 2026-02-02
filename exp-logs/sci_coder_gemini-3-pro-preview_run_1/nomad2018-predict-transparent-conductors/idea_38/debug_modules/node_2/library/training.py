import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from library.config import (
    WORKING_DIR,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE,
    BATCH_SIZE,
    SEED,
    SUBMISSION_DIR,
)
from library.utils import (
    seed_everything,
    get_logger,
    AverageMeter,
    rmsle,
    save_checkpoint,
    save_submission,
    count_parameters,
)
from library.data_loader import get_dataloaders
from library.architecture import SSAWDSModel

# Set up logger
logger = get_logger("training", log_file=os.path.join(WORKING_DIR, "training.log"))


class Trainer:
    """
    Handles the training and validation loop for the SSA-WDS model.
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        patience=PATIENCE,
        save_dir=WORKING_DIR,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.patience = patience
        self.save_dir = save_dir
        self.best_val_loss = float("inf")
        self.counter = 0  # For early stopping

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        losses = AverageMeter("Loss", ":.4e")

        for batch_idx, batch_data in enumerate(train_loader):
            # Move data to device
            atomic_feats = batch_data["atomic"].to(self.device)
            atomic_mask = batch_data["atomic_mask"].to(self.device)
            global_feats = batch_data["global"].to(self.device)
            targets = batch_data["targets"].to(self.device)

            # Forward pass
            outputs = self.model(atomic_feats, atomic_mask, global_feats)
            loss = self.criterion(outputs, targets)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), targets.size(0))

        return losses.avg

    def validate(self, val_loader):
        self.model.eval()
        losses = AverageMeter("Val Loss", ":.4e")
        rmsle_meter = AverageMeter("RMSLE", ":.4f")

        with torch.no_grad():
            for batch_data in val_loader:
                atomic_feats = batch_data["atomic"].to(self.device)
                atomic_mask = batch_data["atomic_mask"].to(self.device)
                global_feats = batch_data["global"].to(self.device)
                targets = batch_data["targets"].to(self.device)

                outputs = self.model(atomic_feats, atomic_mask, global_feats)
                loss = self.criterion(outputs, targets)

                losses.update(loss.item(), targets.size(0))

                # Calculate RMSLE on original scale
                # Targets were log1p transformed, so we apply expm1
                preds_original = torch.expm1(outputs)
                targets_original = torch.expm1(targets)

                # Clamp predictions to be non-negative for physical validity
                preds_original = torch.clamp(preds_original, min=0.0)

                batch_rmsle = rmsle(targets_original, preds_original)
                rmsle_meter.update(batch_rmsle.item(), targets.size(0))

        return losses.avg, rmsle_meter.avg

    def train(self, train_loader, val_loader, epochs):
        logger.info(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, val_rmsle = self.validate(val_loader)

            # Learning rate scheduling
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)

            duration = time.time() - start_time

            logger.info(
                f"Epoch [{epoch}/{epochs}] "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val RMSLE: {val_rmsle:.6f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {duration:.1f}s"
            )

            # Checkpointing and Early Stopping
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.counter = 0
                logger.info(f"New best model found! Val Loss: {val_loss:.6f}")
            else:
                self.counter += 1

            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "best_val_loss": self.best_val_loss,
                    "optimizer": self.optimizer.state_dict(),
                },
                is_best,
                save_dir=self.save_dir,
            )

            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break


def generate_submission(model, test_loader, device, output_file="submission.csv"):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    ids_all = []
    preds_all = []

    logger.info("Generating predictions for test set...")

    with torch.no_grad():
        for batch_data in test_loader:
            atomic_feats = batch_data["atomic"].to(device)
            atomic_mask = batch_data["atomic_mask"].to(device)
            global_feats = batch_data["global"].to(device)
            ids = batch_data["ids"]

            outputs = model(atomic_feats, atomic_mask, global_feats)

            # Inverse transform: log1p -> expm1
            preds_original = torch.expm1(outputs)
            preds_original = torch.clamp(preds_original, min=0.0)

            ids_all.extend(ids)
            preds_all.append(preds_original.cpu().numpy())

    preds_all = np.concatenate(preds_all, axis=0)

    # Split into formation energy and bandgap energy
    formation_energies = preds_all[:, 0]
    bandgap_energies = preds_all[:, 1]

    save_submission(ids_all, formation_energies, bandgap_energies, filename=output_file)


def run_training(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    load_cached_data=True,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
):
    """
    Main execution function to setup data, model, and run training.
    """
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data, batch_size=batch_size
    )

    # 2. Initialize Model
    model = SSAWDSModel().to(device)
    logger.info(f"Model initialized with {count_parameters(model):,} parameters.")

    # 3. Setup Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    criterion = nn.MSELoss()

    # 4. Train
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        patience=PATIENCE,
        save_dir=WORKING_DIR,
    )

    trainer.train(train_loader, val_loader, epochs)

    # 5. Load Best Model for Inference
    best_model_path = os.path.join(WORKING_DIR, "model_best.pth.tar")
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        logger.warning("No best model checkpoint found. Using current model state.")

    # 6. Generate Submission
    generate_submission(model, test_loader, device)
