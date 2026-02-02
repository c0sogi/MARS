import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import (
    Standardizer,
    calculate_lmae,
    save_checkpoint,
    load_checkpoint,
    Logger,
)
from library.data import get_dataloaders
from library.model import SDIN


class Trainer:
    """
    Manages the training and validation lifecycle of the SDIN model.
    """

    def __init__(self, model, train_loader, val_loader, standardizer, logger):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.standardizer = standardizer
        self.logger = logger
        self.device = Config.DEVICE

        # Loss function: L1 Loss on standardized targets
        self.criterion = nn.L1Loss()

        # Optimizer: AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Cosine Annealing with Warm Restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=Config.T_0, T_mult=Config.T_MULT, eta_min=Config.ETA_MIN
        )

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch in self.train_loader:
            # Move data to device
            atom_types = batch["atom_types"].to(self.device)
            atom_coords = batch["atom_coords"].to(self.device)
            batch_index = batch["batch_index"].to(self.device)
            coupling_pairs = batch["coupling_pairs"].to(self.device)
            coupling_types = batch["coupling_types"].to(self.device)
            coupling_values = batch["coupling_values"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(
                atom_types=atom_types,
                atom_coords=atom_coords,
                batch_index=batch_index,
                coupling_pairs=coupling_pairs,
                coupling_types=coupling_types,
            )

            # Transform targets to standardized scale (Z-score)
            targets_std = self.standardizer.transform(coupling_values, coupling_types)

            # Compute loss
            loss = self.criterion(preds, targets_std)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )
            self.optimizer.step()

            running_loss += loss.item() * len(coupling_values)
            count += len(coupling_values)

        return running_loss / count if count > 0 else 0.0

    def validate(self):
        """
        Evaluates the model on the validation set using the competition metric (LMAE).
        Predictions are inverse-transformed to the original scale before metric calculation.
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        all_types = []

        with torch.no_grad():
            for batch in self.val_loader:
                atom_types = batch["atom_types"].to(self.device)
                atom_coords = batch["atom_coords"].to(self.device)
                batch_index = batch["batch_index"].to(self.device)
                coupling_pairs = batch["coupling_pairs"].to(self.device)
                coupling_types = batch["coupling_types"].to(self.device)
                coupling_values = batch["coupling_values"].to(self.device)

                # Forward pass
                preds_std = self.model(
                    atom_types=atom_types,
                    atom_coords=atom_coords,
                    batch_index=batch_index,
                    coupling_pairs=coupling_pairs,
                    coupling_types=coupling_types,
                )

                # Inverse transform to original physical scale
                preds_orig = self.standardizer.inverse_transform(
                    preds_std, coupling_types
                )

                all_preds.append(preds_orig.cpu())
                all_targets.append(coupling_values.cpu())
                all_types.append(coupling_types.cpu())

        # Concatenate all batches
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_types = torch.cat(all_types)

        # Calculate LMAE
        avg_lmae, per_type_lmae = calculate_lmae(all_preds, all_targets, all_types)
        return avg_lmae, per_type_lmae

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        best_score = float("inf")
        patience_counter = 0

        for epoch in range(1, Config.MAX_EPOCHS + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_score, val_per_type = self.validate()

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - start_time

            # Log metrics
            # Printing full precision for val_score as requested
            log_msg = (
                f"Epoch {epoch}/{Config.MAX_EPOCHS} | "
                f"Time: {elapsed:.1f}s | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss (Std L1): {train_loss:.6f} | "
                f"Val LMAE: {val_score}"
            )
            self.logger.log(log_msg)

            # Check Early Stopping
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_score,
                    Config.MODEL_SAVE_PATH,
                )
                self.logger.log(f"  New Best Score! Model saved.")
            else:
                patience_counter += 1
                self.logger.log(
                    f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
                )

            if patience_counter >= Config.PATIENCE:
                self.logger.log("Early stopping triggered.")
                break

        return best_score


def predict_test(model, test_loader, standardizer, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    ids_list = []
    preds_list = []
    device = Config.DEVICE

    print("Generating predictions for test set...")
    with torch.no_grad():
        for batch in test_loader:
            atom_types = batch["atom_types"].to(device)
            atom_coords = batch["atom_coords"].to(device)
            batch_index = batch["batch_index"].to(device)
            coupling_pairs = batch["coupling_pairs"].to(device)
            coupling_types = batch["coupling_types"].to(device)
            coupling_ids = batch["coupling_ids"]

            # Forward pass
            preds_std = model(
                atom_types=atom_types,
                atom_coords=atom_coords,
                batch_index=batch_index,
                coupling_pairs=coupling_pairs,
                coupling_types=coupling_types,
            )

            # Inverse transform
            preds_orig = standardizer.inverse_transform(preds_std, coupling_types)

            ids_list.append(coupling_ids.cpu().numpy())
            preds_list.append(preds_orig.cpu().numpy())

    # Flatten and save
    if len(ids_list) > 0:
        all_ids = np.concatenate(ids_list)
        all_preds = np.concatenate(preds_list)

        df = pd.DataFrame({"id": all_ids, "scalar_coupling_constant": all_preds})
    else:
        print("Warning: No predictions generated. Test loader yielded no batches.")
        df = pd.DataFrame(columns=["id", "scalar_coupling_constant"])

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(load_cached_data=True):
    """
    Main entry point for the training pipeline.
    """
    # Initialize Logger
    logger = Logger(os.path.join(Config.WORKING_DIR, "train.log"))
    logger.log("Starting training run...")

    # Load Data
    train_loader, val_loader, test_loader, standardizer = get_dataloaders(
        load_cached_data=load_cached_data
    )
    logger.log("Data loaders ready.")

    # Initialize Model
    model = SDIN().to(Config.DEVICE)
    logger.log(f"Model initialized on {Config.DEVICE}.")

    # Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader, standardizer, logger)

    # Run Training
    best_score = trainer.fit()
    logger.log(f"Training completed. Best Val LMAE: {best_score}")

    # Load Best Model for Inference
    logger.log("Loading best model for inference...")
    load_checkpoint(model, path=Config.MODEL_SAVE_PATH, device=Config.DEVICE)

    # Generate Submission
    predict_test(model, test_loader, standardizer, Config.SUBMISSION_PATH)
