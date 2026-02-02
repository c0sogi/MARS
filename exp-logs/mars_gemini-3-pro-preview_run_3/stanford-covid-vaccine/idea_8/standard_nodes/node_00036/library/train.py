import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed, MCRMSELoss, save_checkpoint, load_checkpoint
from library.data import get_dataloaders
from library.model import SpatiallyAugmentedBiGRU


class Trainer:
    def __init__(self, config, model, train_loader, val_loader):
        self.config = config
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.EPOCHS, eta_min=1e-6
        )

        # Loss Function (Unweighted MCRMSE on all targets)
        self.criterion = MCRMSELoss()

        # Metric calculation indices (reactivity, deg_Mg_pH10, deg_Mg_50C)
        # Indices in target list: 0, 1, 3
        self.scored_indices = [0, 1, 3]

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0

        for features, targets in self.train_loader:
            features = features.to(self.config.DEVICE)
            targets = targets.to(self.config.DEVICE)

            self.optimizer.zero_grad()

            # Forward pass
            # Model output: (Batch, 107, 5)
            preds = self.model(features)

            # Slice predictions to match target length (68)
            preds_sliced = preds[:, : self.config.PRED_LEN, :]

            # Compute loss on all 5 columns
            loss = self.criterion(preds_sliced, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * features.size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def validate(self):
        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for features, targets in self.val_loader:
                features = features.to(self.config.DEVICE)
                targets = targets.to(self.config.DEVICE)

                # Forward pass
                preds = self.model(features)

                # Slice to scored length (68)
                preds_sliced = preds[:, : self.config.PRED_LEN, :]

                all_preds.append(preds_sliced.cpu())
                all_targets.append(targets.cpu())

        # Concatenate all batches
        # Shape: (Total_Val_Samples, 68, 5)
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate MCRMSE specifically on scored columns
        # We calculate RMSE per column, then mean across the 3 scored columns
        mse_per_col = torch.mean((all_preds - all_targets) ** 2, dim=(0, 1))
        rmse_per_col = torch.sqrt(mse_per_col)

        # Select only the scored indices [0, 1, 3]
        scored_rmse = rmse_per_col[self.scored_indices]
        val_score = torch.mean(scored_rmse).item()

        return val_score

    def fit(self):
        best_score = float("inf")
        patience_counter = 0

        print(f"Starting training on {self.config.DEVICE}...")

        for epoch in range(self.config.EPOCHS):
            train_loss = self.train_epoch()
            val_score = self.validate()

            # Update scheduler
            self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{self.config.EPOCHS} | "
                f"Train Loss: {train_loss:.8f} | "
                f"Val MCRMSE (Scored): {val_score:.15f}"
            )

            # Early Stopping and Model Checkpointing
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                save_checkpoint(self.model.state_dict(), self.config.MODEL_PATH)
            else:
                patience_counter += 1
                if patience_counter >= self.config.PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        print(f"Training complete. Best Val Score: {best_score:.15f}")


def generate_submission(config, model, test_loader):
    print("Generating submission...")
    model.eval()

    all_preds = []

    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(config.DEVICE)

            # Forward pass
            # Output shape: (Batch, 107, 5)
            preds = model(features)
            all_preds.append(preds.cpu().numpy())

    # Concatenate: (N_Test, 107, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # Load Test Metadata to get IDs
    df_test = pd.read_parquet(config.TEST_PATH)
    ids = df_test["id"].values

    # Prepare submission data
    # We need to flatten: 240 samples * 107 positions = 25680 rows
    submission_ids = []
    submission_data = []

    for i, sample_id in enumerate(ids):
        sample_preds = all_preds[i]  # Shape (107, 5)
        for seqpos in range(config.SEQ_LEN):
            submission_ids.append(f"{sample_id}_{seqpos}")
            submission_data.append(sample_preds[seqpos])

    submission_data = np.array(submission_data)

    # Create DataFrame
    df_sub = pd.DataFrame(submission_data, columns=config.TARGET_COLS)
    df_sub.insert(0, "id_seqpos", submission_ids)

    # Save
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")


def run_training(debug=False, epochs=None):
    # Initialize Config
    config = Config(debug=debug)
    if epochs is not None:
        config.EPOCHS = epochs

    # Set Seeds
    set_seed(config.SEED)

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=True
    )

    # Initialize Model
    model = SpatiallyAugmentedBiGRU(config)

    # Initialize Trainer
    trainer = Trainer(config, model, train_loader, val_loader)

    # Run Training
    trainer.fit()

    # Load Best Model for Inference
    print("Loading best model for inference...")
    best_model_state = torch.load(
        config.MODEL_PATH, map_location=config.DEVICE, weights_only=False
    )
    model.load_state_dict(best_model_state)

    # Generate Submission
    generate_submission(config, model, test_loader)
