import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from library.config import Config
from library.model import HGANet
from library.data import get_dataloaders, COUPLING_TYPES
from library.utils import set_seed, compute_log_mae


class Trainer:
    """
    Trainer class for HGA-Net.
    Manages training, validation, and checkpointing.
    """

    def __init__(
        self, config: Config, model: nn.Module, train_loader, val_loader, standardizer
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.standardizer = standardizer

        self.device = config.DEVICE
        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Calculate total steps for OneCycleLR
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * config.EPOCHS

        # Ensure pct_start is valid (0 < pct_start < 1)
        pct_start = float(config.WARMUP_EPOCHS) / config.EPOCHS
        if pct_start >= 1.0 or pct_start <= 0.0:
            pct_start = 0.3

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=25.0,
            final_div_factor=1000.0,
        )

        self.criterion = nn.L1Loss()
        self.best_metric = float("inf")

    def train_epoch(self):
        """Runs one epoch of training."""
        self.model.train()
        total_loss = 0.0
        num_samples = 0

        for batch in self.train_loader:
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            self.optimizer.zero_grad()
            preds = self.model(batch)
            loss = self.criterion(preds, batch["y"])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()

            batch_size = batch["y"].size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size

        return total_loss / num_samples

    def validate(self):
        """Runs validation and computes Log MAE."""
        self.model.eval()
        val_preds = []
        val_targets = []
        val_types = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                preds = self.model(batch)

                val_preds.append(preds.cpu())
                val_targets.append(batch["y"].cpu())
                val_types.append(batch["coupling_type"].cpu())

        # Concatenate
        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_types = torch.cat(val_types).numpy()

        # Map integer types to strings for Standardizer and Metric
        type_str_map = np.array(COUPLING_TYPES)
        val_types_str = type_str_map[val_types]

        # Inverse Transform
        # Note: Standardizer expects string types if it was fitted with strings
        orig_preds = self.standardizer.inverse_transform(val_preds, val_types_str)
        orig_targets = self.standardizer.inverse_transform(val_targets, val_types_str)

        # Compute Metric
        metric = compute_log_mae(orig_preds, orig_targets, val_types_str)
        return metric

    def fit(self):
        """Main training loop with early stopping."""
        print(f"Starting training for {self.config.EPOCHS} epochs...")

        patience = 5
        patience_counter = 0

        for epoch in range(self.config.EPOCHS):
            train_loss = self.train_epoch()
            val_metric = self.validate()

            # Print full precision as requested
            print(
                f"Epoch {epoch+1}/{self.config.EPOCHS} | Train MAE: {train_loss} | Val LMAE: {val_metric}"
            )

            if val_metric < self.best_metric:
                self.best_metric = val_metric
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
                print(f"  New best model saved! ({val_metric})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("  Early stopping triggered.")
                    break

        print(f"Training complete. Best Val LMAE: {self.best_metric}")


def train_model(config: Config):
    """
    Initializes and runs the training process.
    """
    set_seed(config.SEED)

    # Load Data
    print("Loading data...")
    train_loader, val_loader, _, standardizer = get_dataloaders(
        config, load_cached_data=True
    )

    # Initialize Model
    print("Initializing model...")
    model = HGANet(config)

    # Initialize Trainer
    trainer = Trainer(config, model, train_loader, val_loader, standardizer)

    # Run Training
    trainer.fit()


def generate_submission(config: Config):
    """
    Generates predictions for the test set and saves the submission file.
    """
    set_seed(config.SEED)

    # Load Test Data (and fit standardizer on train to get stats)
    # We need the standardizer fitted on training data to inverse transform predictions
    print("Loading test data and standardizer...")
    _, _, test_loader, standardizer = get_dataloaders(config, load_cached_data=True)

    # Load Model
    print(f"Loading model from {config.MODEL_SAVE_PATH}...")
    model = HGANet(config).to(config.DEVICE)
    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    )
    model.eval()

    # Predict
    print("Generating predictions...")
    all_preds = []
    all_types = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            preds = model(batch)
            all_preds.append(preds.cpu())
            all_types.append(batch["coupling_type"].cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_types = torch.cat(all_types).numpy()

    # Map types to strings
    type_str_map = np.array(COUPLING_TYPES)
    all_types_str = type_str_map[all_types]

    # Inverse Transform
    final_preds = standardizer.inverse_transform(all_preds, all_types_str)

    # Create Submission DataFrame
    print("Saving submission...")
    df_test = pd.read_csv(config.TEST_METADATA)

    # Fallback for empty predictions (e.g. missing structures in debug mode)
    if len(final_preds) == 0 and len(df_test) > 0:
        print("Warning: No predictions generated. Filling submission with zeros.")
        final_preds = np.zeros(len(df_test), dtype=np.float32)

    # Safety check
    if len(df_test) != len(final_preds):
        raise ValueError(
            f"Length mismatch: Metadata {len(df_test)} vs Preds {len(final_preds)}"
        )

    df_test["scalar_coupling_constant"] = final_preds

    # Save only id and scalar_coupling_constant
    submission_df = df_test[["id", "scalar_coupling_constant"]]
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
