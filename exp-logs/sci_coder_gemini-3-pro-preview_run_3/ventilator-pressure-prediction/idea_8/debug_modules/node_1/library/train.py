import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, compute_mae
from library.data import DataProcessor, VentilatorDataset
from library.model import PSANet


class Trainer:
    """
    Trainer class for the PSA-Net model.
    Handles training, validation, and submission generation.
    """

    def __init__(self, config=Config):
        self.config = config
        seed_everything(self.config.SEED)

        self.device = torch.device(self.config.DEVICE)

        # Initialize Model
        self.model = PSANet(self.config).to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        # Initialize Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.SCHEDULER_FACTOR,
            patience=self.config.SCHEDULER_PATIENCE,
        )

        # Loss Function (L1 Loss) - Reduction handled manually for masking
        self.criterion = nn.L1Loss(reduction="none")

        self.best_val_mae = float("inf")

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            u_out = batch["u_out"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(x)

            # Masked L1 Loss (Calculate only for inspiratory phase: u_out == 0)
            mask = (u_out == 0).float()
            loss_per_element = self.criterion(preds, y)

            # Apply mask: sum of masked losses / sum of mask
            masked_loss_sum = (loss_per_element * mask).sum()
            mask_sum = mask.sum()

            if mask_sum > 0:
                loss = masked_loss_sum / mask_sum
            else:
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.MAX_GRAD_NORM
            )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self, val_loader):
        """
        Runs validation and calculates MAE.
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        all_u_out = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                u_out = batch["u_out"].to(self.device)

                preds = self.model(x)

                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())
                all_u_out.append(u_out.cpu())

        # Concatenate all batches
        if not all_preds:
            return 0.0

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_u_out = torch.cat(all_u_out)

        # Compute MAE using the provided utility
        mae = compute_mae(all_preds, all_targets, all_u_out)
        return mae

    def predict(self, test_loader):
        """
        Generates predictions for the test set.
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(self.device)
                preds = self.model(x)
                predictions.append(preds.cpu().numpy())

        # Concatenate (N_breaths, Seq_Len) and flatten to (N_breaths * Seq_Len)
        if predictions:
            predictions = np.concatenate(predictions, axis=0).flatten()
        else:
            predictions = np.array([])

        return predictions

    def run(self):
        """
        Main execution method: Data Prep -> Train -> Validate -> Submit.
        """
        # 1. Data Preparation
        print("Initializing DataProcessor...")
        processor = DataProcessor(self.config)

        # Load data (handles caching internally)
        (
            (train_x, train_y, train_u_out),
            (val_x, val_y, val_u_out),
            (test_x, _, test_u_out),
        ) = processor.prepare_data(load_cached_data=True)

        # Create Datasets
        train_dataset = VentilatorDataset(train_x, train_y, train_u_out)
        val_dataset = VentilatorDataset(val_x, val_y, val_u_out)
        test_dataset = VentilatorDataset(test_x, None, test_u_out)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        # 2. Training Loop
        print(f"Starting training on {self.device} for {self.config.EPOCHS} epochs...")
        best_model_path = os.path.join(self.config.WORKING_DIR, "model.pth")
        patience_counter = 0

        for epoch in range(self.config.EPOCHS):
            train_loss = self.train_epoch(train_loader)
            val_mae = self.validate(val_loader)

            print(
                f"Epoch {epoch+1}/{self.config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MAE: {val_mae}"
            )

            # Scheduler Step
            self.scheduler.step(val_mae)

            # Checkpoint & Early Stopping
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                torch.save(self.model.state_dict(), best_model_path)
                print(f"New best model saved with MAE: {self.best_val_mae}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break

        # 3. Submission Generation
        print("Generating submission...")

        # Load best model
        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            print("Loaded best model checkpoint.")
        else:
            print("Warning: No checkpoint found. Using current model state.")

        preds = self.predict(test_loader)

        # Load Test Metadata to get IDs
        # We must replicate the sorting logic used in DataProcessor to align IDs with predictions
        test_df = pd.read_csv(self.config.TEST_PATH)

        # Handle DEBUG mode truncation if active
        if self.config.DEBUG:
            test_ids = test_df["breath_id"].unique()[: self.config.DEBUG_SAMPLES]
            test_df = test_df[test_df["breath_id"].isin(test_ids)].copy()

        # Sort to match the order of 'test_x' (breath_id, time_step)
        test_df = test_df.sort_values(["breath_id", "time_step"])

        # Verify lengths match
        if len(preds) != len(test_df):
            print(
                f"Error: Prediction length ({len(preds)}) does not match Test DF length ({len(test_df)})."
            )
        else:
            # Create Submission DataFrame
            submission = pd.DataFrame({"id": test_df["id"], "pressure": preds})

            # Sort by ID for final output format
            submission = submission.sort_values("id")

            # Save
            os.makedirs(self.config.SUBMISSION_DIR, exist_ok=True)
            submission.to_csv(self.config.SUBMISSION_PATH, index=False)
            print(f"Submission saved to {self.config.SUBMISSION_PATH}")
