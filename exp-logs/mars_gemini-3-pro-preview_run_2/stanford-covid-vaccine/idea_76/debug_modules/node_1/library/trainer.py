import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, get_global_rmse
from library.dataset import RNADataset
from library.model import AHCHIDN
from library.loss import AnchoredMCRMSELoss


class Trainer:
    """
    Manages the training, validation, and saving of the AHC-HIDN model.
    """

    def __init__(self):
        # 1. Setup Reproducibility
        set_seed(Config.SEED)
        self.device = torch.device(Config.DEVICE)

        # 2. Initialize Model
        self.model = AHCHIDN().to(self.device)

        # 3. Initialize Loss
        self.criterion = AnchoredMCRMSELoss()

        # 4. Initialize Optimizer and Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-2
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # 5. Paths
        self.best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    def get_dataloader(self, mode):
        """Creates a DataLoader for the specified mode."""
        dataset = RNADataset(mode=mode, load_cached_data=True)
        shuffle = mode == "train"

        return DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if self.device.type == "cuda" else False,
        )

    def train_epoch(self, train_loader):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            inputs = batch["inputs"].to(self.device)
            partner_indices = batch["partner_indices"].to(self.device)
            targets = batch["targets"].to(self.device)

            self.optimizer.zero_grad()

            # --- Pass 1: Initial Prediction (Zero Feedback) ---
            # prev_preds defaults to None inside model, which initializes zeros
            preds_1 = self.model(inputs, partner_indices, prev_preds=None)

            # --- Pass 2: Refined Prediction (Recycled Feedback) ---
            # Detach gradients from Pass 1 to stop backprop through time across passes
            # (We treat Pass 1 as a static input generator for Pass 2)
            preds_1_detached = preds_1.detach()
            preds_2 = self.model(inputs, partner_indices, prev_preds=preds_1_detached)

            # --- Loss Calculation ---
            # Calculate loss for both passes
            loss_1 = self.criterion(preds_1, targets)
            loss_2 = self.criterion(preds_2, targets)

            # Weighted Total Loss
            # We prioritize the final output (loss_2) but supervise the intermediate output (loss_1)
            total_loss = loss_2 + (Config.AUX_LOSS_WEIGHT * loss_1)

            # --- Optimization ---
            total_loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            running_loss += total_loss.item()

        return running_loss / len(train_loader)

    def validate(self, val_loader):
        """Runs validation and calculates metrics."""
        self.model.eval()
        running_loss = 0.0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["inputs"].to(self.device)
                partner_indices = batch["partner_indices"].to(self.device)
                targets = batch["targets"].to(self.device)

                # --- Pass 1 ---
                preds_1 = self.model(inputs, partner_indices, prev_preds=None)

                # --- Pass 2 ---
                # Use Pass 1 output as feedback
                preds_2 = self.model(inputs, partner_indices, prev_preds=preds_1)

                # Calculate Validation Loss (on Pass 2)
                loss = self.criterion(preds_2, targets)
                running_loss += loss.item()

                # Store predictions and targets for global metric calculation
                all_preds.append(preds_2.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate Global RMSE (MCRMSE on scored positions)
        global_rmse = get_global_rmse(all_preds, all_targets)
        avg_loss = running_loss / len(val_loader)

        return avg_loss, global_rmse

    def fit(self):
        """Main training loop with Early Stopping."""
        print(f"Starting training on device: {self.device}")

        train_loader = self.get_dataloader("train")
        val_loader = self.get_dataloader("val")

        best_metric = float("inf")
        patience_counter = 0
        patience_limit = 10  # Stop if no improvement for 10 epochs

        start_time = time.time()

        for epoch in range(1, Config.EPOCHS + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss, val_metric = self.validate(val_loader)

            # Scheduler Step
            self.scheduler.step(val_metric)

            epoch_duration = time.time() - epoch_start

            print(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val MCRMSE: {val_metric:.10f} | "
                f"Time: {epoch_duration:.2f}s"
            )

            # Checkpointing & Early Stopping
            if val_metric < best_metric:
                best_metric = val_metric
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  >>> New Best Model Saved (MCRMSE: {best_metric:.10f})")
            else:
                patience_counter += 1
                print(
                    f"  >>> No improvement. Patience: {patience_counter}/{patience_limit}"
                )
                if patience_counter >= patience_limit:
                    print("Early stopping triggered.")
                    break

        total_time = time.time() - start_time
        print(f"Training complete. Total time: {total_time:.2f}s")
        print(f"Best Validation MCRMSE: {best_metric:.10f}")
