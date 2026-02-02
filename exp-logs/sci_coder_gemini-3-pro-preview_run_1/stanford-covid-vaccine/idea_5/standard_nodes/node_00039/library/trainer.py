import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    SEQ_SCORED,
    TARGET_COLS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    PATIENCE,
    EPOCHS,
    SEED,
    BATCH_SIZE,
)
from library.model import HybridRNNTransformer
from library.loss import SignalWeightedMSELoss
from library.dataset import get_dataloaders


class Trainer:
    def __init__(self, model, device, criterion, optimizer, scheduler=None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_score = float("inf")

        # Indices of the columns used for the competition metric (MCRMSE)
        # TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # Scored: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
        self.scored_target_indices = [0, 1, 3]

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0

        for batch in dataloader:
            # Move data to device
            seq = batch["sequence"].to(self.device)
            struct = batch["structure"].to(self.device)
            loop = batch["predicted_loop_type"].to(self.device)
            targets = batch["targets"].to(self.device)
            masks = batch["mask"].to(self.device)
            weights = batch["weight"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(seq, struct, loop)

            # Compute loss
            loss = self.criterion(outputs, targets, masks, weights)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                seq = batch["sequence"].to(self.device)
                struct = batch["structure"].to(self.device)
                loop = batch["predicted_loop_type"].to(self.device)
                targets = batch["targets"].to(self.device)

                outputs = self.model(seq, struct, loop)

                # Collect predictions and targets for metric calculation
                # We only care about the scored positions (first 68) for the metric
                all_preds.append(outputs[:, :SEQ_SCORED, :].cpu().numpy())
                all_targets.append(targets[:, :SEQ_SCORED, :].cpu().numpy())

        # Concatenate all batches
        # Shape: (Total_Samples, 68, 5)
        preds_arr = np.concatenate(all_preds, axis=0)
        targets_arr = np.concatenate(all_targets, axis=0)

        # Calculate MCRMSE
        # 1. Select only the scored columns [0, 1, 3]
        preds_scored = preds_arr[:, :, self.scored_target_indices]
        targets_scored = targets_arr[:, :, self.scored_target_indices]

        # 2. Calculate MSE per column
        # Flatten sample and sequence dimensions: (N * 68, 3)
        mse_per_col = np.mean((preds_scored - targets_scored) ** 2, axis=(0, 1))

        # 3. RMSE per column
        rmse_per_col = np.sqrt(mse_per_col)

        # 4. Mean of RMSEs
        mcrmse = np.mean(rmse_per_col)

        return mcrmse

    def fit(self, train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE):
        print(f"Starting training on device: {self.device}")

        patience_counter = 0
        best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_mcrmse = self.validate(val_loader)

            if self.scheduler:
                self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse}"
            )

            # Early Stopping and Checkpointing
            if val_mcrmse < self.best_score:
                self.best_score = val_mcrmse
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"  New best model saved with MCRMSE: {self.best_score}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered after {patience} epochs without improvement."
                    )
                    break

        print(f"Training complete. Best Val MCRMSE: {self.best_score}")

    def predict_and_submit(self, test_loader, output_path=SUBMISSION_PATH):
        print("Generating predictions for test set...")

        # Load best model
        best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            print("Loaded best model checkpoint.")
        else:
            print(
                "Warning: Best model checkpoint not found. Using current model weights."
            )

        self.model.eval()

        ids_list = []
        preds_list = []

        with torch.no_grad():
            for batch in test_loader:
                seq = batch["sequence"].to(self.device)
                struct = batch["structure"].to(self.device)
                loop = batch["predicted_loop_type"].to(self.device)
                ids = batch["id"]  # List of IDs

                # Forward pass
                # Output shape: (B, 107, 5)
                outputs = self.model(seq, struct, loop)

                ids_list.extend(ids)
                preds_list.append(outputs.cpu().numpy())

        # Concatenate predictions: (Total_Test_Samples, 107, 5)
        all_preds = np.concatenate(preds_list, axis=0)

        # Prepare submission data
        submission_rows = []

        # Iterate over samples
        for i, sample_id in enumerate(ids_list):
            sample_preds = all_preds[i]  # Shape (107, 5)

            # Iterate over positions (0 to 106)
            for seqpos in range(sample_preds.shape[0]):
                row_id = f"{sample_id}_{seqpos}"
                row_values = sample_preds[seqpos]

                # Create row dict
                row = {
                    "id_seqpos": row_id,
                    "reactivity": row_values[0],
                    "deg_Mg_pH10": row_values[1],
                    "deg_pH10": row_values[2],
                    "deg_Mg_50C": row_values[3],
                    "deg_50C": row_values[4],
                }
                submission_rows.append(row)

        # Create DataFrame
        submission_df = pd.DataFrame(submission_rows)

        # Ensure column order
        cols = ["id_seqpos"] + TARGET_COLS
        submission_df = submission_df[cols]

        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, batch_size=BATCH_SIZE
    )

    # Initialize Model
    model = HybridRNNTransformer().to(device)

    # Loss Function
    criterion = SignalWeightedMSELoss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # Initialize Trainer
    trainer = Trainer(model, device, criterion, optimizer, scheduler)

    # Train
    trainer.fit(train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE)

    # Predict
    trainer.predict_and_submit(test_loader, SUBMISSION_PATH)
