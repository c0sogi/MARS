import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import Config
from library.utils import seed_everything, compute_metric
from library.data_factory import prepare_datasets
from library.model import PCSDHNet


class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize Model
        self.model = PCSDHNet().to(self.device)

        # Optimizer and Scheduler
        self.optimizer = AdamW(
            self.model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-2
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Loss Function (L1 Loss)
        self.criterion = nn.L1Loss(reduction="none")

        # Best model tracking
        self.best_val_mae = float("inf")
        self.best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    def masked_l1_loss(self, y_pred, y_true, u_out):
        """
        Computes L1 loss only for the inspiratory phase (u_out == 0).
        """
        # Create mask (1 for inspiratory, 0 for expiratory)
        mask = (u_out == 0).float()

        # Calculate element-wise L1 loss
        loss = self.criterion(y_pred, y_true)

        # Apply mask
        masked_loss = loss * mask

        # Average over the number of valid elements
        # Add epsilon to avoid division by zero
        return masked_loss.sum() / (mask.sum() + 1e-8)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for X, y, u_out in dataloader:
            X = X.to(self.device)
            y = y.to(self.device).unsqueeze(-1)  # Match output shape (B, L, 1)
            u_out = u_out.to(self.device).unsqueeze(-1)

            self.optimizer.zero_grad()

            preds = self.model(X)

            loss = self.masked_l1_loss(preds, y, u_out)

            loss.backward()

            # Gradient Clipping (Crucial for stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        all_u_out = []

        with torch.no_grad():
            for X, y, u_out in dataloader:
                X = X.to(self.device)
                y_gpu = y.to(self.device).unsqueeze(-1)
                u_out_gpu = u_out.to(self.device).unsqueeze(-1)

                preds = self.model(X)

                loss = self.masked_l1_loss(preds, y_gpu, u_out_gpu)
                total_loss += loss.item()

                # Store for metric calculation
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.numpy())
                all_u_out.append(u_out.numpy())

        # Concatenate all batches
        # Preds: (N_batches, B, L, 1) -> (Total, L)
        y_pred_flat = np.concatenate(all_preds, axis=0).squeeze(-1).flatten()
        y_true_flat = np.concatenate(all_targets, axis=0).flatten()
        u_out_flat = np.concatenate(all_u_out, axis=0).flatten()

        # Compute Metric
        mae = compute_metric(y_pred_flat, y_true_flat, u_out_flat)

        return total_loss / len(dataloader), mae

    def fit(self, train_loader, val_loader):
        print(f"Starting training for {Config.EPOCHS} epochs...")
        patience = 15
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss, val_mae = self.validate(val_loader)

            self.scheduler.step(val_loss)

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss:.8f} | "
                f"Val Loss: {val_loss:.8f} | "
                f"Val MAE: {val_mae:.16f} | "
                f"Time: {elapsed:.2f}s"
            )

            # Checkpointing
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"Saved new best model with MAE: {self.best_val_mae:.16f}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs of no improvement."
                )
                break

    def predict(self, test_loader):
        print("Loading best model for inference...")
        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )
        self.model.eval()

        all_preds = []

        with torch.no_grad():
            for X, _, _ in test_loader:
                X = X.to(self.device)
                preds = self.model(X)
                all_preds.append(preds.cpu().numpy())

        # Flatten predictions: (N, 80, 1) -> (N*80,)
        predictions = np.concatenate(all_preds, axis=0).squeeze(-1).flatten()
        return predictions


def main():
    # Ensure reproducibility
    seed_everything(Config.SEED)
    Config.setup_dirs()

    # Load Data
    print("Preparing datasets...")
    train_dataset, val_dataset, test_dataset, _ = prepare_datasets(
        load_cached_data=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Train
    trainer = Trainer()
    trainer.fit(train_loader, val_loader)

    # Inference
    print("Generating predictions...")
    predictions = trainer.predict(test_loader)

    # Create Submission
    print("Saving submission...")
    # Load sample submission to get IDs
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Safety check
    if len(predictions) != len(sample_sub):
        print(
            f"Warning: Prediction length ({len(predictions)}) matches sample submission ({len(sample_sub)})?"
        )
        # If lengths mismatch due to drop_last in training or something, we rely on the fact that
        # prepare_datasets preserves order and test_loader is not shuffled.
        # The test set size is fixed.

    sample_sub["pressure"] = predictions
    sample_sub.to_csv(Config.OUTPUT_PATH, index=False)
    print(f"Submission saved to {Config.OUTPUT_PATH}")


if __name__ == "__main__":
    main()
