import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library components
from library.model import ResUNet1D
from library.data import load_data
from library.utils import set_seed

# Constants
METADATA_DIR = "./metadata"
TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
MODEL_SAVE_PATH = "./working/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, model, learning_rate=1e-3, weight_decay=1e-4):
        self.model = model.to(DEVICE)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.L1Loss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )
        self.best_val_loss = float("inf")

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        count = 0

        for batch_idx, (x, y, meta) in enumerate(train_loader):
            # x shape: (Batch=1, SeqLen, Features)
            # y shape: (Batch=1, SeqLen, Targets=2)
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(x)

            # Loss calculation
            loss = self.criterion(output, y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)  # scale by batch size (which is 1)
            count += x.size(0)

        return total_loss / count if count > 0 else 0.0

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        total_dist_error = 0.0
        count = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (x, y, meta) in enumerate(val_loader):
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                output = self.model(x)
                loss = self.criterion(output, y)

                total_loss += loss.item() * x.size(0)
                count += x.size(0)

                # Calculate Euclidean distance error for metrics
                # output: (B, L, 2), y: (B, L, 2)
                # Error = sqrt((pred_e - true_e)^2 + (pred_n - true_n)^2)
                diff = output - y
                dist_error = torch.sqrt(torch.sum(diff**2, dim=2))  # (B, L)

                total_dist_error += dist_error.sum().item()
                total_samples += dist_error.numel()

        avg_loss = total_loss / count if count > 0 else 0.0
        avg_dist_error = total_dist_error / total_samples if total_samples > 0 else 0.0

        return avg_loss, avg_dist_error

    def fit(self, train_loader, val_loader, epochs=20, patience=5):
        print(f"Starting training on {DEVICE}...")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_dist_error = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss (MAE): {train_loss:.6f} | "
                f"Val Loss (MAE): {val_loss:.6f} | "
                f"Val Mean Dist Error: {val_dist_error:.6f} m"
            )

            # Checkpointing and Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
                print(f"  Model saved. Best Val Loss: {self.best_val_loss:.6f}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        print("Training complete.")


def train_model(
    epochs=30,
    batch_size=1,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=7,
    max_drives=None,
    load_cached_data=True,
):
    """
    Main function to train the model.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size (keep 1 for variable sequence lengths).
        learning_rate (float): Initial learning rate.
        weight_decay (float): Weight decay for optimizer.
        patience (int): Early stopping patience.
        max_drives (int): Limit number of drives for debugging/quick runs.
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    set_seed(42)

    # 1. Load Data
    print("Loading datasets...")
    # Ensure metadata exists
    if not os.path.exists(TRAIN_META) or not os.path.exists(VAL_META):
        print("Metadata files not found. Ensure metadata generation was run.")
        return

    train_dataset = load_data(
        TRAIN_META,
        split="train",
        load_cached_data=load_cached_data,
        max_drives=max_drives,
    )
    val_dataset = load_data(
        VAL_META,
        split="train",
        load_cached_data=load_cached_data,
        max_drives=max_drives,
    )

    if len(train_dataset) == 0:
        print("No training data found.")
        return

    # Determine input features dimension from the first sample
    sample_x, _, _ = train_dataset[0]
    in_channels = sample_x.shape[1]  # (SeqLen, Features) -> Features is dim 1
    print(f"Input Feature Dimension: {in_channels}")

    # Create DataLoaders
    # batch_size=1 is crucial here because sequences have different lengths.
    # To use batch_size > 1, a custom collate_fn with padding is required.
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 2. Initialize Model
    model = ResUNet1D(in_channels=in_channels, out_channels=2, base_channels=32)

    # 3. Train
    trainer = Trainer(model, learning_rate=learning_rate, weight_decay=weight_decay)
    trainer.fit(train_loader, val_loader, epochs=epochs, patience=patience)

    return trainer
