import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import set_seed
from library.dataset import load_data, GNSSDataset, gnss_collate_fn
from library.model import ResUNet1D
from library.loss import DeepSupervisionMAELoss


class Trainer:
    """
    Manages the training and validation process for the ResUNet1D model.
    """

    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Permute features to (Batch, Channels, Length) for Conv1d
            # Input features are (Batch, Length, Channels) from collate_fn
            features = features.permute(0, 2, 1)

            self.optimizer.zero_grad()

            # Forward pass (returns tuple of outputs due to deep supervision)
            outputs = self.model(features)

            # Calculate loss
            # Loss expects (preds, targets, mask)
            loss = self.criterion(outputs, targets, mask)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        return running_loss / num_batches if num_batches > 0 else 0.0

    def validate(self):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["features"].to(self.device)
                targets = batch["targets"].to(self.device)
                mask = batch["mask"].to(self.device)

                features = features.permute(0, 2, 1)

                # Forward pass (returns single tensor during eval)
                output = self.model(features)

                loss = self.criterion(output, targets, mask)
                running_loss += loss.item()
                num_batches += 1

        return running_loss / num_batches if num_batches > 0 else 0.0

    def fit(self, epochs, patience, checkpoint_path):
        """
        Executes the training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")
        print(f"Epochs: {epochs}, Patience: {patience}")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            print(f"Epoch {epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}")

            # Checkpoint and Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  New best model saved to {checkpoint_path}")
            else:
                self.patience_counter += 1
                print(f"  No improvement. Patience: {self.patience_counter}/{patience}")

            if self.patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        print(f"Training complete. Best Val Loss: {self.best_val_loss}")


def run_training(load_cached_data=True):
    """
    Main function to setup and run the training pipeline.
    """
    # 1. Reproducibility
    set_seed(Config.SEED)

    # 2. Data Loading
    print("Loading training data...")
    train_df = load_data(
        Config.TRAIN_METADATA_PATH, Config.TRAIN_CACHE, load_cached_data
    )

    print("Loading validation data...")
    val_df = load_data(Config.VAL_METADATA_PATH, Config.VAL_CACHE, load_cached_data)

    # Debug Sampling
    if Config.DEBUG:
        print(
            f"DEBUG MODE: Sampling {Config.DEBUG_SAMPLE_SIZE} trips for training/validation."
        )
        train_trip_ids = train_df["trip_id"].unique()[: Config.DEBUG_SAMPLE_SIZE]
        val_trip_ids = val_df["trip_id"].unique()[: Config.DEBUG_SAMPLE_SIZE]

        train_df = train_df[train_df["trip_id"].isin(train_trip_ids)].reset_index(
            drop=True
        )
        val_df = val_df[val_df["trip_id"].isin(val_trip_ids)].reset_index(drop=True)

    # 3. Dataset Creation
    train_dataset = GNSSDataset(train_df)
    val_dataset = GNSSDataset(val_df)

    print(f"Train Dataset: {len(train_dataset)} trips")
    print(f"Val Dataset: {len(val_dataset)} trips")

    # 4. DataLoader Creation
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=gnss_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=gnss_collate_fn,
        pin_memory=True,
    )

    # 5. Model Initialization
    device = torch.device(Config.DEVICE)
    model = ResUNet1D().to(device)

    # 6. Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = DeepSupervisionMAELoss()

    # 7. Trainer Setup and Execution
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    trainer.fit(
        epochs=Config.EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
        checkpoint_path=Config.MODEL_CHECKPOINT,
    )
