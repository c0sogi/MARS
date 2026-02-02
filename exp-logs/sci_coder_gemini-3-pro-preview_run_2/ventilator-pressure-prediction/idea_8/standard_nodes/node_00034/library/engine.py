import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import WeightedL1Loss, compute_metric, seed_everything
from library.model import GIDBiLSTM
from library.data_processing import prepare_datasets


class Trainer:
    """
    Manages the training, validation, and inference processes for the Ventilator Pressure Prediction model.
    """

    def __init__(self, model, device, optimizer=None, scheduler=None, criterion=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

    def train_one_epoch(self, dataloader):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # Unpack batch and move to device
            # Batch structure from VentilatorDataset: X, u_out, y
            X = batch[0].to(self.device, non_blocking=True)
            u_out = batch[1].to(self.device, non_blocking=True)
            y = batch[2].to(self.device, non_blocking=True)

            # Forward pass
            preds = self.model(X)

            # Compute loss
            loss = self.criterion(preds, y, u_out)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            # Optimizer step
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, dataloader):
        """
        Evaluates the model on the validation set.
        Returns average loss and Inspiratory MAE.
        """
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                X = batch[0].to(self.device, non_blocking=True)
                u_out = batch[1].to(self.device, non_blocking=True)
                y = batch[2].to(self.device, non_blocking=True)

                preds = self.model(X)

                # Compute Loss
                loss = self.criterion(preds, y, u_out)
                total_loss += loss.item()

                # Compute Metric (Inspiratory MAE)
                mae = compute_metric(preds, y, u_out)
                total_mae += mae

                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        return avg_loss, avg_mae

    def predict(self, dataloader):
        """
        Generates predictions for the test set.
        Returns a flat numpy array of predictions.
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                X = batch[0].to(self.device, non_blocking=True)
                # We don't need u_out or y for inference, but dataset returns them

                preds = self.model(X)

                # Move to CPU and flatten
                preds_np = preds.cpu().numpy().flatten()
                all_preds.append(preds_np)

        return np.concatenate(all_preds)


def run_training(debug=Config.DEBUG):
    """
    Main execution function to prepare data, train the model, and save the best checkpoint.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Starting training on device: {device}")

    # 2. Data Preparation
    # Note: prepare_datasets handles caching internally
    train_dataset, val_dataset, _ = prepare_datasets(debug=debug, load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    # 3. Model Initialization
    model = GIDBiLSTM().to(device)

    criterion = WeightedL1Loss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    trainer = Trainer(model, device, optimizer, scheduler, criterion)

    # 4. Training Loop
    best_val_mae = float("inf")
    patience = 15  # Early stopping patience
    patience_counter = 0

    print("Starting training loop...")
    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = trainer.train_one_epoch(train_loader)

        # Validate
        val_loss, val_mae = trainer.validate(val_loader)

        # Step Scheduler
        if scheduler:
            scheduler.step()

        elapsed = time.time() - start_time

        # Logging (Full Precision)
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | Val MAE: {val_mae}"
        )

        # Checkpoint & Early Stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)
            print(f"New best model saved with Val MAE: {best_val_mae}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break

    print(f"Training complete. Best Val MAE: {best_val_mae}")


def generate_submission(debug=Config.DEBUG):
    """
    Loads the best model, generates predictions on the test set, and creates the submission file.
    """
    print("Generating submission...")
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Data
    # We only need the test dataset here
    _, _, test_dataset = prepare_datasets(debug=debug, load_cached_data=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Crucial: must not shuffle to maintain order
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # 2. Load Model
    model = GIDBiLSTM().to(device)
    if not os.path.exists(Config.MODEL_CHECKPOINT):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_CHECKPOINT}"
        )

    state_dict = torch.load(Config.MODEL_CHECKPOINT, map_location=device)
    model.load_state_dict(state_dict)

    trainer = Trainer(model, device)

    # 3. Inference
    print("Running inference on test set...")
    predictions = trainer.predict(test_loader)

    # 4. Create Submission File
    # The data processing pipeline sorts test data by ['breath_id', 'id'].
    # We need to map these predictions back to the sample submission format (sorted by 'id').

    print("Loading test metadata for alignment...")
    test_meta = pd.read_csv(Config.TEST_META)

    # Ensure metadata is sorted exactly how the dataset was processed
    test_meta.sort_values(["breath_id", "id"], inplace=True)

    # Verify lengths match
    if len(predictions) != len(test_meta):
        raise ValueError(
            f"Prediction length {len(predictions)} does not match metadata length {len(test_meta)}"
        )

    # Assign predictions
    test_meta["pressure"] = predictions

    # Sort by 'id' to match submission requirement
    test_meta.sort_values("id", inplace=True)

    # Select columns and save
    submission_df = test_meta[["id", "pressure"]]
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
