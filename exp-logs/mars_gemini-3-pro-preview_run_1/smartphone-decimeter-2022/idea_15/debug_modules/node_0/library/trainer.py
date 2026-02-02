import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import seed_everything, get_logger, enu_to_latlon
from library.model import AttentionGatedResUNet1D
from library.loss import MultiScaleMAELoss
from library.data_processing import get_data
from library.dataset import get_datasets

# Initialize logger
logger = get_logger(os.path.join(Config.WORKING_DIR, "train.log"))


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, patience=10):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.best_val_loss = float("inf")
        self.counter = 0

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0

        for batch in dataloader:
            features = batch["features"].to(self.device, dtype=torch.float32)
            mask = batch["mask"].to(self.device, dtype=torch.float32)

            # Targets is a list of tensors for multi-scale supervision
            targets = [t.to(self.device, dtype=torch.float32) for t in batch["targets"]]

            self.optimizer.zero_grad()

            # Forward pass: returns list of outputs [scale1, scale2, scale4, scale8]
            outputs = self.model(features)

            # Compute Multi-Scale Loss
            loss = self.criterion(outputs, targets, mask)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * features.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss

    def validate(self, dataloader):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device, dtype=torch.float32)
                mask = batch["mask"].to(self.device, dtype=torch.float32)
                targets = [
                    t.to(self.device, dtype=torch.float32) for t in batch["targets"]
                ]

                outputs = self.model(features)
                loss = self.criterion(outputs, targets, mask)

                running_loss += loss.item() * features.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss

    def fit(self, train_loader, val_loader, epochs):
        logger.info(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            # Scheduler step
            if self.scheduler:
                self.scheduler.step(val_loss)

            duration = time.time() - start_time

            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Time: {duration:.2f}s"
            )

            # Early Stopping and Model Saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                logger.info(
                    f"  -> Validation loss improved. Model saved to {Config.MODEL_SAVE_PATH}"
                )
            else:
                self.counter += 1
                logger.info(
                    f"  -> Early stopping counter: {self.counter}/{self.patience}"
                )

            if self.counter >= self.patience:
                logger.info("Early stopping triggered.")
                break

        logger.info(
            f"Training complete. Best Validation Loss: {self.best_val_loss:.6f}"
        )


def generate_submission(model, test_loader, device):
    """
    Generates submission file using the trained model.
    """
    logger.info("Generating submission...")
    model.eval()

    results = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device, dtype=torch.float32)

            # Forward pass
            # The model returns a list [scale1, scale2, ...]. We want the highest resolution (index 0).
            outputs_list = model(features)
            predictions = outputs_list[0].cpu().numpy()  # Shape: (B, 2, L)

            # Metadata
            meta = batch["meta"]
            original_lengths = batch["original_length"].numpy()

            batch_size = features.size(0)

            for i in range(batch_size):
                # Extract valid sequence length (remove padding)
                length = original_lengths[i]

                # Extract predictions (Delta East, Delta North)
                # Shape (2, L) -> (L, 2)
                pred_seq = predictions[i, :, :length].transpose(1, 0)
                delta_east = pred_seq[:, 0]
                delta_north = pred_seq[:, 1]

                # Extract metadata for reconstruction
                drive_id = meta["drive_id"][i]
                phone_name = meta["phone_name"][i]

                # meta['UnixTimeMillis'] is a tensor from collate
                timestamps = meta["UnixTimeMillis"][i][:length].numpy()
                wls_lat = meta["wls_lat"][i][:length].numpy()
                wls_lon = meta["wls_lon"][i][:length].numpy()

                # Convert ENU offsets to Lat/Lon
                pred_lat, pred_lon = enu_to_latlon(
                    delta_east, delta_north, wls_lat, wls_lon
                )

                # Construct result rows
                for t, lat, lon in zip(timestamps, pred_lat, pred_lon):
                    trip_id = f"{drive_id}-{phone_name}"
                    results.append(
                        {
                            "tripId": trip_id,
                            "UnixTimeMillis": int(t),
                            "LatitudeDegrees": lat,
                            "LongitudeDegrees": lon,
                        }
                    )

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure correct column order
    cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    submission_df = submission_df[cols]

    # Save
    submission_df.to_csv(Config.SUBMISSION_SAVE_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_SAVE_PATH}")


def run_training(
    load_cached_data=True, num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE
):
    seed_everything(Config.SEED)

    # 1. Load Data
    logger.info("Loading data...")
    train_df, val_df, test_df = get_data(load_cached_data=load_cached_data)

    if train_df.empty:
        logger.error("Training data is empty. Exiting.")
        return

    # 2. Create Datasets
    logger.info("Creating datasets...")
    train_dataset, val_dataset, test_dataset = get_datasets(train_df, val_df, test_df)

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Initialize Model
    device = Config.DEVICE
    logger.info(f"Initializing model on {device}...")
    model = AttentionGatedResUNet1D().to(device)

    # 5. Setup Training Components
    criterion = MultiScaleMAELoss().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # 6. Train
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=Config.EARLY_STOPPING_PATIENCE,
    )

    trainer.fit(train_loader, val_loader, epochs=num_epochs)

    # 7. Generate Submission
    # Load best model
    logger.info("Loading best model for submission...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    generate_submission(model, test_loader, device)
