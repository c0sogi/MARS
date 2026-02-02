import os
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import Config
from library.utils import set_seed, WGS84Converter
from library.dataset import get_dataloaders, get_test_dataloader
from library.model import MultiScaleResUNet1D
from library.loss import DeepSupervisionMAELoss


class Trainer:
    """
    Manages the training lifecycle, validation, and inference for the GNSS model.
    """

    def __init__(self, model, criterion, optimizer, scheduler, device, checkpoint_path):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.best_loss = float("inf")

    def train_epoch(self, dataloader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        for batch in dataloader:
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            mask = batch["mask"].to(self.device)

            batch_size = features.size(0)

            self.optimizer.zero_grad()

            # Forward pass (returns list of outputs for deep supervision)
            outputs = self.model(features)

            # Calculate loss
            loss = self.criterion(outputs, targets, mask)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_size
            total_samples += batch_size

        return running_loss / total_samples

    def validate(self, dataloader):
        """
        Runs validation on the given dataloader.
        """
        self.model.eval()
        running_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device)
                targets = batch["targets"].to(self.device)
                mask = batch["mask"].to(self.device)

                batch_size = features.size(0)

                # Forward pass (returns single tensor in eval mode)
                outputs = self.model(features)

                # Calculate loss
                loss = self.criterion(outputs, targets, mask)

                running_loss += loss.item() * batch_size
                total_samples += batch_size

        return running_loss / total_samples

    def fit(self, train_loader, val_loader, epochs, patience):
        """
        Executes the training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            # Scheduler step
            if self.scheduler:
                self.scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.9f} - Val Loss: {val_loss:.9f}"
            )

            # Early Stopping and Checkpointing
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"  New best model saved! Loss: {val_loss:.9f}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # Load best model for inference state
        print(f"Loading best model from {self.checkpoint_path}...")
        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device)
        )
        return self.best_loss

    def predict(self, test_loader, output_path):
        """
        Generates predictions for the test set and saves to CSV.
        """
        self.model.eval()
        converter = WGS84Converter()

        results = []

        print("Generating predictions...")
        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(self.device)

                # Metadata for reconstruction
                trip_ids = batch["trip_id"]  # list of strings
                timestamps = batch["UnixTimeMillis"].numpy()  # (B, T)
                wls_lats = batch["wls_lat"].numpy()  # (B, T)
                wls_lons = batch["wls_lon"].numpy()  # (B, T)
                seq_lens = batch["seq_len"]  # list/tensor

                # Forward pass
                # Output shape: (B, 2, T) -> (d_east, d_north)
                preds = self.model(features)
                preds = preds.cpu().numpy()

                # Iterate over batch
                for i in range(features.size(0)):
                    length = seq_lens[i]
                    trip_id = trip_ids[i]

                    # Extract valid sequence (ignore padding)
                    d_east_seq = preds[i, 0, :length]
                    d_north_seq = preds[i, 1, :length]

                    ts_seq = timestamps[i, :length]
                    lat_seq = wls_lats[i, :length]
                    lon_seq = wls_lons[i, :length]

                    # Convert meters back to degrees
                    pred_lats, pred_lons = converter.meters_to_deg(
                        d_east_seq, d_north_seq, lat_seq, lon_seq
                    )

                    # Append results
                    for t, lat, lon in zip(ts_seq, pred_lats, pred_lons):
                        results.append(
                            {
                                "tripId": trip_id,
                                "UnixTimeMillis": t,
                                "LatitudeDegrees": lat,
                                "LongitudeDegrees": lon,
                            }
                        )

        # Create DataFrame
        df_pred = pd.DataFrame(results)

        # Merge with sample submission to ensure correct order and rows
        if os.path.exists(Config.SAMPLE_SUBMISSION_PATH):
            df_sample = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
            # We perform a left join on the sample submission to keep its structure
            df_final = df_sample[["tripId", "UnixTimeMillis"]].merge(
                df_pred, on=["tripId", "UnixTimeMillis"], how="left"
            )

            # Fill missing predictions if any (fallback to WLS/baseline implicitly if 0 offset,
            # but here we just fill NaNs to avoid errors, though NaNs shouldn't occur with correct processing)
            if df_final["LatitudeDegrees"].isnull().any():
                print("Warning: Some predictions are missing. Filling NaNs.")
                df_final = df_final.fillna(0)  # Should ideally not happen
        else:
            df_final = df_pred

        # Save submission
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_final.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")


def train_pipeline(
    debug=False, epochs=Config.NUM_EPOCHS, patience=Config.EARLY_STOPPING_PATIENCE
):
    """
    Main function to run the training pipeline.
    """
    set_seed(Config.SEED)

    # 1. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, stats = get_dataloaders(debug=debug)

    # 2. Model Setup
    device = torch.device(Config.DEVICE)
    model = MultiScaleResUNet1D().to(device)

    criterion = DeepSupervisionMAELoss()
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=False
    )

    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # 3. Training
    trainer = Trainer(model, criterion, optimizer, scheduler, device, checkpoint_path)
    trainer.fit(train_loader, val_loader, epochs=epochs, patience=patience)

    # 4. Inference
    print("Initializing Test DataLoader...")
    test_loader = get_test_dataloader(stats, debug=debug)

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    trainer.predict(test_loader, submission_path)
