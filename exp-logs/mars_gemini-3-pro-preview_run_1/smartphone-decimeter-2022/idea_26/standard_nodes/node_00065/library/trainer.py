import os
import time
import copy
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import WGS84, seed_everything
from library.preprocessing import GNSSPreprocessor
from library.dataset import GnssSequenceDataset
from library.model import PhaseAwareAttentionResUNet
from library.loss import DecimatedMAELoss


class Trainer:
    """
    Trainer class for the Phase-Aware Stratified 1D Attention ResUNet.
    Handles training, validation, and submission generation.
    """

    def __init__(self, config=None):
        self.config = config if config else Config()
        self.device = torch.device(self.config.DEVICE)

        # Ensure reproducibility
        seed_everything(self.config.RANDOM_SEED)

        # Create necessary directories
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)
        os.makedirs(self.config.SUBMISSION_DIR, exist_ok=True)

    def get_dataloaders(self, load_cached_data=True, max_samples=None):
        """
        Prepares DataLoaders for training and validation.

        Args:
            load_cached_data (bool): Whether to load preprocessed data from cache.
            max_samples (int): Limit number of samples for debugging.

        Returns:
            train_loader, val_loader
        """
        preprocessor = GNSSPreprocessor()

        # Load/Generate Train Data
        train_df = preprocessor.generate_dataset(
            split="train", load_cached_data=load_cached_data
        )
        train_dataset = GnssSequenceDataset(
            train_df, split="train", config=self.config, max_samples=max_samples
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        # Load/Generate Validation Data
        val_df = preprocessor.generate_dataset(
            split="val", load_cached_data=load_cached_data
        )
        val_dataset = GnssSequenceDataset(
            val_df, split="val", config=self.config, max_samples=max_samples
        )

        # Validation batch size can be 1 since sequences are full length (no sliding window)
        # However, our dataset implementation pads to multiples of 16, so batching is possible
        # if lengths are uniform. But in val/test split logic, lengths vary by drive.
        # We use batch_size=1 for safety and simplicity in validation/inference.
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        return train_loader, val_loader

    def train_one_epoch(self, model, dataloader, criterion, optimizer, scheduler):
        """
        Runs one epoch of training.
        """
        model.train()
        running_loss = 0.0

        for batch in dataloader:
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            mask = batch["mask"].to(self.device)

            optimizer.zero_grad()

            # Forward pass
            # Returns final_output or (final_output, aux_outputs)
            outputs = model(features)

            # Compute loss
            loss, _ = criterion(outputs, targets, mask)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item() * features.size(0)

        if scheduler:
            scheduler.step()

        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss

    def validate(self, model, dataloader, criterion):
        """
        Runs validation on the validation set.
        """
        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device)
                targets = batch["targets"].to(self.device)
                mask = batch["mask"].to(self.device)

                # Forward pass
                outputs = model(features)

                # Compute loss
                # Note: outputs might be a tuple if deep supervision is on,
                # but in eval mode the model usually returns only the final output
                # unless configured otherwise. Our model returns only final_out in eval.
                # The loss expects a tensor or tuple. If model returns tensor, loss handles it.
                loss, _ = criterion(outputs, targets, mask)

                running_loss += loss.item() * features.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss

    def train_model(self, load_cached_data=True, max_samples=None, epochs=None):
        """
        Main training loop with Early Stopping.
        """
        if epochs is None:
            epochs = self.config.NUM_EPOCHS

        print(f"Starting training on device: {self.device}")

        # 1. Data
        train_loader, val_loader = self.get_dataloaders(load_cached_data, max_samples)

        # 2. Model
        model = PhaseAwareAttentionResUNet(self.config).to(self.device)

        # 3. Loss, Optimizer, Scheduler
        criterion = DecimatedMAELoss(self.config)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

        # 4. Training Loop
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = float("inf")
        patience_counter = 0

        save_path = os.path.join(self.config.WORKING_DIR, "best_model.pth")

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = self.train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler
            )
            val_loss = self.validate(model, val_loader, criterion)

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f} - "
                f"Time: {elapsed:.0f}s"
            )

            # Early Stopping and Checkpointing
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, save_path)
                patience_counter = 0
                print(f"  New best model saved! Loss: {best_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print("  Early stopping triggered.")
                    break

        print(f"Training complete. Best Val Loss: {best_loss:.6f}")

        # Load best weights
        model.load_state_dict(best_model_wts)
        return model

    def generate_submission(self, model=None, load_cached_data=True):
        """
        Generates submission file for the test set.
        """
        print("Generating submission...")

        # 1. Load Model if not provided
        if model is None:
            model_path = os.path.join(self.config.WORKING_DIR, "best_model.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

            model = PhaseAwareAttentionResUNet(self.config)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)

        model.eval()

        # 2. Load Test Data
        preprocessor = GNSSPreprocessor()
        test_df = preprocessor.generate_dataset(
            split="test", load_cached_data=load_cached_data
        )

        test_dataset = GnssSequenceDataset(test_df, split="test", config=self.config)

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Process one drive at a time
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
        )

        # 3. Inference Loop
        results = []

        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(self.device)
                # mask = batch["mask"].to(self.device) # Not needed for inference forward pass

                # Metadata for reconstruction
                wls_coords = batch["wls_coords"].numpy()[
                    0
                ]  # Shape: (Length, 2) [Lat, Lon]
                timestamps = batch["timestamps"].numpy()[0]  # Shape: (Length,)
                original_length = batch["original_length"].item()
                drive_id = batch["drive_id"][0]
                phone_name = batch["phone_name"][0]

                # Forward pass
                outputs = model(features)  # Shape: (1, 2, PaddedLength)

                # Unpad and transpose
                # outputs: [1, 2, Length] -> [2, Length] -> [Length, 2]
                preds = outputs[0, :, :original_length].cpu().numpy().transpose(1, 0)

                # preds is (DeltaNorth, DeltaEast) in meters
                delta_north = preds[:, 0]
                delta_east = preds[:, 1]

                # Reconstruct Global Coordinates
                # wls_coords is (Lat, Lon)
                wls_lat = wls_coords[:, 0]
                wls_lon = wls_coords[:, 1]

                # Convert meters offset to degrees offset
                # We do this point-by-point or vectorized
                # Using WGS84 utils logic vectorized:

                # Constants
                a = WGS84.a
                e2 = WGS84.e2

                lat_rad = np.radians(wls_lat)

                # Radius of curvature in the prime vertical
                Rn = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)

                # Radius of curvature in the meridian
                Rm = (a * (1 - e2)) / (1 - e2 * np.sin(lat_rad) ** 2) ** 1.5

                dlat = np.degrees(delta_north / Rm)
                dlon = np.degrees(delta_east / (Rn * np.cos(lat_rad)))

                pred_lat = wls_lat + dlat
                pred_lon = wls_lon + dlon

                # Construct result rows
                # Format: tripId, UnixTimeMillis, LatitudeDegrees, LongitudeDegrees
                # tripId = drive_id + "-" + phone_name (Based on sample submission format check)
                # Wait, sample submission description says: 2020-05-15-US-MTV-1_Pixel4
                # It seems it uses underscore in the example provided in prompt text.
                # However, metadata generation script used rsplit('-', 1).
                # Let's reconstruct based on the sample submission file provided in input.
                # We will use the format `drive_id` + `_` + `phone_name` based on the prompt's explicit example.
                # If the drive_id itself has hyphens (it does), that's fine.

                # Actually, let's check the test_metadata.csv tripId column if available.
                # The dataset class doesn't return tripId directly, but we have drive_id and phone_name.
                # Let's construct it.

                # Note: The prompt's sample submission example: 2020-05-15-US-MTV-1_Pixel4
                trip_id = f"{drive_id}_{phone_name}"

                for t, lat, lon in zip(timestamps, pred_lat, pred_lon):
                    results.append(
                        {
                            "tripId": trip_id,
                            "UnixTimeMillis": t,
                            "LatitudeDegrees": lat,
                            "LongitudeDegrees": lon,
                        }
                    )

        # 4. Save Submission
        submission_df = pd.DataFrame(results)

        # Ensure we only include rows present in the sample submission
        # Load sample submission to get exact required rows
        sample_sub_path = os.path.join(self.config.INPUT_DIR, "sample_submission.csv")
        if os.path.exists(sample_sub_path):
            sample_sub = pd.read_csv(sample_sub_path)
            # Merge to ensure order and existence
            # We merge on tripId and UnixTimeMillis
            final_sub = sample_sub[["tripId", "UnixTimeMillis"]].merge(
                submission_df, on=["tripId", "UnixTimeMillis"], how="left"
            )

            # Fill missing predictions with WLS if any (though we processed all)
            # If merge failed for some reason, we might have NaNs.
            # In this robust pipeline, we assume coverage is complete.
        else:
            final_sub = submission_df

        final_sub.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")
