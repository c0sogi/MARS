import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.utils import set_seed, WGS84Utils, haversine_distance
from library.preprocessing import GnssPreprocessor
from library.dataset import DualStreamGnssDataset, GnssScaler
from library.model import DualResUNet


class DecimatedMAELoss(nn.Module):
    """
    Computes weighted MAE loss for main output and auxiliary deep supervision heads.
    Automatically subsamples targets to match auxiliary head resolution.
    """

    def __init__(self, aux_weight=0.4, decimation_factor=4):
        super(DecimatedMAELoss, self).__init__()
        self.aux_weight = aux_weight
        self.decimation_factor = decimation_factor
        self.mae = nn.L1Loss()

    def forward(self, preds, targets):
        # preds can be a dict {'output': ..., 'aux': ...} or just tensor
        if isinstance(preds, dict):
            main_out = preds["output"]
            aux_out = preds["aux"]

            # Main loss (Full resolution)
            loss_main = self.mae(main_out, targets)

            # Aux loss (Decimated resolution)
            # Subsample targets: [Batch, Channels, Length] -> slice last dim
            # We assume the aux head aligns with the strided indices 0, factor, 2*factor...
            targets_decimated = targets[:, :, :: self.decimation_factor]

            # Ensure shapes match (handle potential edge cases with padding/odd lengths)
            if aux_out.shape[-1] != targets_decimated.shape[-1]:
                # Fallback: Interpolate targets to match aux output size
                targets_decimated = torch.nn.functional.interpolate(
                    targets, size=aux_out.shape[-1], mode="nearest"
                )

            loss_aux = self.mae(aux_out, targets_decimated)

            return loss_main + (self.aux_weight * loss_aux)
        else:
            # Inference mode or no aux head
            return self.mae(preds, targets)


class Trainer:
    def __init__(self, run_dir=None):
        self.config = Config
        self.device = torch.device(self.config.DEVICE)
        self.run_dir = run_dir if run_dir else self.config.WORKING_DIR
        os.makedirs(self.run_dir, exist_ok=True)

        self.model = DualResUNet().to(self.device)
        self.criterion = DecimatedMAELoss(
            aux_weight=self.config.AUX_LOSS_WEIGHT,
            decimation_factor=self.config.DECIMATION_FACTOR,
        )

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.EPOCHS, eta_min=self.config.ETA_MIN
        )

        self.scaler = None

    def load_data(self, load_cached=True):
        """
        Loads and processes train, validation, and test data.
        """
        preprocessor = GnssPreprocessor()

        # Train
        train_df = preprocessor.process_data(
            self.config.TRAIN_METADATA_PATH, load_cached_data=load_cached, split="train"
        )

        # Validation
        val_df = preprocessor.process_data(
            self.config.VAL_METADATA_PATH, load_cached_data=load_cached, split="val"
        )

        # Test
        test_df = preprocessor.process_data(
            self.config.TEST_METADATA_PATH, load_cached_data=load_cached, split="test"
        )

        return train_df, val_df, test_df

    def fit(self, train_df, val_df):
        """
        Trains the model with Early Stopping.
        """
        # Create Datasets
        train_dataset = DualStreamGnssDataset(train_df, mode="train", scaler=None)
        self.scaler = train_dataset.scaler  # Save scaler for inference

        val_dataset = DualStreamGnssDataset(val_df, mode="val", scaler=self.scaler)

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

        best_val_score = float("inf")
        patience_counter = 0

        print(f"Starting training on {self.device}...")
        print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

        for epoch in range(self.config.EPOCHS):
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                stream_a = batch["stream_a"].to(self.device)
                stream_b = batch["stream_b"].to(self.device)
                targets = batch["target"].to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(stream_a, stream_b)
                loss = self.criterion(preds, targets)

                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.GRADIENT_CLIP
                )

                self.optimizer.step()
                train_loss += loss.item() * stream_a.size(0)

            train_loss /= len(train_dataset)
            self.scheduler.step()

            # Validation
            val_score, val_loss = self.evaluate(val_loader)

            print(
                f"Epoch {epoch+1}/{self.config.EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Score (50/95 metric): {val_score:.9f}"
            )

            # Early Stopping & Model Checkpoint
            if val_score < best_val_score:
                best_val_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.MODEL_PATH)
                print("  -> New best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        print(f"Training complete. Best Validation Score: {best_val_score:.9f}")

    def evaluate(self, dataloader):
        """
        Evaluates the model on the validation set.
        Returns the competition metric (Mean of 50th and 95th percentile errors) and loss.
        """
        self.model.eval()
        total_loss = 0.0
        all_errors = []
        all_phones = []

        with torch.no_grad():
            for batch in dataloader:
                stream_a = batch["stream_a"].to(self.device)
                stream_b = batch["stream_b"].to(self.device)
                targets = batch["target"].to(self.device)

                # Forward pass (inference mode returns tensor, not dict)
                preds = self.model(stream_a, stream_b)

                # If model returns dict in eval (shouldn't based on implementation, but safe check)
                if isinstance(preds, dict):
                    preds = preds["output"]

                # Compute Loss (MAE on Cartesian residuals)
                loss = self.criterion(preds, targets)
                total_loss += loss.item() * stream_a.size(0)

                # --- Metric Calculation ---
                # 1. Get Predictions (Delta East, Delta North) -> (B, 2, L)
                # 2. Get Baseline WLS (Lat, Lon, Alt) -> (B, L, 3)
                # 3. Get GT Lat/Lon -> We need to reconstruct GT from targets or pass GT in metadata
                #    Actually, we can just reconstruct predicted Lat/Lon and compare to GT Lat/Lon
                #    But we don't have GT Lat/Lon in the batch directly, we have targets (Delta E, N).
                #    We can reconstruct Predicted Lat/Lon and use WLS+Target to get GT Lat/Lon.
                #    Or simpler: Error distance in meters is sqrt((pred_e - target_e)^2 + (pred_n - target_n)^2)
                #    This is a valid approximation for small distances. The competition metric uses Haversine,
                #    but Cartesian distance on ENU plane is very close locally.
                #    Let's use Cartesian distance for speed and simplicity in validation loop.

                # Shape: (B, 2, L) -> (B, L, 2)
                preds_np = preds.cpu().numpy().transpose(0, 2, 1)
                targets_np = targets.cpu().numpy().transpose(0, 2, 1)

                # Calculate Euclidean distance error for each point
                # error = sqrt((pred_e - true_e)^2 + (pred_n - true_n)^2)
                dists = np.sqrt(np.sum((preds_np - targets_np) ** 2, axis=2))  # (B, L)

                # We need to mask padded values
                valid_lens = batch["valid_len"].numpy()

                for i in range(len(valid_lens)):
                    length = valid_lens[i]
                    valid_dists = dists[i, :length]
                    phone = batch["phone_name"][
                        i
                    ]  # This repeats for windows, but we group later
                    drive = batch["drive_id"][i]

                    # Store errors with identifier to group by phone
                    # Note: A phone is identified by (drive_id + phone_name) in the competition context usually,
                    # but the metric says "For every phone... averaged for each phone".
                    # Usually this means per trace (trip).
                    trip_id = f"{drive}_{phone}"

                    all_errors.extend(valid_dists)
                    all_phones.extend([trip_id] * length)

        avg_loss = total_loss / len(dataloader.dataset)

        # Compute Competition Metric
        # 1. Create DataFrame
        df_metrics = pd.DataFrame({"error": all_errors, "tripId": all_phones})

        # 2. Calculate 50th and 95th percentile per phone (trip)
        scores = df_metrics.groupby("tripId")["error"].quantile([0.5, 0.95]).unstack()
        scores["avg"] = (scores[0.5] + scores[0.95]) / 2

        # 3. Mean across all phones
        final_metric = scores["avg"].mean()

        return final_metric, avg_loss

    def generate_submission(self, test_df, output_path):
        """
        Generates predictions for the test set and saves to CSV.
        """
        print("Generating submission...")

        # Load best model
        self.model.load_state_dict(
            torch.load(self.config.MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

        # Create Dataset
        # Note: We must use the SAME scaler fitted on training data
        if self.scaler is None:
            raise ValueError(
                "Scaler not found. Run fit() first or load scaler manually."
            )

        test_dataset = DualStreamGnssDataset(test_df, mode="test", scaler=self.scaler)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
        )

        results = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                stream_a = batch["stream_a"].to(self.device)
                stream_b = batch["stream_b"].to(self.device)

                # (B, 2, L)
                preds = self.model(stream_a, stream_b)
                if isinstance(preds, dict):
                    preds = preds["output"]

                preds_np = (
                    preds.cpu().numpy().transpose(0, 2, 1)
                )  # (B, L, 2) -> (dE, dN)

                # Metadata
                wls = batch["wls"].numpy()  # (B, L, 3) -> Lat, Lon, Alt
                times = batch["time"].numpy()  # (B, L)
                valid_lens = batch["valid_len"].numpy()
                drives = batch["drive_id"]
                phones = batch["phone_name"]

                for i in range(len(valid_lens)):
                    length = valid_lens[i]

                    # Get valid sequence data
                    pred_enu = preds_np[i, :length, :]  # (L_valid, 2)
                    base_wls = wls[i, :length, :]  # (L_valid, 3)
                    t_stamps = times[i, :length]
                    drive_id = drives[i]
                    phone_name = phones[i]

                    # Reconstruct Lat/Lon
                    # We iterate through points (vectorized would be faster but WGS84Utils is static per point usually)
                    # Actually WGS84Utils methods use numpy, so we can vectorize if inputs are arrays.

                    # Reference point for ENU is the WLS point itself?
                    # NO. The model predicts offset relative to the WLS baseline at THAT epoch.
                    # So for each epoch t, we have a local tangent plane centered at WLS[t].
                    # We convert (DeltaE, DeltaN, 0) back to Lat/Lon using WLS[t] as origin.

                    # Unpack
                    dE = pred_enu[:, 0]
                    dN = pred_enu[:, 1]
                    dU = np.zeros_like(dE)

                    ref_lat = base_wls[:, 0]
                    ref_lon = base_wls[:, 1]
                    ref_alt = base_wls[:, 2]

                    # Vectorized conversion
                    # enu_to_ecef
                    # x, y, z = WGS84Utils.enu_to_ecef(dE, dN, dU, ref_lat, ref_lon, ref_alt)
                    # This function expects scalars or arrays. Let's verify utils.py content.
                    # It uses np.radians, np.sin, etc. which work on arrays.

                    x_pred, y_pred, z_pred = WGS84Utils.enu_to_ecef(
                        dE, dN, dU, ref_lat, ref_lon, ref_alt
                    )

                    # ecef_to_geodetic
                    pred_lat, pred_lon, _ = WGS84Utils.ecef_to_geodetic(
                        x_pred, y_pred, z_pred
                    )

                    # Construct Trip ID
                    # Format in sample_submission: drive_id-phone_name
                    trip_id = f"{drive_id}-{phone_name}"

                    for t, lat, lon in zip(t_stamps, pred_lat, pred_lon):
                        results.append(
                            {
                                "tripId": trip_id,
                                "UnixTimeMillis": t,
                                "LatitudeDegrees": lat,
                                "LongitudeDegrees": lon,
                            }
                        )

        # Create DataFrame
        submission_df = pd.DataFrame(results)

        # The test set processing splits drives into windows. This might result in overlapping predictions
        # for the same timestamp if STRIDE < SEQ_LEN.
        # However, Config.TEST_STRIDE = SEQ_LEN (128), so no overlap except potentially the padded tail.
        # We should drop duplicates, keeping the first or averaging.
        # Since we align windows, averaging is safer for boundaries, but drop_duplicates is faster.
        # Let's average if duplicates exist.

        submission_df = submission_df.groupby(
            ["tripId", "UnixTimeMillis"], as_index=False
        ).mean()

        # Load sample submission to ensure correct order and rows
        sample_sub = pd.read_csv(
            os.path.join(self.config.INPUT_DIR, "sample_submission.csv")
        )

        # Merge to enforce structure
        final_sub = sample_sub[["tripId", "UnixTimeMillis"]].merge(
            submission_df, on=["tripId", "UnixTimeMillis"], how="left"
        )

        # Fill missing (if any) with WLS baseline?
        # We don't have WLS easily accessible here without re-reading.
        # Assuming test_df covers all timestamps in sample_submission.
        # If NaNs remain, forward fill or interpolation is a reasonable fallback.
        if final_sub.isnull().values.any():
            print("Warning: Missing predictions found. Filling with interpolation.")
            final_sub = final_sub.interpolate().ffill().bfill()

        # Save
        final_sub.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")


def run_pipeline(load_cached=True):
    set_seed(Config.SEED)

    trainer = Trainer()

    # 1. Load Data
    train_df, val_df, test_df = trainer.load_data(load_cached=load_cached)

    # 2. Train
    trainer.fit(train_df, val_df)

    # 3. Predict
    trainer.generate_submission(test_df, Config.SUBMISSION_PATH)


if __name__ == "__main__":
    # This block is for local testing only, will not be executed in the module context
    pass
