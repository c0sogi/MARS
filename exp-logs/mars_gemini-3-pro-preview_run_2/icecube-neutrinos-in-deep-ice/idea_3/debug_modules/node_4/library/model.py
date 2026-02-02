import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    SUBMISSION_DIR,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    SENSOR_GEO_PATH,
    SEQ_LEN,
    INPUT_CHANNELS,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    PATIENCE,
    NUM_WORKERS,
    DEVICE,
    STATS,
    MAX_TRAIN_SAMPLES,
    MAX_VAL_SAMPLES,
    SEED,
)
from library.utils import (
    load_geometry,
    angles_to_direction,
    direction_to_angles,
    angular_dist_score,
)

# =============================================================================
# 1. Model Architecture
# =============================================================================


class TemporalCNN(nn.Module):
    def __init__(self, in_channels=INPUT_CHANNELS, seq_len=SEQ_LEN):
        super(TemporalCNN, self).__init__()

        # Convolutional Blocks
        # Input: (B, 6, 128)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # -> (B, 64, 64)
            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # -> (B, 128, 32)
            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # -> (B, 256, 16)
            # Block 4
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # -> (B, 512, 8)
        )

        # Global Aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # -> (B, 512, 1)

        # Prediction Head
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3),  # Output vector (nx, ny, nz)
        )

    def forward(self, x):
        # x shape: (B, 6, L)
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)  # -> (B, 512)
        direction = self.regressor(x)
        return direction


# =============================================================================
# 2. Dataset
# =============================================================================


class IceCubeDataset(Dataset):
    def __init__(self, metadata_path, mode="train", max_samples=None):
        """
        Args:
            metadata_path: Path to the parquet metadata file.
            mode: 'train', 'val', or 'test'.
            max_samples: Limit the number of events (for debugging/speed).
        """
        self.mode = mode
        self.seq_len = SEQ_LEN

        # Load Metadata
        print(f"[{mode.upper()}] Loading metadata from {metadata_path}...")
        self.meta = pd.read_parquet(metadata_path)

        if max_samples is not None:
            self.meta = self.meta.iloc[:max_samples]
            print(f"[{mode.upper()}] Limited to {len(self.meta)} samples.")

        # Load Geometry
        self.geo = load_geometry()  # Index is sensor_id, cols: x, y, z

        # Cache for Batch Data
        # We use lazy loading to avoid reading all files at initialization.
        # Cite debug_lesson_1: Avoid Eager Preloading of Large File Chunks
        self.batch_data = {}

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        bid = row["batch_id"]

        # Lazy load batch data if not present
        if bid not in self.batch_data:
            base_dir = "train" if self.mode in ["train", "val"] else "test"
            batch_file = os.path.join(INPUT_DIR, base_dir, f"batch_{bid}.parquet")

            if os.path.exists(batch_file):
                df = pd.read_parquet(batch_file)
                self.batch_data[bid] = df
            else:
                self.batch_data[bid] = pd.DataFrame()

        # Retrieve pulses using the pre-calculated indices
        # Note: These indices are relative to the start of the batch file
        f_idx = row["first_pulse_index"]
        l_idx = row["last_pulse_index"]

        if not self.batch_data[bid].empty:
            # iloc is relatively fast on in-memory dataframe
            pulses = self.batch_data[bid].iloc[f_idx : l_idx + 1].copy()
            # Map Geometry
            pulses["sensor_x"] = (
                pulses["sensor_id"].map(self.geo["x"]).astype(np.float32)
            )
            pulses["sensor_y"] = (
                pulses["sensor_id"].map(self.geo["y"]).astype(np.float32)
            )
            pulses["sensor_z"] = (
                pulses["sensor_id"].map(self.geo["z"]).astype(np.float32)
            )
            pulses.fillna(0, inplace=True)
        else:
            pulses = pd.DataFrame()

        # --- Feature Engineering ---
        if len(pulses) == 0:
            # Handle empty event
            features = np.zeros((INPUT_CHANNELS, self.seq_len), dtype=np.float32)
        else:
            # 1. Prioritize highest charge
            # We want to keep the pulses with most info.
            if len(pulses) > self.seq_len:
                pulses = pulses.sort_values(by="charge", ascending=False).iloc[
                    : self.seq_len
                ]

            # 2. Sort strictly by time for 1D CNN
            pulses = pulses.sort_values(by="time", ascending=True)

            # 3. Extract columns
            # Channels: [x, y, z, time, charge, auxiliary]
            x = pulses["sensor_x"].values
            y = pulses["sensor_y"].values
            z = pulses["sensor_z"].values
            t = pulses["time"].values
            c = pulses["charge"].values
            a = pulses["auxiliary"].values.astype(float)

            # 4. Normalize
            x = (x - STATS["x_mean"]) / STATS["x_std"]
            y = (y - STATS["y_mean"]) / STATS["y_std"]
            z = (z - STATS["z_mean"]) / STATS["z_std"]
            t = (t - STATS["time_mean"]) / STATS["time_std"]
            c = np.log1p(c)  # Log transform for charge

            # Stack features: (L, 6) -> Transpose to (6, L)
            # Pad if length < seq_len
            curr_len = len(pulses)
            features = np.zeros((INPUT_CHANNELS, self.seq_len), dtype=np.float32)

            features[0, :curr_len] = x
            features[1, :curr_len] = y
            features[2, :curr_len] = z
            features[3, :curr_len] = t
            features[4, :curr_len] = c
            features[5, :curr_len] = a

        # --- Targets ---
        if self.mode != "test":
            azimuth = row["azimuth"]
            zenith = row["zenith"]
            # Convert to vector for regression
            tx, ty, tz = angles_to_direction(azimuth, zenith)
            target = np.array([tx, ty, tz], dtype=np.float32)
            return torch.tensor(features), torch.tensor(target)
        else:
            event_id = row["event_id"]
            return torch.tensor(features), event_id


# =============================================================================
# 3. Training Function
# =============================================================================


def train_model(max_train_samples=MAX_TRAIN_SAMPLES, max_val_samples=MAX_VAL_SAMPLES):
    # Set seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Initialize Datasets
    # If max_samples is None, we set a safe upper limit for the baseline to ensure runtime compliance
    # 5 Million events is substantial for a baseline.
    safe_train_limit = 5_000_000 if max_train_samples is None else max_train_samples
    safe_val_limit = 500_000 if max_val_samples is None else max_val_samples

    train_dataset = IceCubeDataset(
        TRAIN_META_PATH, mode="train", max_samples=safe_train_limit
    )
    val_dataset = IceCubeDataset(VAL_META_PATH, mode="val", max_samples=safe_val_limit)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Model Setup
    model = TemporalCNN().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    best_val_loss = float("inf")
    patience_counter = 0
    model_save_path = os.path.join(WORKING_DIR, "best_model.pth")

    print(f"\nStarting training on {DEVICE} for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0

        # Training Loop
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            preds = model(inputs)

            # Loss: 1 - Cosine Similarity
            # preds and targets are (B, 3)
            # F.cosine_similarity returns (B,)
            cos_sim = F.cosine_similarity(preds, targets, dim=1)
            loss = 1.0 - cos_sim.mean()

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)

        # Validation Loop
        model.eval()
        val_loss_sum = 0.0
        angular_errors = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                preds = model(inputs)

                # Loss
                cos_sim = F.cosine_similarity(preds, targets, dim=1)
                loss = 1.0 - cos_sim.mean()
                val_loss_sum += loss.item()

                # Metric: Mean Angular Error
                # We need to convert preds back to CPU numpy for the metric function
                # Normalize preds first
                preds_norm = F.normalize(preds, p=2, dim=1)

                # Convert vector to angles
                pred_az, pred_zen = direction_to_angles(
                    preds_norm[:, 0], preds_norm[:, 1], preds_norm[:, 2]
                )
                true_az, true_zen = direction_to_angles(
                    targets[:, 0], targets[:, 1], targets[:, 2]
                )

                # Stack for metric calc
                y_pred = torch.stack([pred_az, pred_zen], dim=1).cpu().numpy()
                y_true = torch.stack([true_az, true_zen], dim=1).cpu().numpy()

                score = angular_dist_score(y_true, y_pred)
                angular_errors.append(score)

        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_metric = np.mean(angular_errors)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.8f} | "
            f"Val Loss: {avg_val_loss:.8f} | "
            f"Val MAE: {avg_val_metric:.8f}"
        )

        # Scheduler Step
        scheduler.step(avg_val_loss)

        # Checkpoint & Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Model saved to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("  -> Early stopping triggered.")
                break

    return model_save_path


# =============================================================================
# 4. Inference Function
# =============================================================================


def predict_and_submit(model_path):
    print("\nStarting inference on Test Set...")

    # Load Model
    model = TemporalCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # We process test set batch-by-batch to be safe, but the Dataset class
    # already handles preloading.
    # Given the test set size (13M), we might want to iterate over test batches
    # manually if we can't load all 13M metadata at once.
    # However, for simplicity and consistency with the Dataset class:
    # We will load the full test metadata but the Dataset logic will handle it.
    # NOTE: Loading 13M rows of metadata is fine. Preloading 60 batch files (~10GB) is also fine.

    test_dataset = IceCubeDataset(TEST_META_PATH, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,  # Larger batch for inference
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    all_event_ids = []
    all_azimuths = []
    all_zeniths = []

    with torch.no_grad():
        for inputs, event_ids in test_loader:
            inputs = inputs.to(DEVICE)
            preds = model(inputs)

            # Normalize to unit vectors
            preds = F.normalize(preds, p=2, dim=1)

            # Convert to angles
            az, zen = direction_to_angles(preds[:, 0], preds[:, 1], preds[:, 2])

            all_event_ids.extend(event_ids.numpy())
            all_azimuths.extend(az.cpu().numpy())
            all_zeniths.extend(zen.cpu().numpy())

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"event_id": all_event_ids, "azimuth": all_azimuths, "zenith": all_zeniths}
    )

    # Sort by event_id to match sample submission format usually
    submission_df.sort_values("event_id", inplace=True)

    out_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}. Shape: {submission_df.shape}")


# =============================================================================
# 5. Main Execution Wrapper
# =============================================================================


def run_pipeline():
    # 1. Train
    best_model_path = train_model()

    # 2. Predict
    predict_and_submit(best_model_path)
