import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from library.config import Config
from library.utils import (
    load_sensor_geometry,
    angular_dist_score,
    angles_to_direction,
    direction_to_angles,
)


# Set seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)

# -----------------------------------------------------------------------------
# Model Definition
# -----------------------------------------------------------------------------


class PointNetBaseline(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, output_dim=3, dropout=0.1):
        super(PointNetBaseline, self).__init__()

        # Encoder: Processes each pulse independently
        # Input: (B, N, 6) -> Output: (B, N, hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
        )

        # Regressor: Processes the global feature vector
        # Input: (B, hidden_dim) -> Output: (B, output_dim)
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        # x shape: (Batch, Pulses, Features)

        # 1. Pulse Encoding
        # Apply MLP to each pulse. Weights are shared.
        x = self.encoder(x)  # (B, N, H)

        # 2. Symmetric Aggregation (Global Max Pooling)
        # Max over the pulse dimension (dim=1)
        x, _ = torch.max(x, dim=1)  # (B, H)

        # 3. Direction Regression
        direction = self.regressor(x)  # (B, 3)

        return direction


# -----------------------------------------------------------------------------
# Data Processing & Caching
# -----------------------------------------------------------------------------


def process_batch_data(
    batch_id, meta_df, sensor_geo, mode="train", load_cached_data=True
):
    """
    Loads a batch file, processes it into a tensor of shape (N_events, 128, 6),
    and returns the tensor and targets.
    Implements caching to disk.
    """
    cache_dir = os.path.join(Config.WORKING_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    cache_path_x = os.path.join(cache_dir, f"{mode}_batch_{batch_id}_x.pt")
    cache_path_y = os.path.join(cache_dir, f"{mode}_batch_{batch_id}_y.pt")
    cache_path_ids = os.path.join(cache_dir, f"{mode}_batch_{batch_id}_ids.pt")

    # Try loading from cache
    if (
        load_cached_data
        and os.path.exists(cache_path_x)
        and os.path.exists(cache_path_ids)
    ):
        # If mode is train/val, we also need y
        if mode != "test" and not os.path.exists(cache_path_y):
            pass  # Cache incomplete, recompute
        else:
            try:
                # Using weights_only=True for security if supported, else fallback
                try:
                    X = torch.load(cache_path_x, weights_only=True)
                    ids = torch.load(cache_path_ids, weights_only=True)
                    y = None
                    if mode != "test":
                        y = torch.load(cache_path_y, weights_only=True)
                except TypeError:
                    # Fallback for older torch versions
                    X = torch.load(cache_path_x)
                    ids = torch.load(cache_path_ids)
                    y = None
                    if mode != "test":
                        y = torch.load(cache_path_y)

                return X, y, ids
            except Exception:
                pass  # Corrupt cache, recompute

    # ---------------------------------------------------------
    # Compute from scratch
    # ---------------------------------------------------------

    # Filter metadata for this batch
    batch_meta = meta_df[meta_df["batch_id"] == batch_id].copy()
    if batch_meta.empty:
        return None, None, None

    # Load Raw Batch Data
    batch_file = os.path.join(Config.INPUT_DIR, mode, f"batch_{batch_id}.parquet")
    if not os.path.exists(batch_file):
        return None, None, None

    df_batch = pd.read_parquet(batch_file)

    # Merge Sensor Geometry
    df_batch = df_batch.join(sensor_geo, on="sensor_id", how="left")

    # Feature Engineering on the whole batch dataframe (vectorized)
    # 1. Auxiliary: bool -> float
    df_batch["auxiliary"] = df_batch["auxiliary"].astype(np.float32)

    # 2. Charge: log10(charge) (handle small values)
    df_batch["charge"] = np.log10(np.clip(df_batch["charge"], 1e-3, None)).astype(
        np.float32
    )

    # 3. Coordinates: Scale
    df_batch["x"] /= Config.COORD_SCALE
    df_batch["y"] /= Config.COORD_SCALE
    df_batch["z"] /= Config.COORD_SCALE

    # Convert to Numpy for fast slicing
    # Columns: x, y, z, time, charge, auxiliary
    feature_cols = ["x", "y", "z", "time", "charge", "auxiliary"]
    data_arr = df_batch[feature_cols].to_numpy(dtype=np.float32)

    # Prepare Output Arrays
    num_events = len(batch_meta)
    X_tensor = np.zeros((num_events, Config.NUM_PULSES, 6), dtype=np.float32)
    event_ids = batch_meta["event_id"].values

    starts = batch_meta["first_pulse_index"].values
    ends = batch_meta["last_pulse_index"].values

    # Loop over events
    for i in range(num_events):
        s, e = starts[i], ends[i]
        event_pulses = data_arr[s : e + 1]

        # Normalize Time: t = (t - t_min) / scale
        if len(event_pulses) > 0:
            t_min = np.min(event_pulses[:, 3])
            event_pulses[:, 3] = (event_pulses[:, 3] - t_min) / Config.TIME_SCALE

        # Sampling Strategy: Top N by charge (index 4)
        n_pulses = event_pulses.shape[0]
        if n_pulses > Config.NUM_PULSES:
            idx = np.argsort(event_pulses[:, 4])
            top_idx = idx[-Config.NUM_PULSES :]
            selected = event_pulses[top_idx]
            X_tensor[i, :, :] = selected
        else:
            X_tensor[i, :n_pulses, :] = event_pulses

    # Convert to Tensor
    X_t = torch.tensor(X_tensor, dtype=torch.float32)
    ids_t = torch.tensor(event_ids, dtype=torch.int64)

    # Handle Targets
    y_t = None
    if mode != "test":
        az = batch_meta["azimuth"].values.astype(np.float32)
        ze = batch_meta["zenith"].values.astype(np.float32)
        y_t = torch.tensor(np.stack([az, ze], axis=1), dtype=torch.float32)

    # Save to Cache
    torch.save(X_t, cache_path_x)
    torch.save(ids_t, cache_path_ids)
    if y_t is not None:
        torch.save(y_t, cache_path_y)

    return X_t, y_t, ids_t


# -----------------------------------------------------------------------------
# Training Logic
# -----------------------------------------------------------------------------


def cosine_loss(pred_vectors, target_angles):
    """
    pred_vectors: (B, 3) unnormalized
    target_angles: (B, 2) azimuth, zenith
    """
    pred_norm = F.normalize(pred_vectors, p=2, dim=1)
    target_vectors = angles_to_direction(target_angles[:, 0], target_angles[:, 1])
    target_vectors = target_vectors.to(pred_vectors.device)

    # Loss = 1 - cos(theta)
    cos_sim = torch.sum(pred_norm * target_vectors, dim=1)
    loss = 1.0 - cos_sim.mean()
    return loss


def train_model(config=Config):
    print("Initializing Training...")

    # 1. Load Metadata
    train_meta = pd.read_parquet(config.TRAIN_META)
    val_meta = pd.read_parquet(config.VAL_META)

    if config.DEBUG:
        train_meta = train_meta.iloc[: config.DEBUG_SUBSET_SIZE]
        val_meta = val_meta.iloc[: config.DEBUG_SUBSET_SIZE]
        print(f"DEBUG Mode: Reduced train size to {len(train_meta)}")

    train_batches = train_meta["batch_id"].unique()
    val_batches = val_meta["batch_id"].unique()

    sensor_geo = load_sensor_geometry(config.SENSOR_GEOMETRY_PATH)

    device = torch.device(config.DEVICE)
    model = PointNetBaseline(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
        dropout=config.DROPOUT,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, verbose=True
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {len(train_batches)} batches...")

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss_sum = 0
        train_batches_count = 0

        np.random.shuffle(train_batches)

        for batch_id in train_batches:
            X, y, _ = process_batch_data(
                batch_id, train_meta, sensor_geo, mode="train", load_cached_data=True
            )
            if X is None:
                continue

            dataset = TensorDataset(X, y)
            loader = DataLoader(
                dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
            )

            batch_loss_accum = 0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                preds = model(X_batch)
                loss = cosine_loss(preds, y_batch)
                loss.backward()
                optimizer.step()

                batch_loss_accum += loss.item()

            train_loss_sum += batch_loss_accum / len(loader)
            train_batches_count += 1

            del X, y, dataset, loader

        avg_train_loss = train_loss_sum / max(1, train_batches_count)

        # Validation
        model.eval()
        val_loss_sum = 0
        val_mae_sum = 0
        val_batches_count = 0

        with torch.no_grad():
            for batch_id in val_batches:
                X_val, y_val, _ = process_batch_data(
                    batch_id, val_meta, sensor_geo, mode="train", load_cached_data=True
                )
                if X_val is None:
                    continue

                dataset = TensorDataset(X_val, y_val)
                loader = DataLoader(
                    dataset, batch_size=config.BATCH_SIZE, shuffle=False
                )

                for X_b, y_b in loader:
                    X_b = X_b.to(device)
                    y_b = y_b.to(device)

                    preds = model(X_b)
                    loss = cosine_loss(preds, y_b)
                    val_loss_sum += loss.item()

                    pred_vecs = F.normalize(preds, p=2, dim=1).cpu()
                    az_pred, zen_pred = direction_to_angles(pred_vecs)
                    y_pred_np = torch.stack([az_pred, zen_pred], dim=1).numpy()
                    y_true_np = y_b.cpu().numpy()

                    mae = angular_dist_score(y_true_np, y_pred_np)
                    val_mae_sum += mae

                val_batches_count += len(loader)
                del X_val, y_val, dataset, loader

        avg_val_loss = val_loss_sum / max(1, val_batches_count)
        avg_val_mae = val_mae_sum / max(1, val_batches_count)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {avg_train_loss} | Val Loss: {avg_val_loss} | Val MAE: {avg_val_mae}"
        )

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print("Early stopping triggered.")
                break

    return model


# -----------------------------------------------------------------------------
# Inference Logic
# -----------------------------------------------------------------------------


def generate_submission(config=Config):
    print("Starting Inference...")

    test_meta = pd.read_parquet(config.TEST_META)
    test_batches = test_meta["batch_id"].unique()

    sensor_geo = load_sensor_geometry(config.SENSOR_GEOMETRY_PATH)

    device = torch.device(config.DEVICE)
    model = PointNetBaseline(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
    ).to(device)

    if os.path.exists(config.MODEL_PATH):
        try:
            model.load_state_dict(
                torch.load(config.MODEL_PATH, map_location=device, weights_only=True)
            )
        except TypeError:
            model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    else:
        print("Warning: No trained model found. Using random weights.")

    model.eval()

    all_event_ids = []
    all_azimuths = []
    all_zeniths = []

    with torch.no_grad():
        for batch_id in test_batches:
            X, _, ids = process_batch_data(
                batch_id, test_meta, sensor_geo, mode="test", load_cached_data=True
            )
            if X is None:
                continue

            dataset = TensorDataset(X)
            loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

            batch_preds = []
            for (X_b,) in loader:
                X_b = X_b.to(device)
                preds = model(X_b)
                preds = F.normalize(preds, p=2, dim=1)
                batch_preds.append(preds.cpu())

            batch_preds = torch.cat(batch_preds, dim=0)
            az, zen = direction_to_angles(batch_preds)

            all_event_ids.append(ids.numpy())
            all_azimuths.append(az.numpy())
            all_zeniths.append(zen.numpy())

            del X, ids, dataset, loader, batch_preds
            gc.collect()

    final_ids = np.concatenate(all_event_ids)
    final_az = np.concatenate(all_azimuths)
    final_zen = np.concatenate(all_zeniths)

    submission_df = pd.DataFrame(
        {"event_id": final_ids, "azimuth": final_az, "zenith": final_zen}
    )

    submission_df = submission_df.sort_values("event_id")
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
