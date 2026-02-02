import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import shutil

# Import library components
from library.config import Config
from library.utils import set_seed, enu_to_geodetic
from library.dataset import GnssSequenceDataset
from library.model import ResUNet1D
from library.engine import train_one_epoch, evaluate


# --- 1. Setup and Configuration ---
class DemoConfig(Config):
    """
    Configuration overrides for the demonstration run.
    """

    # Use working directory for outputs to avoid permission issues
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "demo_cache")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "demo_submission")

    # Training hyperparameters for speed
    BATCH_SIZE = 2
    EPOCHS = 1
    NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure these exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)


def create_mini_metadata(source_path, dest_path, num_drives=1):
    """
    Creates a smaller metadata file containing only a few drives to speed up loading.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source metadata not found: {source_path}")

    df = pd.read_csv(source_path)

    # Get unique drives
    all_drives = sorted(df["drive_id"].unique().tolist())

    # Prioritize known good drives for demonstration stability
    known_good_train = ["2020-05-15-US-MTV-1", "2020-05-21-US-MTV-1"]
    known_good_test = ["2020-06-04-US-MTV-1"]

    selected_drives = []
    # Add prioritized drives if they exist in source
    for d in known_good_train + known_good_test:
        if d in all_drives and d not in selected_drives:
            selected_drives.append(d)

    # Fill remaining slots with other drives
    for d in all_drives:
        if len(selected_drives) >= num_drives:
            break
        if d not in selected_drives:
            selected_drives.append(d)

    df_mini = df[df["drive_id"].isin(selected_drives)].copy()

    df_mini.to_csv(dest_path, index=False)
    print(
        f"Created mini metadata at {dest_path} with {len(df_mini)} rows (Drives: {df_mini['drive_id'].unique()})"
    )
    return df_mini


# --- Main Execution ---
if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)

    # Define paths
    train_meta_src = os.path.join(Config.METADATA_DIR, "train_metadata.csv")
    test_meta_src = os.path.join(Config.METADATA_DIR, "test_metadata.csv")

    mini_train_meta_path = os.path.join(DemoConfig.WORKING_DIR, "mini_train_meta.csv")
    mini_test_meta_path = os.path.join(DemoConfig.WORKING_DIR, "mini_test_meta.csv")

    print("--- 1. Data Preparation ---")
    # Create mini metadata files
    # Increase num_drives to ensure we get valid data even if some are empty
    create_mini_metadata(train_meta_src, mini_train_meta_path, num_drives=3)
    create_mini_metadata(test_meta_src, mini_test_meta_path, num_drives=2)

    # Patch the Config used by library modules to point to our demo cache
    # This is necessary because the library modules import Config directly
    Config.CACHE_DIR = DemoConfig.CACHE_DIR

    print("\n--- 2. Dataset Loading & Verification ---")
    # Instantiate Training Dataset
    train_dataset = GnssSequenceDataset(
        metadata_path=mini_train_meta_path,
        split="train",
        max_seq_len=128,  # Force small sequence length for speed
        load_cached_data=False,  # Force processing to demonstrate logic
    )

    # Verify dataset is not empty
    assert len(train_dataset) > 0, "Training dataset is empty!"

    # Get a sample
    features, targets, mask, info = train_dataset[0]

    print(f"Sample Feature Shape: {features.shape} (Channels, Length)")
    print(f"Sample Target Shape: {targets.shape} (Channels, Length)")
    print(f"Sample Mask Shape: {mask.shape} (Length)")

    # Assertions
    assert (
        features.shape[0] == DemoConfig.IN_CHANNELS
    ), f"Expected {DemoConfig.IN_CHANNELS} input channels"
    assert (
        targets.shape[0] == DemoConfig.OUT_CHANNELS
    ), f"Expected {DemoConfig.OUT_CHANNELS} output channels"
    assert features.shape[1] == train_dataset.max_seq_len, "Sequence length mismatch"
    assert not torch.isnan(features).any(), "Features contain NaNs"

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=DemoConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=DemoConfig.NUM_WORKERS,
    )

    print("\n--- 3. Model Initialization ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ResUNet1D(config=DemoConfig).to(device)

    # Dummy forward pass
    dummy_input = features.unsqueeze(0).to(device)  # Add batch dim
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"Model Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (
        1,
        DemoConfig.OUT_CHANNELS,
        train_dataset.max_seq_len,
    ), "Model output shape mismatch"

    print("\n--- 4. Training Loop Demonstration ---")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=DemoConfig.LEARNING_RATE,
        weight_decay=DemoConfig.WEIGHT_DECAY,
    )

    # Train for 1 epoch
    avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch=1)
    print(f"Epoch 1 Training Loss (MAE): {avg_loss:.4f}")

    # Verify loss is valid
    assert avg_loss > 0, "Training loss should be positive"
    assert not np.isnan(avg_loss), "Training loss is NaN"

    print("\n--- 5. Evaluation Demonstration ---")
    # Use the same loader for validation in this demo
    val_loss, val_score = evaluate(model, train_loader, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Competition Score (Mean 50th/95th percentile error): {val_score:.4f}")

    print("\n--- 6. Inference and Coordinate Reconstruction ---")
    # Load Test Dataset
    test_dataset = GnssSequenceDataset(
        metadata_path=mini_test_meta_path,
        split="test",
        max_seq_len=128,
        load_cached_data=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=DemoConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=DemoConfig.NUM_WORKERS,
    )

    model.eval()
    results = []

    with torch.no_grad():
        for batch_idx, (features, _, mask, info) in enumerate(test_loader):
            features = features.to(device)
            mask = mask.to(device)

            # Predict ENU residuals
            outputs = model(features)  # (B, 2, L) -> DeltaEast, DeltaNorth

            outputs_np = outputs.cpu().numpy()
            mask_np = mask.cpu().numpy()
            metadata_np = info[
                "metadata"
            ].numpy()  # (B, L, 4) -> Time, WlsLat, WlsLon, WlsAlt

            batch_size = features.size(0)

            for i in range(batch_size):
                # Get valid length
                valid_len = int(mask_np[i].sum())

                # Extract valid predictions and metadata
                pred_east = outputs_np[i, 0, :valid_len]
                pred_north = outputs_np[i, 1, :valid_len]

                # Metadata: [UnixTimeMillis, WlsLat, WlsLon, WlsAlt]
                timestamps = metadata_np[i, :valid_len, 0]
                wls_lat = metadata_np[i, :valid_len, 1]
                wls_lon = metadata_np[i, :valid_len, 2]
                wls_alt = metadata_np[i, :valid_len, 3]

                # Reconstruct Geodetic Coordinates
                # We assume DeltaUp is 0 for the prediction, utilizing WLS altitude
                pred_lat, pred_lon, _ = enu_to_geodetic(
                    pred_east,
                    pred_north,
                    np.zeros_like(pred_east),
                    wls_lat,
                    wls_lon,
                    wls_alt,
                )

                # Store results
                drive_id = info["drive_id"][i]
                phone_name = info["phone_name"][i]
                trip_id = f"{drive_id}-{phone_name}"

                for t, lat, lon in zip(timestamps, pred_lat, pred_lon):
                    results.append(
                        {
                            "tripId": trip_id,
                            "UnixTimeMillis": int(t),
                            "LatitudeDegrees": lat,
                            "LongitudeDegrees": lon,
                        }
                    )

    # Create Submission DataFrame
    submission_df = pd.DataFrame(results)

    # Verify Submission
    print(f"Generated {len(submission_df)} predictions.")
    if not submission_df.empty:
        print("Sample Predictions:")
        print(submission_df.head())

        # Check for NaNs
        assert not submission_df.isnull().values.any(), "Submission contains NaNs"

        # Save submission
        sub_path = os.path.join(DemoConfig.SUBMISSION_DIR, "demo_submission.csv")
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print("Warning: No predictions generated (empty test set?)")

    print("\n--- Demonstration Complete ---")
