import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# Import provided library modules
from library.config import Config
from library.utils import TargetScaler, seed_everything
from library.data_processing import process_and_cache_features
from library.dataset import VolcanoDataset
from library.model import ChannelAdaptiveHybridModel
from library.trainer import Trainer


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Configuration and Setup
    # -------------------------------------------------------------------------
    print(">>> Setting up Demo Configuration...")

    # Define a specific working directory for this demo
    DEMO_WORKING_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    # Create subdirectories
    os.makedirs(os.path.join(DEMO_WORKING_DIR, "metadata"), exist_ok=True)
    os.makedirs(os.path.join(DEMO_WORKING_DIR, "working"), exist_ok=True)
    os.makedirs(os.path.join(DEMO_WORKING_DIR, "submission"), exist_ok=True)

    # Override Config parameters for speed
    Config.WORKING_DIR = os.path.join(DEMO_WORKING_DIR, "working")
    Config.SUBMISSION_DIR = os.path.join(DEMO_WORKING_DIR, "submission")
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Process only 10 files

    # Update Artifact Paths in Config to point to new working dir
    Config.TRAIN_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "train_features.parquet"
    )
    Config.VAL_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "test_features.parquet"
    )
    Config.TARGET_MEAN_PATH = os.path.join(Config.WORKING_DIR, "target_mean.npy")
    Config.TARGET_STD_PATH = os.path.join(Config.WORKING_DIR, "target_std.npy")
    Config.STATS_SCALER_MEAN_PATH = os.path.join(
        Config.WORKING_DIR, "stats_scaler_mean.npy"
    )
    Config.STATS_SCALER_SCALE_PATH = os.path.join(
        Config.WORKING_DIR, "stats_scaler_scale.npy"
    )
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Initialize environment
    Config.setup()
    seed_everything(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Prepare Mini-Metadata (Subsetting)
    # -------------------------------------------------------------------------
    print(">>> Creating Mini-Metadata for Speed...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Subset
    mini_train = orig_train.head(Config.DEBUG_SAMPLE_SIZE).copy()
    mini_val = orig_val.head(Config.DEBUG_SAMPLE_SIZE).copy()
    mini_test = orig_test.head(Config.DEBUG_SAMPLE_SIZE).copy()

    # Save to demo directory
    mini_train_path = os.path.join(DEMO_WORKING_DIR, "metadata", "train.csv")
    mini_val_path = os.path.join(DEMO_WORKING_DIR, "metadata", "val.csv")
    mini_test_path = os.path.join(DEMO_WORKING_DIR, "metadata", "test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # Update Config to point to mini metadata
    Config.TRAIN_METADATA = mini_train_path
    Config.VAL_METADATA = mini_val_path
    Config.TEST_METADATA = mini_test_path

    print(
        f"Subset sizes -> Train: {len(mini_train)}, Val: {len(mini_val)}, Test: {len(mini_test)}"
    )

    # -------------------------------------------------------------------------
    # 3. Feature Processing Demonstration
    # -------------------------------------------------------------------------
    print(">>> Processing Features...")

    # Process Train
    df_train_stats = process_and_cache_features(
        Config.TRAIN_METADATA, Config.TRAIN_FEATURES_PATH, load_cached_data=False
    )
    # Process Val
    df_val_stats = process_and_cache_features(
        Config.VAL_METADATA, Config.VAL_FEATURES_PATH, load_cached_data=False
    )

    # Validation
    assert (
        len(df_train_stats) == Config.DEBUG_SAMPLE_SIZE
    ), "Train features count mismatch"
    assert (
        "sensor_1_mean" in df_train_stats.columns
    ), "Feature extraction failed to create sensor columns"
    assert (
        "time_to_eruption" in df_train_stats.columns
    ), "Target column missing in train features"

    # -------------------------------------------------------------------------
    # 4. Scaler Logic Demonstration
    # -------------------------------------------------------------------------
    print(">>> Verifying Scaler Logic...")

    # A. Target Scaler
    target_scaler = TargetScaler()
    targets = df_train_stats["time_to_eruption"].values
    target_scaler.fit(targets)

    # Test transform -> inverse_transform
    scaled = target_scaler.transform(targets)
    inversed = target_scaler.inverse_transform(scaled)

    # Check reconstruction
    assert np.allclose(
        targets, inversed, atol=1e-5
    ), "TargetScaler inverse transform failed"

    # Save and Load
    target_scaler.save(Config.TARGET_MEAN_PATH, Config.TARGET_STD_PATH)
    loaded_scaler = TargetScaler().load(Config.TARGET_MEAN_PATH, Config.TARGET_STD_PATH)
    assert np.allclose(
        target_scaler.scaler.mean_, loaded_scaler.scaler.mean_
    ), "TargetScaler save/load mismatch"

    # B. Stats Scaler
    feature_cols = sorted(
        [
            c
            for c in df_train_stats.columns
            if c not in ["segment_id", "time_to_eruption"]
        ]
    )
    stats_scaler = StandardScaler()
    stats_scaler.fit(df_train_stats[feature_cols].values)

    # Save stats scaler
    np.save(Config.STATS_SCALER_MEAN_PATH, stats_scaler.mean_)
    np.save(Config.STATS_SCALER_SCALE_PATH, stats_scaler.scale_)

    # -------------------------------------------------------------------------
    # 5. Dataset & DataLoader Demonstration
    # -------------------------------------------------------------------------
    print(">>> Verifying Dataset and DataLoader...")

    train_dataset = VolcanoDataset(
        metadata_df=mini_train,
        stats_df=df_train_stats,
        augment=True,
        target_scaler=target_scaler,
        stats_scaler=stats_scaler,
    )

    # Check length
    assert len(train_dataset) == Config.DEBUG_SAMPLE_SIZE

    # Check item structure
    spec, stats, target = train_dataset[0]

    # Spectrogram shape: [Channels, n_mels, Time]
    # Time dimension depends on audio length (60001 samples) / hop_length (256) ~ 235
    assert spec.shape[0] == 10, f"Expected 10 channels, got {spec.shape[0]}"
    assert (
        spec.shape[1] == Config.N_MELS
    ), f"Expected {Config.N_MELS} mels, got {spec.shape[1]}"
    assert isinstance(spec, torch.Tensor)

    # Stats shape
    assert stats.shape[0] == len(feature_cols)

    # Target shape
    assert target.ndim == 0 or target.shape == (1,)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    batch_spec, batch_stats, batch_target = next(iter(train_loader))

    assert batch_spec.shape[0] == Config.BATCH_SIZE
    print(f"Batch Spectrogram Shape: {batch_spec.shape}")

    # -------------------------------------------------------------------------
    # 6. Model Demonstration
    # -------------------------------------------------------------------------
    print(">>> Verifying Model Architecture...")

    model = ChannelAdaptiveHybridModel(num_stats_features=len(feature_cols))
    model.to(Config.DEVICE)

    # Forward pass
    with torch.no_grad():
        output = model(batch_spec.to(Config.DEVICE), batch_stats.to(Config.DEVICE))

    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch: {output.shape}"
    print("Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 7. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print(">>> Running Training Loop (Mini-Batch)...")

    # Prepare Val Dataset/Loader
    val_dataset = VolcanoDataset(
        metadata_df=mini_val,
        stats_df=df_val_stats,
        augment=False,
        target_scaler=target_scaler,
        stats_scaler=stats_scaler,
    )
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Initialize Trainer
    trainer = Trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        target_scaler=target_scaler,
        num_stats_features=len(feature_cols),
    )

    # Run Fit
    trainer.fit()

    # Check if model saved
    assert os.path.exists(Config.MODEL_PATH), "Model file was not saved after training."
    print("Training loop completed.")

    # -------------------------------------------------------------------------
    # 8. Inference Demonstration
    # -------------------------------------------------------------------------
    print(">>> Running Inference Demonstration...")

    # Process Test Features
    df_test_stats = process_and_cache_features(
        Config.TEST_METADATA, Config.TEST_FEATURES_PATH, load_cached_data=False
    )

    # Test Dataset (No target)
    test_dataset = VolcanoDataset(
        metadata_df=mini_test,
        stats_df=df_test_stats,
        augment=False,
        target_scaler=None,  # Not needed for input
        stats_scaler=stats_scaler,
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Load Model
    inference_model = ChannelAdaptiveHybridModel(num_stats_features=len(feature_cols))
    inference_model.load_state_dict(
        torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
    )
    inference_model.to(Config.DEVICE)
    inference_model.eval()

    predictions = []
    segment_ids = []

    with torch.no_grad():
        for i, (spectrogram, stats) in enumerate(test_loader):
            spectrogram = spectrogram.to(Config.DEVICE)
            stats = stats.to(Config.DEVICE)

            output = inference_model(spectrogram, stats)

            # Inverse transform
            pred_np = output.cpu().numpy().flatten()
            pred_orig = target_scaler.inverse_transform(pred_np)

            predictions.extend(pred_orig)

            # Get segment IDs for this batch
            # Calculate indices based on batch size
            start_idx = i * Config.BATCH_SIZE
            end_idx = start_idx + len(spectrogram)
            batch_ids = mini_test.iloc[start_idx:end_idx]["segment_id"].values
            segment_ids.extend(batch_ids)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"segment_id": segment_ids, "time_to_eruption": predictions}
    )

    print("Sample Submission Head:")
    print(submission_df.head())

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    print("\n>>> Demo Completed Successfully!")


if __name__ == "__main__":
    run_demo()
