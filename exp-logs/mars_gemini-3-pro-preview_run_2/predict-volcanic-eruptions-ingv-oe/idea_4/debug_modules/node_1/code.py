import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, TargetScaler
from library.dataset import VolcanoDataset
from library.model import AttentionPooledHybridEfficientNet
from library.train import run_training
from library.predict import generate_predictions


def create_mini_metadata():
    """
    Creates small subsets of the original metadata to speed up
    feature engineering and training for this demonstration.
    """
    print("\n[Demo] Creating mini metadata files...")

    # Define paths
    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test.csv")

    # Load original metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)
    df_test = pd.read_csv(Config.TEST_METADATA)

    # Sample (20 for train, 10 for val/test)
    df_train_mini = df_train.head(20).copy()
    df_val_mini = df_val.head(10).copy()
    df_test_mini = df_test.head(10).copy()

    # Save to working directory
    df_train_mini.to_csv(mini_train_path, index=False)
    df_val_mini.to_csv(mini_val_path, index=False)
    df_test_mini.to_csv(mini_test_path, index=False)

    print(f"[Demo] Mini metadata saved to {Config.WORKING_DIR}")

    return mini_train_path, mini_val_path, mini_test_path


def verify_target_scaler():
    """
    Verifies the functionality of the TargetScaler class.
    """
    print("\n[Demo] Verifying TargetScaler...")
    scaler = TargetScaler()

    # Create dummy data
    dummy_targets = np.array([10, 20, 30, 40, 50], dtype=np.float32)

    # Fit
    scaler.fit(dummy_targets)
    assert scaler.is_fitted, "Scaler should be fitted after calling fit()"
    assert os.path.exists(scaler.mean_path), "Scaler mean file should be saved"
    assert os.path.exists(scaler.std_path), "Scaler std file should be saved"

    # Transform
    transformed = scaler.transform(dummy_targets)
    # Standard scaling: mean=30, std=14.14
    # (10-30)/14.14 = -1.414
    assert np.isclose(transformed.mean(), 0, atol=1e-5), "Transformed mean should be 0"
    assert np.isclose(transformed.std(), 1, atol=1e-5), "Transformed std should be 1"

    # Inverse Transform (Numpy)
    inverted = scaler.inverse_transform(transformed).flatten()
    assert np.allclose(
        dummy_targets, inverted, atol=1e-5
    ), "Inverse transform should recover original values"

    # Inverse Transform (Torch)
    transformed_tensor = torch.tensor(transformed)
    inverted_tensor = scaler.inverse_transform(transformed_tensor)
    assert torch.is_tensor(inverted_tensor), "Output should be a tensor"
    assert np.allclose(
        dummy_targets, inverted_tensor.numpy().flatten(), atol=1e-5
    ), "Tensor inverse transform failed"

    print("[Demo] TargetScaler verification passed.")


def verify_dataset_logic(train_meta_path):
    """
    Verifies the VolcanoDataset loading, feature generation, and shapes.
    """
    print("\n[Demo] Verifying VolcanoDataset...")

    # Initialize dataset (this triggers feature engineering)
    ds = VolcanoDataset(metadata_path=train_meta_path, mode="train", target_scaler=None)

    assert len(ds) == 20, f"Dataset length mismatch. Expected 20, got {len(ds)}"

    # Fetch one sample
    sample = ds[0]

    # Check keys
    expected_keys = {"spectrogram", "features", "target", "segment_id"}
    assert expected_keys.issubset(sample.keys()), "Dataset sample missing keys"

    # Check Spectrogram Shape: [Channels, Freq, Time]
    # Channels=10, Freq=128 (N_MELS), Time depends on signal length/hop (~235)
    spec = sample["spectrogram"]
    assert spec.ndim == 3, "Spectrogram should be 3D"
    assert spec.shape[0] == 10, f"Expected 10 channels, got {spec.shape[0]}"
    assert spec.shape[1] == 128, f"Expected 128 mel bins, got {spec.shape[1]}"

    # Check Features Shape: [100]
    feats = sample["features"]
    assert feats.ndim == 1, "Features should be 1D"
    assert feats.shape[0] == 100, f"Expected 100 stat features, got {feats.shape[0]}"

    # Check Target
    target = sample["target"]
    assert target.ndim == 0, "Target should be a scalar tensor"

    print("[Demo] VolcanoDataset verification passed.")


def verify_model_logic():
    """
    Verifies the AttentionPooledHybridEfficientNet architecture.
    """
    print("\n[Demo] Verifying Model Architecture...")

    model = AttentionPooledHybridEfficientNet()
    model.eval()

    # Create dummy batch
    # Batch size 2
    dummy_spec = torch.randn(2, 10, 128, 235)
    dummy_feats = torch.randn(2, 100)

    with torch.no_grad():
        output = model(dummy_spec, dummy_feats)

    # Check output shape: [Batch, 1]
    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"

    print("[Demo] Model architecture verification passed.")


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)

    # 2. Configure for Demo (Override Config)
    # We use a separate sub-directory in working for this demo run
    Config.IDEA_NAME = "demo_execution"
    Config.WORKING_DIR = os.path.join("./working", Config.IDEA_NAME)
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.setup()  # Re-run setup to create new dirs

    # Override paths for outputs
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Override scaler paths to avoid overwriting production scalers
    Config.TARGET_SCALER_MEAN = os.path.join(Config.WORKING_DIR, "target_mean.npy")
    Config.TARGET_SCALER_STD = os.path.join(Config.WORKING_DIR, "target_std.npy")
    Config.STATS_SCALER_MEAN = os.path.join(Config.WORKING_DIR, "stats_scaler_mean.npy")
    Config.STATS_SCALER_SCALE = os.path.join(
        Config.WORKING_DIR, "stats_scaler_scale.npy"
    )

    # Override feature cache paths
    Config.TRAIN_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "train_features.parquet"
    )
    Config.VAL_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "test_features.parquet"
    )

    # Hyperparameters for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = False  # We are manually creating mini datasets, so we don't need the internal debug flag

    # 3. Create Mini Metadata
    mini_train, mini_val, mini_test = create_mini_metadata()

    # Point Config to mini metadata
    Config.TRAIN_METADATA = mini_train
    Config.VAL_METADATA = mini_val
    Config.TEST_METADATA = mini_test

    # 4. Run Verifications
    verify_target_scaler()
    verify_dataset_logic(mini_train)
    verify_model_logic()

    # 5. Run Training Pipeline
    print("\n[Demo] Starting Training Pipeline...")
    run_training(
        num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE, debug=False
    )

    # Verify model was saved
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError("Model checkpoint was not created after training.")
    print(f"[Demo] Training complete. Model saved to {Config.MODEL_SAVE_PATH}")

    # 6. Run Inference Pipeline
    print("\n[Demo] Starting Inference Pipeline...")
    generate_predictions(batch_size=Config.BATCH_SIZE)

    # Verify submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created after inference.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"[Demo] Submission generated with {len(df_sub)} rows.")

    # Check format
    assert "segment_id" in df_sub.columns and "time_to_eruption" in df_sub.columns
    assert len(df_sub) == 10  # We used mini_test with 10 rows

    print("\n[Demo] All tasks completed successfully.")
