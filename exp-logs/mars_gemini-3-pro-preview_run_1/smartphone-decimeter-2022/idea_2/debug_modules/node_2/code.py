import os
import sys
import numpy as np
import pandas as pd
import torch
import random
import warnings

# Add the current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.model import SensorFusionTCN, ecef_to_lla
from library.data_loader import SmartphoneDataset
from library.trainer import train_model
from library.inference import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_subset_metadata():
    """
    Creates small subset CSVs of the metadata to allow for rapid
    demonstration of the training and inference pipelines.
    """
    print("\n[Demo] Creating metadata subsets for rapid execution...")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Train Subset
    if os.path.exists(Config.TRAIN_METADATA_PATH):
        df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
        # Pick the first drive found
        if not df_train.empty:
            first_drive = df_train["drive_id"].unique()[0]
            df_train_subset = df_train[df_train["drive_id"] == first_drive].head(
                500
            )  # Take first 500 rows
            subset_train_path = os.path.join(Config.WORKING_DIR, "demo_train_meta.csv")
            df_train_subset.to_csv(subset_train_path, index=False)
            print(
                f"  Created train subset: {subset_train_path} ({len(df_train_subset)} rows)"
            )
        else:
            raise ValueError("Train metadata is empty.")
    else:
        raise FileNotFoundError(
            f"Train metadata not found at {Config.TRAIN_METADATA_PATH}"
        )

    # 2. Val Subset
    if os.path.exists(Config.VAL_METADATA_PATH):
        df_val = pd.read_csv(Config.VAL_METADATA_PATH)
        if not df_val.empty:
            first_drive = df_val["drive_id"].unique()[0]
            df_val_subset = df_val[df_val["drive_id"] == first_drive].head(200)
            subset_val_path = os.path.join(Config.WORKING_DIR, "demo_val_meta.csv")
            df_val_subset.to_csv(subset_val_path, index=False)
            print(
                f"  Created val subset: {subset_val_path} ({len(df_val_subset)} rows)"
            )
        else:
            # If val is empty (e.g. split issue), just use train subset as val for demo
            subset_val_path = subset_train_path
            print("  Val metadata empty, using train subset for demo validation.")
    else:
        raise FileNotFoundError("Val metadata not found.")

    # 3. Test Subset
    if os.path.exists(Config.TEST_METADATA_PATH):
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)
        if not df_test.empty:
            first_drive = df_test["drive_id"].unique()[0]
            df_test_subset = df_test[df_test["drive_id"] == first_drive].head(200)
            subset_test_path = os.path.join(Config.WORKING_DIR, "demo_test_meta.csv")
            df_test_subset.to_csv(subset_test_path, index=False)
            print(
                f"  Created test subset: {subset_test_path} ({len(df_test_subset)} rows)"
            )
        else:
            raise ValueError("Test metadata is empty.")
    else:
        raise FileNotFoundError("Test metadata not found.")

    return subset_train_path, subset_val_path, subset_test_path


def test_coordinate_transform():
    print("\n[Demo] Testing Coordinate Transform (ECEF to LLA)...")
    # Approximate ECEF coordinates for a point on the equator
    x, y, z = 6378137.0, 0.0, 0.0
    lat, lon, alt = ecef_to_lla(x, y, z)

    # Expected: Lat 0, Lon 0, Alt ~0
    print(f"  Input (XYZ): {x}, {y}, {z}")
    print(f"  Output (LLA): {lat:.4f}, {lon:.4f}, {alt:.4f}")

    assert abs(lat) < 1e-4, "Latitude should be approx 0"
    assert abs(lon) < 1e-4, "Longitude should be approx 0"
    assert abs(alt) < 1.0, "Altitude should be approx 0"
    print("  Verification Passed.")


def test_model_architecture():
    print("\n[Demo] Testing Model Architecture...")
    model = SensorFusionTCN(
        num_inputs=Config.NUM_FEATURES,
        num_channels=[32, 32],  # Smaller for demo
        kernel_size=3,
        dropout=0.1,
    ).to("cpu")

    # Create dummy input: (Batch, Features, Seq_Len)
    batch_size = 4
    seq_len = Config.WINDOW_SIZE
    dummy_input = torch.randn(batch_size, Config.NUM_FEATURES, seq_len)

    output = model(dummy_input)

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")

    assert output.shape == (batch_size, 2), "Output shape mismatch! Expected (Batch, 2)"
    print("  Verification Passed.")


def test_dataset_loading(train_meta_path):
    print("\n[Demo] Testing Dataset Loading...")
    # Initialize dataset with the subset metadata
    dataset = SmartphoneDataset(
        metadata_path=train_meta_path, window_size=Config.WINDOW_SIZE, mode="train"
    )

    print(f"  Dataset length (windows): {len(dataset)}")

    if len(dataset) > 0:
        features, target = dataset[0]
        print(f"  Sample 0 features shape: {features.shape}")
        print(f"  Sample 0 target shape: {target.shape}")

        # Validation
        assert features.shape == (
            Config.NUM_FEATURES,
            Config.WINDOW_SIZE,
        ), "Feature shape mismatch"
        assert target.shape == (2,), "Target shape mismatch"
        assert not np.isnan(features).any(), "Features contain NaNs"
        print("  Verification Passed.")
    else:
        print(
            "  Dataset is empty (window size might be too large for subset). Skipping item check."
        )


def run_training_pipeline(train_meta, val_meta):
    print("\n[Demo] Running Training Pipeline...")

    # Train for 1 epoch just to verify the loop works
    model = train_model(
        train_meta_path=train_meta, val_meta_path=val_meta, epochs=1, batch_size=16
    )

    # Check if weights were saved
    weights_path = os.path.join(Config.WORKING_DIR, "model_weights.pth")
    if os.path.exists(weights_path):
        print(f"  Weights saved successfully at: {weights_path}")
    else:
        raise FileNotFoundError("Model weights file was not created!")

    return weights_path


def run_inference_pipeline(test_meta, weights_path):
    print("\n[Demo] Running Inference Pipeline...")

    output_csv = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    generate_submission(
        test_meta_path=test_meta,
        model_weights_path=weights_path,
        output_path=output_csv,
        batch_size=16,
        device="cpu",  # Force CPU for demo stability
    )

    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        print(f"  Submission created: {output_csv}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")

        required_cols = [
            "tripId",
            "UnixTimeMillis",
            "LatitudeDegrees",
            "LongitudeDegrees",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column {col} in submission"
        print("  Verification Passed.")
    else:
        raise FileNotFoundError("Submission file was not created!")


if __name__ == "__main__":
    set_seed(42)

    # 1. Prepare Data Subsets
    train_meta, val_meta, test_meta = create_subset_metadata()

    # 2. Verify Helper Logic
    test_coordinate_transform()

    # 3. Verify Model Logic
    test_model_architecture()

    # 4. Verify Dataset Logic
    test_dataset_loading(train_meta)

    # 5. Run Training (Short)
    weights_path = run_training_pipeline(train_meta, val_meta)

    # 6. Run Inference
    run_inference_pipeline(test_meta, weights_path)

    print("\n[Demo] All demonstrations completed successfully.")
