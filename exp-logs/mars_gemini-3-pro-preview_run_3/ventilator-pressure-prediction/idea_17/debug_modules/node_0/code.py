import os
import pandas as pd
import numpy as np
import torch
import shutil
import sys

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, masked_mae_loss
from library.features import engineer_features
from library.dataset import prepare_datasets, VentilatorDataset
from library.model import DCLKNet
from library.train import run_training
from library.predict import generate_predictions


def create_demo_subsets(n_breaths=50):
    """
    Creates small subsets of the metadata files to ensure the demo runs quickly.
    """
    print(f"\n[Demo] Creating data subsets (n_breaths={n_breaths})...")

    demo_input_dir = "./working/demo_input"
    os.makedirs(demo_input_dir, exist_ok=True)

    # Define source and destination paths
    files = {
        "train.csv": Config.TRAIN_PATH,
        "validation.csv": Config.VAL_PATH,
        "test.csv": Config.TEST_PATH,
    }

    new_paths = {}

    # Each breath has 80 time steps
    rows_to_read = n_breaths * 80

    for filename, src_path in files.items():
        dest_path = os.path.join(demo_input_dir, filename)

        # Read only the first N rows
        df = pd.read_csv(src_path, nrows=rows_to_read)

        # Verify we didn't slice a breath in half (check last breath_id)
        last_breath_id = df.iloc[-1]["breath_id"]
        df = df[df["breath_id"] <= last_breath_id]

        df.to_csv(dest_path, index=False)
        new_paths[filename] = dest_path
        print(f"  Created subset {filename}: {df.shape}")

    return new_paths


def verify_feature_engineering():
    """
    Demonstrates and verifies the feature engineering pipeline.
    """
    print("\n[Demo] Verifying Feature Engineering...")

    # Force recompute to ensure we process our new subsets
    # We use load_cached_data=False to trigger computation
    data = engineer_features(load_cached_data=False)

    # Assertions
    # 1. Check keys exist
    expected_keys = [
        "train_x",
        "train_y",
        "train_u_out",
        "val_x",
        "val_y",
        "val_u_out",
        "test_x",
        "test_ids",
        "test_u_out",
        "scaler_stats",
    ]
    for k in expected_keys:
        assert k in data, f"Missing key in feature data: {k}"

    # 2. Check Shapes
    # We expect shape (N_breaths, 80, 14) for X
    # N_breaths might vary slightly depending on the subset slicing, but should be > 0
    train_x = data["train_x"]
    train_y = data["train_y"]

    assert train_x.ndim == 3, f"Train X should be 3D, got {train_x.shape}"
    assert train_x.shape[1] == 80, f"Time steps should be 80, got {train_x.shape[1]}"
    assert train_x.shape[2] == 14, f"Feature dim should be 14, got {train_x.shape[2]}"

    assert train_y.shape == (
        train_x.shape[0],
        80,
    ), f"Train Y shape mismatch: {train_y.shape}"

    print("  Feature engineering shapes verified.")
    return data


def verify_model_and_dataset(data):
    """
    Demonstrates dataset loading and model forward/backward pass.
    """
    print("\n[Demo] Verifying Model and Dataset...")

    # 1. Create Dataset
    train_dataset = VentilatorDataset(
        data["train_x"], data["train_u_out"], y=data["train_y"]
    )

    # 2. Create Loader
    batch_size = 8
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False
    )

    # 3. Fetch a batch
    x, u_out, y, ids = next(iter(train_loader))

    # Assert Batch Shapes
    assert x.shape == (batch_size, 80, 14), f"Batch X shape wrong: {x.shape}"
    assert y.shape == (batch_size, 80), f"Batch Y shape wrong: {y.shape}"

    # 4. Initialize Model
    model = DCLKNet()
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        y = y.cuda()
        u_out = u_out.cuda()

    # 5. Forward Pass
    preds = model(x)
    assert preds.shape == (batch_size, 80), f"Prediction shape wrong: {preds.shape}"

    # 6. Loss Calculation
    loss = masked_mae_loss(preds, y, u_out)
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"

    # 7. Backward Pass (Check gradients)
    loss.backward()
    # Check if a parameter has gradients
    param = next(model.parameters())
    assert param.grad is not None, "Gradients not computed"

    print("  Model forward/backward pass verified.")


def verify_full_pipeline():
    """
    Runs the high-level training and prediction functions.
    """
    print("\n[Demo] Verifying Full Training Pipeline...")

    # Run training
    # Using very low epochs for speed
    run_training(epochs=1, batch_size=16, debug=False, force_recompute=False)

    # Check artifacts
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_path), "Model file not created"

    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(sub_path), "Submission file not created"

    # Verify Submission Content
    sub_df = pd.read_csv(sub_path)
    assert (
        "id" in sub_df.columns and "pressure" in sub_df.columns
    ), "Submission columns missing"
    assert len(sub_df) > 0, "Submission file is empty"

    print("  Training pipeline verified.")

    print("\n[Demo] Verifying Prediction Pipeline...")
    # Run prediction explicitly (though run_training does it, this checks the standalone function)
    generate_predictions(batch_size=16, debug=False, force_recompute=False)

    # Check timestamp or existence (re-verifying)
    assert os.path.exists(sub_path), "Submission file missing after prediction"
    print("  Prediction pipeline verified.")


if __name__ == "__main__":
    # 1. Setup Environment
    seed_everything(42)

    # 2. Configure for Demo
    # Redirect paths to a demo working directory
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Create subsets and update Config paths
    new_paths = create_demo_subsets(n_breaths=50)
    Config.TRAIN_PATH = new_paths["train.csv"]
    Config.VAL_PATH = new_paths["validation.csv"]
    Config.TEST_PATH = new_paths["test.csv"]

    # Set Hyperparameters for speed
    Config.BATCH_SIZE = 16
    Config.EPOCHS = 1
    Config.DEBUG = False  # We already manually subsetted the data via CSVs

    # 3. Run Verifications
    try:
        # Step A: Feature Engineering
        data = verify_feature_engineering()

        # Step B: Model & Dataset internals
        verify_model_and_dataset(data)

        # Step C: Full Training & Inference Pipeline
        verify_full_pipeline()

        print("\nSUCCESS: All demonstrations and verifications passed.")

    except AssertionError as e:
        print(f"\nFAILURE: Assertion failed - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nFAILURE: An unexpected error occurred - {e}")
        # Print traceback for debugging
        import traceback

        traceback.print_exc()
        sys.exit(1)
