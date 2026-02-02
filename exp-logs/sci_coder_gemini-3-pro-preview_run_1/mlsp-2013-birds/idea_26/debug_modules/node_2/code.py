import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders, Mixup
from library.model import create_model
from library.engine import SWAEngine
from library.pipeline import run_pipeline


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Configuration Setup
    # We use debug=True to use a tiny subset of data (10 samples) and few epochs (2).
    # This ensures the script runs quickly while exercising all code paths.
    output_dir = "./working/demo_execution"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    config = Config(debug=True, output_dir=output_dir)

    # Set seed for reproducibility
    set_seed(config.SEED)

    print(
        f"Configuration: Debug={config.DEBUG}, Device={config.DEVICE}, Output Dir={config.OUTPUT_DIR}"
    )

    # 2. Data Loading & Verification
    print("\n--- Verifying Data Loading ---")
    dataloaders = get_dataloaders(config)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # Check if loader is empty (due to drop_last=True and small debug dataset)
    if len(train_loader) == 0:
        print("Adjusting batch size for small debug dataset...")
        config.BATCH_SIZE = 2
        # Re-initialize dataloaders with new batch size
        dataloaders = get_dataloaders(config)
        train_loader = dataloaders["train"]

    # Fetch one batch
    images, targets = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    assert images.dim() == 4, "Images must be 4D tensors (B, C, H, W)"
    assert images.shape[1] == 3, "Images must have 3 channels"
    assert images.shape[2] == config.IMG_HEIGHT, f"Height must be {config.IMG_HEIGHT}"
    assert images.shape[3] == config.IMG_WIDTH, f"Width must be {config.IMG_WIDTH}"
    assert targets.shape[1] == 19, "Targets must have 19 classes"
    print("Data shapes verified successfully.")

    # 3. Model Instantiation & Forward Pass
    print("\n--- Verifying Model Architecture ---")
    model = create_model(config)
    model = model.to(config.DEVICE)
    model.eval()

    with torch.no_grad():
        # Move images to device
        images_dev = images.to(config.DEVICE)
        outputs = model(images_dev)

    print(f"Model Output Shape: {outputs.shape}")
    assert outputs.shape == (images.shape[0], 19), "Output shape mismatch"
    assert not torch.isnan(outputs).any(), "Model produced NaNs"
    print("Model forward pass verified successfully.")

    # 4. Mixup Verification
    print("\n--- Verifying Mixup ---")
    mixup_fn = Mixup(alpha=1.0)
    # Create dummy data on device
    x = torch.randn(4, 3, 256, 640).to(config.DEVICE)
    y = torch.randint(0, 2, (4, 19)).float().to(config.DEVICE)

    mixed_x, mixed_y = mixup_fn(x, y)

    assert mixed_x.shape == x.shape, "Mixup altered input shape"
    assert mixed_y.shape == y.shape, "Mixup altered target shape"
    assert not torch.equal(
        x, mixed_x
    ), "Mixup did not modify inputs (probability is extremely low for exact match)"
    print("Mixup verified successfully.")

    # 5. SWA Engine Verification
    print("\n--- Verifying SWA Engine ---")
    # Mock optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    swa_engine = SWAEngine(model, optimizer, config, swa_start_epoch=1)

    # Check active status
    assert not swa_engine.is_swa_active(epoch=0), "SWA should not be active at epoch 0"
    assert swa_engine.is_swa_active(epoch=1), "SWA should be active at epoch 1"
    print("SWA Engine logic verified successfully.")

    # 6. Full Pipeline Execution
    print("\n--- Executing Full Pipeline (Teachers -> Pseudo Labels -> Student) ---")
    # This runs the logic in library/pipeline.py
    # Since debug=True, it trains for 2 epochs on ~10 samples.
    # It trains 3 teachers, generates pseudo labels, and trains 1 student.

    try:
        run_pipeline(config)
        print("Pipeline execution completed without errors.")
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        raise e

    # 7. Output Validation
    print("\n--- Validating Outputs ---")

    # Check Submission File
    submission_path = config.SUBMISSION_PATH
    if os.path.exists(submission_path):
        print(f"Found submission file at {submission_path}")
        df_sub = pd.read_csv(submission_path)
        print(f"Submission shape: {df_sub.shape}")

        # In debug mode, we used head(10) for test set.
        # There are 19 classes per recording.
        # So we expect roughly 10 * 19 = 190 rows.
        # However, run_pipeline calls get_dataloaders(config) inside.
        # If config.DEBUG is True, get_dataloaders returns head(10).
        expected_rows = 10 * 19
        assert (
            len(df_sub) == expected_rows
        ), f"Expected {expected_rows} rows in submission, found {len(df_sub)}"

        # Check columns
        assert (
            "Id" in df_sub.columns and "Probability" in df_sub.columns
        ), "Missing required columns in submission"

        # Check values
        assert (
            df_sub["Probability"].min() >= 0 and df_sub["Probability"].max() <= 1
        ), "Probabilities out of range [0, 1]"
    else:
        raise FileNotFoundError(f"Submission file not created at {submission_path}")

    # Check Pseudo Labels
    pseudo_path = os.path.join(config.OUTPUT_DIR, "pseudo_labels.parquet")
    if os.path.exists(pseudo_path):
        print(f"Found pseudo-labels at {pseudo_path}")
        df_pseudo = pd.read_parquet(pseudo_path)
        assert (
            len(df_pseudo) == 10
        ), "Pseudo labels should match debug test set size (10)"
    else:
        raise FileNotFoundError(f"Pseudo labels not found at {pseudo_path}")

    # Check Checkpoints
    # We expect teacher_0_swa.pth, teacher_1_swa.pth, teacher_2_swa.pth, student_swa.pth
    expected_models = [
        "teacher_0_swa.pth",
        "teacher_1_swa.pth",
        "teacher_2_swa.pth",
        "student_swa.pth",
    ]
    for model_name in expected_models:
        path = os.path.join(config.OUTPUT_DIR, model_name)
        if os.path.exists(path):
            print(f"Found checkpoint: {model_name}")
        else:
            raise FileNotFoundError(f"Missing checkpoint: {model_name}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
