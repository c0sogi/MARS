import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, dice_coef, hausdorff_3d
from library.data import prepare_data, UWDataset, get_transforms
from library.model import UNetEfficientNet
from library.loss import BCETverskyLoss
from library.train import run_training
from library.inference import predict


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup & Configuration Overrides for Speed
    # We override Config parameters to run a very fast "debug" cycle.
    print("[1] Configuring environment...")
    set_seed(Config.SEED)

    # Patch Config to ensure rapid execution
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG_SAMPLE_SIZE = 16  # Small subset for demonstration
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure working directory exists
    Config.setup()

    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"    Device: {Config.DEVICE}")

    # 2. Data Pipeline Verification
    print("\n[2] Verifying Data Pipeline...")

    # Load raw metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)

    # Run prepare_data (generates 2.5D context columns)
    # We force load_cached_data=False to ensure logic runs at least once
    df_processed = prepare_data(df_train, load_cached_data=False, split="train")

    # Verify new columns exist
    assert (
        "prev_path" in df_processed.columns
    ), "prepare_data failed to create 'prev_path'"
    assert (
        "next_path" in df_processed.columns
    ), "prepare_data failed to create 'next_path'"

    # Subset for dataset testing
    df_sample = df_processed.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)

    # Instantiate Dataset
    dataset = UWDataset(
        df_sample, transforms=get_transforms(data="train"), mode="train"
    )

    # Fetch one sample
    img, mask = dataset[0]

    # Verify Shapes
    # Image: (Channels, Height, Width) -> (3, 320, 320)
    # Mask: (Classes, Height, Width) -> (3, 320, 320)
    print(f"    Sample Image Shape: {img.shape}")
    print(f"    Sample Mask Shape: {mask.shape}")

    assert img.shape == (
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    ), f"Incorrect image shape: {img.shape}"
    assert mask.shape == (
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    ), f"Incorrect mask shape: {mask.shape}"
    assert img.dtype == torch.float32, "Image tensor should be float32"
    assert mask.dtype == torch.float32, "Mask tensor should be float32"

    # 3. Model & Loss Verification
    print("\n[3] Verifying Model and Loss...")

    model = UNetEfficientNet(
        backbone_name=Config.BACKBONE, pretrained=False, classes=Config.NUM_CLASSES
    )
    model.to(Config.DEVICE)
    model.eval()

    # Create a dummy batch
    dummy_input = img.unsqueeze(0).to(Config.DEVICE)  # (1, 3, 320, 320)
    dummy_target = mask.unsqueeze(0).to(Config.DEVICE)  # (1, 3, 320, 320)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Model Output Shape: {output.shape}")
    assert output.shape == (
        1,
        Config.NUM_CLASSES,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    ), "Model output shape mismatch"

    # Loss calculation
    criterion = BCETverskyLoss()
    loss = criterion(output, dummy_target)

    print(f"    Calculated Loss: {loss.item():.6f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    # 4. Metric Verification
    print("\n[4] Verifying Metrics...")

    # Synthetic data for metrics
    # Case 1: Perfect match
    y_true = np.ones((10, 10, 10), dtype=np.uint8)
    y_pred = np.ones((10, 10, 10), dtype=np.uint8)

    d_score = dice_coef(y_true, y_pred)
    h_score = hausdorff_3d(y_true, y_pred)

    print(f"    Perfect Match -> Dice: {d_score:.4f}, Hausdorff: {h_score:.4f}")
    assert np.isclose(d_score, 1.0), "Dice should be 1.0 for perfect match"
    assert np.isclose(h_score, 0.0), "Hausdorff should be 0.0 for perfect match"

    # Case 2: Empty vs Empty
    y_true_empty = np.zeros((10, 10, 10), dtype=np.uint8)
    y_pred_empty = np.zeros((10, 10, 10), dtype=np.uint8)

    d_score_e = dice_coef(y_true_empty, y_pred_empty)
    h_score_e = hausdorff_3d(y_true_empty, y_pred_empty)

    print(f"    Empty Match   -> Dice: {d_score_e:.4f}, Hausdorff: {h_score_e:.4f}")
    # Dice is 0 if both empty in definition provided in metric description,
    # but the implementation uses smoothing: (0+smooth)/(0+smooth) = 1.0
    # Let's check the implementation behavior:
    # (2*0 + 1e-6) / (0 + 0 + 1e-6) = 1.0. Correct for implementation.
    assert np.isclose(
        d_score_e, 1.0
    ), "Dice should be 1.0 for empty-empty match due to smoothing"
    assert np.isclose(h_score_e, 0.0), "Hausdorff should be 0.0 for empty-empty match"

    # 5. Training Integration Test
    print("\n[5] Running Training Loop (Debug Mode)...")
    # This runs the full training logic on a tiny subset of data
    run_training(debug=True, load_cached_data=True)

    # Verify checkpoint creation
    if os.path.exists(Config.CHECKPOINT_PATH):
        print(f"    SUCCESS: Checkpoint created at {Config.CHECKPOINT_PATH}")
    else:
        # Note: If validation dice doesn't improve (it starts at -inf, so it should), it might not save.
        # However, with 1 epoch, it should save once.
        raise FileNotFoundError("Checkpoint file was not created during training.")

    # 6. Inference Integration Test
    print("\n[6] Running Inference Loop (Debug Mode)...")
    # This runs prediction on the test set (or subset)
    predict(load_cached_data=True, debug=True)

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"    SUCCESS: Submission file created at {Config.SUBMISSION_PATH}")

        # Validate content format
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"    Submission Rows: {len(df_sub)}")
        print(f"    Columns: {list(df_sub.columns)}")

        expected_cols = ["id", "class", "predicted"]
        assert all(
            col in df_sub.columns for col in expected_cols
        ), "Missing columns in submission file"

        # Check if we have predictions (might be empty strings if threshold not met, which is fine)
        # Just ensure the file isn't empty
        assert len(df_sub) > 0, "Submission file is empty"
    else:
        raise FileNotFoundError("Submission file was not created during inference.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
