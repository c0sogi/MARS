import os
import shutil
import pandas as pd
import torch
import numpy as np
import warnings

# Import from provided library
from library.utils import set_seed, get_device
from library.loss import DeepSupervisionLoss
from library.model import GLT_CRCN
from library.data_loader import DataLoaderConfig, get_data, collate_fn, GestureDataset
from library.train_eval import (
    train_model,
    generate_submission,
    levenshtein_distance,
    post_process_sequence,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_environment(base_dir="./working/demo"):
    """Creates a temporary environment with subset metadata for fast execution."""
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)

    metadata_dir = os.path.join(base_dir, "metadata")
    os.makedirs(metadata_dir)

    # Create subset metadata (top 10 rows)
    for split in ["train", "val", "test"]:
        src_csv = f"./metadata/{split}.csv"
        dst_csv = os.path.join(metadata_dir, f"{split}.csv")
        if os.path.exists(src_csv):
            df = pd.read_csv(src_csv)
            # Take a small subset
            subset = df.head(10)
            subset.to_csv(dst_csv, index=False)
            print(f"Created subset metadata for {split}: {len(subset)} samples")
        else:
            print(f"Warning: {src_csv} not found.")

    # Patch configuration to use demo directories
    DataLoaderConfig.METADATA_DIR = metadata_dir
    DataLoaderConfig.WORKING_DIR = base_dir
    DataLoaderConfig.CACHE_DIR = os.path.join(base_dir, "cache")

    return base_dir


def test_metrics():
    """Validates metric calculation functions."""
    print("\n--- Testing Metrics ---")

    # Test Levenshtein
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist = levenshtein_distance(seq1, seq2)
    assert dist == 0, f"Expected distance 0, got {dist}"

    seq1 = [1, 2, 3]
    seq2 = [1, 2, 4]
    dist = levenshtein_distance(seq1, seq2)
    assert dist == 1, f"Expected distance 1, got {dist}"
    print("Levenshtein distance logic verified.")

    # Test Post-processing
    # Sequence: [1, 1, 1, 0, 0, 2, 2, 2] -> Median(3) -> [1, 1, 0, 0, 2, 2] approx -> Collapse -> [1, 2]
    # Note: The provided post_process uses median filter then collapse
    raw_preds = np.array([1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 2])
    # Median window 3
    processed = post_process_sequence(raw_preds, median_window=3)
    # Expected: 1s dominate start, 0s middle, 2s end. 0 is background (removed). Result: [1, 2]
    assert processed == [1, 2], f"Expected [1, 2], got {processed}"
    print("Post-processing logic verified.")


def test_model_logic():
    """Validates Model and Loss forward passes."""
    print("\n--- Testing Model & Loss ---")
    device = get_device()

    # Configuration
    batch_size = 2
    time_steps = 50
    input_dim = 85  # 36 pos + 36 vel + 13 audio
    num_classes = 21

    # Instantiate Model
    model = GLT_CRCN().to(device)
    model.eval()

    # Create Dummy Input
    # (B, C, T)
    x = torch.randn(batch_size, input_dim, time_steps).to(device)
    # Mask (B, T) - All valid
    mask = torch.ones(batch_size, time_steps).to(device)

    # Forward Pass
    outputs = model(x, mask)

    # Check outputs
    assert isinstance(outputs, list), "Model output should be a list"
    assert len(outputs) == 3, f"Expected 3 stages, got {len(outputs)}"

    # Stage 1 & 2 output shape: (B, NumClasses+1, T)
    assert outputs[0].shape == (
        batch_size,
        num_classes + 1,
        time_steps,
    ), f"Stage 1 shape mismatch: {outputs[0].shape}"
    assert outputs[1].shape == (
        batch_size,
        num_classes + 1,
        time_steps,
    ), f"Stage 2 shape mismatch: {outputs[1].shape}"
    # Stage 3 output shape: (B, NumClasses, T)
    assert outputs[2].shape == (
        batch_size,
        num_classes,
        time_steps,
    ), f"Stage 3 shape mismatch: {outputs[2].shape}"

    print("Model forward pass verified.")

    # Test Loss
    criterion = DeepSupervisionLoss(num_classes=num_classes)
    targets = torch.randint(0, num_classes, (batch_size, time_steps)).to(device)

    loss = criterion(outputs, targets, mask)

    assert torch.is_tensor(loss), "Loss should be a tensor"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert loss.item() > 0, "Loss should be positive"

    print(f"Loss calculation verified. Value: {loss.item():.4f}")


def run_integration_test(demo_dir):
    """Runs the training and inference pipeline on subset data."""
    print("\n--- Running Integration Test (Train/Inference) ---")

    # 1. Train
    print("Starting training loop (1 epoch, subset data)...")
    best_model_path = train_model(
        epochs=1,
        batch_size=4,
        lr=1e-3,
        working_dir=demo_dir,
        load_cached_data=False,  # Force processing of our subset
    )

    assert os.path.exists(best_model_path), "Best model file was not created."
    print("Training completed successfully.")

    # 2. Inference
    print("Starting inference...")
    submission_dir = os.path.join(demo_dir, "submission")
    generate_submission(
        model_path=best_model_path,
        submission_dir=submission_dir,
        load_cached_data=False,
    )

    submission_file = os.path.join(submission_dir, "submission.csv")
    assert os.path.exists(submission_file), "Submission file was not created."

    # Check content
    df_sub = pd.read_csv(submission_file, header=None)
    print(f"Submission generated with {len(df_sub)} rows.")
    # Expect 10 rows based on subset test.csv
    assert len(df_sub) == 10, f"Expected 10 predictions, got {len(df_sub)}"

    print("Inference pipeline verified.")


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)

    # Setup
    demo_dir = setup_demo_environment()

    try:
        # Unit Tests
        test_metrics()
        test_model_logic()

        # Integration Test
        run_integration_test(demo_dir)

        print("\nAll checks passed successfully!")

    except Exception as e:
        print(f"\nFAILED: {e}")
        raise e
    finally:
        # Cleanup
        if os.path.exists(demo_dir):
            shutil.rmtree(demo_dir)
            print("Cleanup complete.")
