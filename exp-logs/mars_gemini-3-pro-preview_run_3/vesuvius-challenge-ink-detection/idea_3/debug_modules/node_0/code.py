import sys
import os
import torch
import numpy as np
import pandas as pd

# Append current directory to path to allow imports from library/
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import fbeta_score, rle_encoding
from library.dataset import InkDataset, get_training_transforms
from library.model import PSDN
from library.train import train_model
from library.inference import generate_submission


def run_demo():
    print("--- Starting Vesuvius Ink Detection Demo ---")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Fast Execution
    # -------------------------------------------------------------------------
    print("[1/6] Configuring environment...")

    # Use a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.ensure_directories()

    # Reduce training parameters for speed
    Config.NUM_EPOCHS = 1
    Config.STEPS_PER_EPOCH = 5  # Run only 5 batches per epoch for demo
    Config.VAL_SAMPLE_SIZE = 20  # Validate on 20 samples
    Config.BATCH_SIZE = 4  # Small batch size

    # Increase inference stride to process fewer patches (Sparse prediction)
    # Normal stride is 64; 1024 ensures we cover the area very sparsely but quickly
    Config.INFERENCE_STRIDE = 1024

    # Ensure reproducibility
    Config.set_seed(42)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("[2/6] Verifying utility functions...")

    # Test RLE Encoding
    # vector: 0 1 1 1 0 0 1 0 -> Indices (1-based): 2,3,4 and 7 -> Runs: (2, 3), (7, 1)
    dummy_mask = np.array([[0, 1, 1, 1, 0, 0, 1, 0]], dtype=np.uint8)
    rle_result = rle_encoding(dummy_mask)
    expected_rle = "2 3 7 1"
    assert (
        rle_result == expected_rle
    ), f"RLE Error: Expected '{expected_rle}', got '{rle_result}'"
    print("    RLE Encoding: OK")

    # Test F-beta Score (Beta=0.5)
    # True: 1 1 0 0
    # Pred: 1 0 1 0
    # TP=1, FP=1, FN=1
    # F0.5 = (1.25 * TP) / (1.25*TP + 0.25*FN + FP)
    #      = 1.25 / (1.25 + 0.25 + 1) = 1.25 / 2.5 = 0.5
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    score = fbeta_score(y_true, y_pred, beta=0.5)
    assert np.isclose(score, 0.5), f"F-beta Error: Expected 0.5, got {score}"
    print("    F-beta Score: OK")

    # -------------------------------------------------------------------------
    # 3. Verify Dataset Loading
    # -------------------------------------------------------------------------
    print("[3/6] Verifying Dataset...")

    # Instantiate training dataset
    # We allow caching to test the full pipeline, but on a small scale
    train_ds = InkDataset(
        split="train", transform=get_training_transforms(), cache_data=True
    )

    # Check virtual length
    expected_len = Config.STEPS_PER_EPOCH * Config.BATCH_SIZE
    assert (
        len(train_ds) == expected_len
    ), f"Dataset Length Error: Expected {expected_len}, got {len(train_ds)}"

    # Check item structure
    vol, label = train_ds[0]

    # Expected shapes: Vol (65, 128, 128), Label (1, 128, 128)
    # Note: Dataset returns torch tensors
    assert vol.shape == (
        Config.Z_DIM,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), f"Volume Shape Error: Got {vol.shape}"
    assert label.shape == (
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), f"Label Shape Error: Got {label.shape}"
    assert isinstance(vol, torch.Tensor), "Volume should be a torch.Tensor"

    print(f"    Dataset loaded successfully. Sample shape: {vol.shape}")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("[4/6] Verifying Model...")

    model = PSDN().to(Config.DEVICE)

    # Create dummy batch: (Batch=2, Z=65, H=128, W=128)
    dummy_input = torch.randn(2, Config.Z_DIM, Config.PATCH_SIZE, Config.PATCH_SIZE).to(
        Config.DEVICE
    )

    with torch.no_grad():
        output = model(dummy_input)

    # Expected output: (2, 1, 128, 128)
    assert output.shape == (
        2,
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), f"Model Output Shape Error: Got {output.shape}"

    print("    Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 5. Execute Training Loop
    # -------------------------------------------------------------------------
    print("[5/6] Executing Training Loop...")

    # Run training (uses the overridden Config parameters)
    best_score, best_thresh = train_model(load_cached_data=True)

    print(f"    Training complete. Best Val Score: {best_score:.4f}")

    # Verify model checkpoint exists
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print("    Checkpoint verified.")

    # -------------------------------------------------------------------------
    # 6. Execute Inference Pipeline
    # -------------------------------------------------------------------------
    print("[6/6] Executing Inference Pipeline...")

    # Run inference
    generate_submission(threshold=best_thresh, load_cached_data=True)

    # Verify submission file
    submission_path = Config.SUBMISSION_FILE
    assert os.path.exists(submission_path), "Submission file not found"

    df_sub = pd.read_csv(submission_path)
    assert list(df_sub.columns) == ["Id", "Predicted"], "Submission columns mismatch"
    assert len(df_sub) > 0, "Submission file is empty"

    print("    Submission file generated and verified.")
    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
