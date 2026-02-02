import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import normalize_range, ash_composite, rle_encode
from library.dataset import ContrailDataset
from library.model import TemporalAshNet
from library.loss import WeightedLoss, DiceLoss, FocalLoss
from library.train import run_training
from library.inference import run_inference


def run_demo():
    print("=== Starting Contrail Detection Library Demo ===\n")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("[1] Setting up Configuration for Fast Execution...")

    # Initialize directories
    Config.setup()

    # Override Config for speed
    Config.EPOCHS = 1
    Config.DEBUG_SAMPLE_SIZE = 10  # Very small subset for demo
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    print("    Config configured: 1 Epoch, Debug Mode enabled.")

    # --------------------------------------------------------------------------
    # 2. Verify Utilities
    # --------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test normalize_range
    data = np.array([-10, 0, 10, 20], dtype=np.float32)
    norm = normalize_range(data, 0, 10)
    # Expected: [-10->0 (clip), 0->0, 10->1, 20->1 (clip)] -> [0, 0, 1, 1]
    assert np.allclose(norm, [0, 0, 1, 1]), "normalize_range failed"
    print("    normalize_range: OK")

    # Test ash_composite
    # Create dummy bands (H, W)
    b11 = np.full((10, 10), 280.0, dtype=np.float32)
    b14 = np.full((10, 10), 290.0, dtype=np.float32)
    b15 = np.full((10, 10), 285.0, dtype=np.float32)

    comp = ash_composite(b11, b14, b15)
    assert comp.shape == (10, 10, 3), f"ash_composite shape mismatch: {comp.shape}"
    assert (
        comp.min() >= 0 and comp.max() <= 1
    ), "ash_composite values out of range [0, 1]"
    print("    ash_composite: OK")

    # Test rle_encode
    # Create a 3x3 mask
    # [[0, 0, 0],
    #  [1, 1, 1],
    #  [0, 0, 0]]
    # Flattened Fortran (Column-major):
    # Col 0: 0,1,0; Col 1: 0,1,0; Col 2: 0,1,0
    # Sequence: 0, 1, 0, 0, 1, 0, 0, 1, 0
    # Indices (1-based): 2, 5, 8 are 1s.
    # Runs: Start 2 Len 1, Start 5 Len 1, Start 8 Len 1 -> "2 1 5 1 8 1"
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, :] = 1
    rle = rle_encode(mask)
    assert rle == "2 1 5 1 8 1", f"rle_encode failed. Got: {rle}"

    # Test empty mask
    empty_rle = rle_encode(np.zeros((3, 3)))
    assert empty_rle == "-", "rle_encode empty check failed"
    print("    rle_encode: OK")

    # --------------------------------------------------------------------------
    # 3. Verify Dataset
    # --------------------------------------------------------------------------
    print("\n[3] Verifying Dataset...")

    # Initialize dataset in train mode
    ds = ContrailDataset(Config.TRAIN_METADATA_PATH, stage="train")

    # Fetch one sample
    sample = ds[0]

    # Verify keys
    assert "image" in sample and "mask" in sample and "record_id" in sample

    # Verify shapes
    # Image: (9 channels, 256, 256)
    img_shape = sample["image"].shape
    assert img_shape == (9, 256, 256), f"Image shape incorrect: {img_shape}"

    # Mask: (1 channel, 256, 256)
    mask_shape = sample["mask"].shape
    assert mask_shape == (1, 256, 256), f"Mask shape incorrect: {mask_shape}"

    # Verify types
    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["mask"], torch.Tensor)

    print(f"    Dataset load successful. Image: {img_shape}, Mask: {mask_shape}")

    # --------------------------------------------------------------------------
    # 4. Verify Model & Loss
    # --------------------------------------------------------------------------
    print("\n[4] Verifying Model and Loss...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalAshNet().to(device)
    criterion = WeightedLoss().to(device)

    # Create dummy batch
    # Batch size 2, 9 channels, 256x256
    dummy_input = torch.randn(2, 9, 256, 256).to(device)
    dummy_target = torch.randint(0, 2, (2, 1, 256, 256)).float().to(device)

    # Forward pass
    logits = model(dummy_input)
    assert logits.shape == (
        2,
        1,
        256,
        256,
    ), f"Model output shape mismatch: {logits.shape}"

    # Loss calculation
    loss, focal, dice = criterion(logits, dummy_target)

    # Check if loss is scalar and valid
    assert loss.ndim == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"

    print(
        f"    Forward pass OK. Loss: {loss.item():.4f} (Focal: {focal.item():.4f}, Dice: {dice.item():.4f})"
    )

    # --------------------------------------------------------------------------
    # 5. Run Training (Debug Mode)
    # --------------------------------------------------------------------------
    print("\n[5] Running Training Loop (Debug Mode)...")

    # We use the run_training function from train.py
    # debug=True forces the use of Config.DEBUG_SAMPLE_SIZE
    run_training(debug=True, early_stopping_patience=1)

    # Verify checkpoint creation
    expected_checkpoint = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(expected_checkpoint):
        print(f"    Training complete. Checkpoint found at: {expected_checkpoint}")
    else:
        # It's possible validation didn't improve if random init was lucky,
        # but usually it saves at least once or we check if the file exists.
        # If not, we might have had a very poor run, but the code executed without error.
        print(
            "    Training complete (No checkpoint saved - metric might not have improved)."
        )

    # --------------------------------------------------------------------------
    # 6. Run Inference
    # --------------------------------------------------------------------------
    print("\n[6] Running Inference...")

    # To ensure inference runs quickly, we create a temporary small test metadata file
    full_test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    small_test_df = full_test_df.head(5)  # Only 5 records

    temp_test_meta_path = os.path.join(Config.METADATA_DIR, "temp_test_small.csv")
    small_test_df.to_csv(temp_test_meta_path, index=False)

    # Temporarily point Config to this small file
    original_test_path = Config.TEST_METADATA_PATH
    Config.TEST_METADATA_PATH = temp_test_meta_path

    try:
        # Run inference
        # If checkpoint exists, use it. If not (training failed to improve),
        # we can't strictly run inference with a trained model, but we can try
        # initializing a dummy file if needed. However, run_training usually saves
        # 'best_model.pth' if we set initial best_dice = -1, but the code has 0.0.
        # For the sake of the demo, if the file is missing, we save the current model state.
        if not os.path.exists(expected_checkpoint):
            print("    Saving current model state for inference demo...")
            torch.save(model.state_dict(), expected_checkpoint)

        submission_df = run_inference(
            checkpoint_path=expected_checkpoint, batch_size=2, load_cached_data=False
        )

        # Verify output
        assert (
            len(submission_df) == 5
        ), f"Expected 5 predictions, got {len(submission_df)}"
        assert "record_id" in submission_df.columns
        assert "encoded_pixels" in submission_df.columns

        # Verify submission file existence
        if os.path.exists(Config.SUBMISSION_PATH):
            print(f"    Submission file generated at: {Config.SUBMISSION_PATH}")
        else:
            raise FileNotFoundError("Submission file not found.")

    finally:
        # Restore Config and cleanup
        Config.TEST_METADATA_PATH = original_test_path
        if os.path.exists(temp_test_meta_path):
            os.remove(temp_test_meta_path)

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
