import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import rle_encode, rle_decode, calc_iou, calc_map
from library.dataset import SaltDataset
from library.model import HighCapacityUNet
from library.losses import CompoundLoss
from library.train import train_model
from library.inference import predict_ensemble

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def run_demo():
    print("=== Starting Salt Segmentation Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Demo
    # -------------------------------------------------------------------------
    # We modify the Config class directly to run a fast, minimal version of the task.
    # This ensures we don't write to protected areas and finish within time limits.

    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)

    # Update Paths
    Config.WORKING_DIR = DEMO_DIR
    Config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    Config.PREDICTION_DIR = os.path.join(DEMO_DIR, "predictions")
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "demo_submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Update Training Parameters for Speed
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.NUM_CYCLES = 1
    Config.EPOCHS_PER_CYCLE = 1
    Config.TOTAL_EPOCHS = 1

    # Enable Debug Mode
    # This limits the dataset size and adjusts internal loop counters
    Config.set_debug_mode(debug=True, epochs_per_cycle=1, data_limit=20)

    # Setup directories
    Config.setup()
    set_seed(Config.SEED)

    print("\n[1/5] Configuration configured for fast execution.")
    print(f"      Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions (RLE & Metrics)
    # -------------------------------------------------------------------------
    print("\n[2/5] Verifying Utility Functions...")

    # Test RLE Encoding/Decoding
    # Create a dummy 101x101 mask with a 10x10 square of 1s
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1

    encoded = rle_encode(dummy_mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    assert isinstance(encoded, str), "RLE encode should return a string"
    assert np.array_equal(dummy_mask, decoded), "RLE decode did not match original mask"
    print("      RLE Encoding/Decoding: PASSED")

    # Test IoU and mAP
    # Case 1: Perfect Match
    iou_perfect = calc_iou(dummy_mask, dummy_mask)
    assert iou_perfect == 1.0, f"Expected IoU 1.0 for perfect match, got {iou_perfect}"

    # Case 2: No Overlap
    dummy_mask_2 = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask_2[50:60, 50:60] = 1
    iou_zero = calc_iou(dummy_mask, dummy_mask_2)
    assert iou_zero == 0.0, f"Expected IoU 0.0 for disjoint masks, got {iou_zero}"

    # Case 3: mAP Calculation
    # List of predictions and targets
    preds = [dummy_mask, dummy_mask]
    targets = [dummy_mask, dummy_mask_2]  # One perfect match, one miss

    # Thresholds are 0.5 to 0.95 (10 steps).
    # Pair 1: IoU=1.0. Passes all 10 thresholds. Precision=1.0.
    # Pair 2: IoU=0.0. Fails all 10 thresholds. Precision=0.0.
    # Mean Precision per threshold = (1.0 + 0.0) / 2 = 0.5
    # Average over all thresholds = 0.5

    map_score = calc_map(preds, targets)
    assert np.isclose(map_score, 0.5), f"Expected mAP 0.5, got {map_score}"
    print("      Metric Calculation (IoU/mAP): PASSED")

    # -------------------------------------------------------------------------
    # 3. Verify Model Components (Forward/Backward Pass)
    # -------------------------------------------------------------------------
    print("\n[3/5] Verifying Model Components...")

    device = torch.device(Config.DEVICE)

    # Load a tiny dataset
    ds = SaltDataset(mode="train", load_cached_data=False, limit=4)
    dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)

    # Get a batch
    images, masks, depths, ids = next(iter(dl))
    images = images.to(device)
    masks = masks.to(device)
    depths = depths.to(device)

    print(f"      Input Shapes -> Images: {images.shape}, Depths: {depths.shape}")

    # Initialize Model
    model = HighCapacityUNet().to(device)

    # Forward Pass
    logits = model(images, depths)
    assert (
        logits.shape == masks.shape
    ), f"Output shape mismatch. Expected {masks.shape}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"
    print("      Forward Pass: PASSED")

    # Loss Calculation
    criterion = CompoundLoss().to(device)
    loss, metrics = criterion(logits, masks)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    print(f"      Loss Calculation: PASSED (Loss: {loss.item():.4f})")

    # Backward Pass check
    loss.backward()
    print("      Backward Pass: PASSED")

    # Clean up memory
    del model, images, masks, depths, logits, loss
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 4. Run Training Pipeline
    # -------------------------------------------------------------------------
    print("\n[4/5] Executing Training Pipeline (Simulated)...")
    # This calls the provided library function train_model()
    # It will use the Config settings we modified (1 epoch, minimal data)
    # and save 'best_model.pth' to our demo checkpoint directory.

    try:
        train_model()
        print("      Training completed successfully.")
    except Exception as e:
        print(f"      Training failed with error: {e}")
        raise e

    # Verify checkpoint creation
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Training did not produce a checkpoint at {best_model_path}"
        )
    else:
        print(f"      Checkpoint verified at: {best_model_path}")

    # -------------------------------------------------------------------------
    # 5. Run Inference Pipeline
    # -------------------------------------------------------------------------
    print("\n[5/5] Executing Inference Pipeline...")

    # This calls the provided library function predict_ensemble()
    # It will load the model we just trained and generate predictions for the test set.
    # We limit to 10 samples for speed.

    try:
        predict_ensemble(debug=True, limit=10)
        print("      Inference completed successfully.")
    except Exception as e:
        print(f"      Inference failed with error: {e}")
        raise e

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_FILE}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"      Submission file loaded. Shape: {df_sub.shape}")

    # Check columns
    expected_cols = ["id", "rle_mask"]
    if list(df_sub.columns) != expected_cols:
        raise ValueError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"
        )

    # Check content format (simple check on first row)
    if len(df_sub) > 0:
        first_rle = df_sub.iloc[0]["rle_mask"]
        if pd.notna(first_rle) and first_rle != "":
            # Try decoding to ensure it's valid RLE
            try:
                rle_decode(first_rle, (101, 101))
            except Exception as e:
                raise ValueError(f"Generated RLE is invalid: {e}")

    print("      Submission Format: PASSED")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
