import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import cv2
from functools import partialmethod
from tqdm.auto import tqdm

# Import library modules
from library.config import CFG
from library.utils import rle_encode, rle_decode, compute_dice, compute_hausdorff_3d
from library.dataset import UWMGIDataset, process_25d_dataframe, get_transforms
from library.model import build_model
from library.losses import CompositeLoss
from library.train import train
from library.inference import predict_and_submit

# Suppress tqdm progress bars for cleaner output
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def run_demo():
    print("=== Starting 2.5D MRI Segmentation Pipeline Demo ===\n")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed
    # ---------------------------------------------------------
    print("[1] Configuring experiment for fast demonstration...")

    # Use a specific working directory for this demo
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)

    # Override CFG settings
    CFG.working_dir = demo_dir
    CFG.checkpoint_dir = os.path.join(demo_dir, "checkpoints")
    CFG.predictions_dir = os.path.join(demo_dir, "predictions")
    CFG.log_dir = os.path.join(demo_dir, "logs")
    CFG.submission_dir = os.path.join(demo_dir, "submission")
    CFG.submission_file = os.path.join(CFG.submission_dir, "submission.csv")

    # Training params
    CFG.debug = True
    CFG.debug_size = 50  # Use only 50 samples
    CFG.epochs = 1
    CFG.train_batch_size = 4
    CFG.valid_batch_size = 4
    CFG.num_workers = 0  # Avoid multiprocessing overhead for small demo

    # Initialize environment
    CFG.setup(verbose=True)
    print("Configuration updated successfully.\n")

    # ---------------------------------------------------------
    # 2. Verify Utility Functions
    # ---------------------------------------------------------
    print("[2] Verifying Utility Functions (RLE, Metrics)...")

    # Test RLE Encoding/Decoding
    dummy_mask = np.zeros((100, 100), dtype=np.uint8)
    dummy_mask[20:30, 20:30] = 1  # Create a 10x10 square

    rle_str = rle_encode(dummy_mask)
    decoded_mask = rle_decode(rle_str, (100, 100))

    if not np.array_equal(dummy_mask, decoded_mask):
        raise AssertionError("RLE Encode -> Decode failed: Masks do not match.")
    print(" - RLE Encode/Decode: OK")

    # Test Metrics
    # Perfect match
    dice_perfect = compute_dice(dummy_mask, dummy_mask)
    if not np.isclose(dice_perfect, 1.0):
        raise AssertionError(
            f"Dice check failed for identical masks. Got {dice_perfect}"
        )

    # Create 3D volume for Hausdorff (D, H, W)
    vol_true = np.zeros((5, 100, 100), dtype=np.uint8)
    vol_true[2] = dummy_mask

    hd_perfect = compute_hausdorff_3d(vol_true, vol_true)
    if not np.isclose(hd_perfect, 0.0):
        raise AssertionError(
            f"Hausdorff check failed for identical volumes. Got {hd_perfect}"
        )
    print(" - Metrics (Dice/Hausdorff): OK\n")

    # ---------------------------------------------------------
    # 3. Verify Dataset Pipeline
    # ---------------------------------------------------------
    print("[3] Verifying Dataset Pipeline...")

    # Load metadata
    df_train = pd.read_csv(CFG.train_csv)
    # Subset for speed before processing
    df_subset = df_train.iloc[: CFG.debug_size].copy()

    # Process 2.5D context
    # We disable loading cached data to force processing on our subset
    df_processed = process_25d_dataframe(
        df_subset, split_name="demo_train", load_cached_data=False
    )

    # Check if columns were added
    if (
        "image_path_prev" not in df_processed.columns
        or "image_path_next" not in df_processed.columns
    ):
        raise AssertionError("2.5D processing failed: Neighbor columns missing.")

    # Instantiate Dataset
    dataset = UWMGIDataset(
        df_processed, label=True, transforms=get_transforms(data="train")
    )

    # Fetch one sample
    img, mask, sample_id = dataset[0]

    # Verify Shapes
    # Image: (3, 320, 320) -> 3 channels for 2.5D
    # Mask: (3, 320, 320) -> 3 classes
    expected_shape = (3, CFG.img_size[0], CFG.img_size[1])

    if img.shape != expected_shape:
        raise AssertionError(
            f"Image shape mismatch. Expected {expected_shape}, got {img.shape}"
        )
    if mask.shape != expected_shape:
        raise AssertionError(
            f"Mask shape mismatch. Expected {expected_shape}, got {mask.shape}"
        )

    print(f" - Dataset Sample ID: {sample_id}")
    print(f" - Image Shape: {img.shape}, Mask Shape: {mask.shape}")
    print(" - Dataset Pipeline: OK\n")

    # ---------------------------------------------------------
    # 4. Verify Model and Loss
    # ---------------------------------------------------------
    print("[4] Verifying Model and Loss...")

    device = CFG.device
    model = build_model()
    model.to(device)
    model.eval()

    # Create a batch
    img_batch = img.unsqueeze(0).to(device)  # (1, 3, 320, 320)
    mask_batch = mask.unsqueeze(0).to(device)  # (1, 3, 320, 320)

    # Forward pass
    with torch.no_grad():
        logits = model(img_batch)

    if logits.shape != mask_batch.shape:
        raise AssertionError(
            f"Model output shape mismatch. Expected {mask_batch.shape}, got {logits.shape}"
        )

    # Loss calculation
    criterion = CompositeLoss().to(device)
    loss = criterion(logits, mask_batch)

    if torch.isnan(loss) or loss.item() < 0:
        raise AssertionError(f"Loss calculation failed. Got {loss.item()}")

    print(f" - Forward Pass Output Shape: {logits.shape}")
    print(f" - Calculated Loss: {loss.item():.4f}")
    print(" - Model & Loss: OK\n")

    # ---------------------------------------------------------
    # 5. Run Training Loop
    # ---------------------------------------------------------
    print("[5] Running Training Loop (1 Epoch, Debug Mode)...")

    # We call the main train function from library.train
    # It will use the CFG settings we modified earlier
    try:
        train()
    except Exception as e:
        raise RuntimeError(f"Training loop failed: {e}")

    # Verify checkpoint creation
    best_model_path = os.path.join(CFG.checkpoint_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            "Training finished but 'best_model.pth' was not created."
        )

    print(" - Training Loop: OK (Checkpoint saved)\n")

    # ---------------------------------------------------------
    # 6. Run Inference
    # ---------------------------------------------------------
    print("[6] Running Inference and Submission Generation...")

    # The inference script loads CFG.test_csv.
    # We need to ensure we process a small subset or the full test set.
    # Since we can't easily subset the test CSV file on disk without writing to input (forbidden),
    # we rely on the inference script reading the full metadata/test.csv.
    # However, for the demo to be fast, we hope the test set isn't massive or we accept the runtime.
    # The provided metadata/test.csv has 6800 rows. Inference on 6800 rows might take > 1 hour on CPU,
    # but we have a GPU (A100).
    # Batch size 4 on A100 is very small, we can increase it for inference speed if needed,
    # but let's stick to the configured 4 to be safe.
    # 6800 / 4 = 1700 batches. It should be fast enough on A100.

    # To be absolutely safe for this "demo" script which requires < 1 hour total including training,
    # we will temporarily mock the test.csv loading in the library by patching pandas read_csv?
    # No, we shouldn't patch library code dynamically if possible.
    # Instead, we can create a temporary test csv in working dir and point CFG to it.

    print(" - Creating subset test metadata for fast inference demo...")
    df_test_full = pd.read_csv("./metadata/test.csv")
    df_test_subset = df_test_full.iloc[:50].copy()  # 50 samples
    temp_test_csv = os.path.join(CFG.working_dir, "test_subset.csv")
    df_test_subset.to_csv(temp_test_csv, index=False)

    # Point CFG to this subset
    CFG.test_csv = temp_test_csv

    try:
        predict_and_submit(load_cached_data=False)
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

    # Verify submission file
    if not os.path.exists(CFG.submission_file):
        raise FileNotFoundError("Submission file was not generated.")

    df_sub = pd.read_csv(CFG.submission_file)
    print(f" - Submission Generated. Rows: {len(df_sub)}")
    print(f" - Sample Prediction: {df_sub.iloc[0].to_dict()}")

    # Check format
    required_cols = {"id", "class", "predicted"}
    if not required_cols.issubset(df_sub.columns):
        raise AssertionError(
            f"Submission columns missing. Expected {required_cols}, got {df_sub.columns}"
        )

    print(" - Inference: OK\n")

    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
