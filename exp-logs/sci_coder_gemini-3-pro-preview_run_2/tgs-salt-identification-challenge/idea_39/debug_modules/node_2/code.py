import os
import shutil
import numpy as np
import torch
import pandas as pd
import cv2

# Import provided library modules
from library.config import Config
from library.utils import set_seed, rle_encode, rle_decode, do_kaggle_metric
from library.dataset import get_dataloaders
from library.model import ResNet34WideLinkNet
from library.losses import CombinedLoss
from library.trainer import Trainer
from library.inference import InferenceEngine


def main():
    print("=== Salt Segmentation Pipeline Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    print("\n[1/7] Setting up environment...")

    # Set seed for reproducibility
    set_seed(42)

    # Clean working directory to ensure fresh execution (removes old cache/checkpoints)
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Re-create necessary directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Override Config for Speed (Debug Mode)
    Config.set_debug_mode(True)  # Sets EPOCHS=2, but we will force 1 below
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2
    # Reduce scan depths for faster inference demo
    Config.SCAN_DEPTHS = [0.0]

    print(f"   Debug Mode: {Config.DEBUG}")
    print(f"   Epochs: {Config.EPOCHS}")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Logic Verification (RLE & Metric)
    # -------------------------------------------------------------------------
    print("\n[2/7] Verifying Utility Logic...")

    # A. RLE Encoding/Decoding
    # Create a dummy mask: 101x101 with a 10x10 square of 1s
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1

    encoded = rle_encode(dummy_mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    assert np.array_equal(dummy_mask, decoded), "RLE Decode mismatch!"
    print("   [PASS] RLE Encoding/Decoding logic verified.")

    # B. Metric Calculation
    # Perfect match
    score_perfect = do_kaggle_metric(
        dummy_mask[None, ...], dummy_mask[None, ...], threshold=0.5
    )
    assert np.isclose(
        score_perfect, 1.0
    ), f"Metric failed perfect match. Got {score_perfect}"

    # No overlap
    dummy_empty = np.zeros((101, 101), dtype=np.uint8)
    score_zero = do_kaggle_metric(
        dummy_mask[None, ...], dummy_empty[None, ...], threshold=0.5
    )
    assert np.isclose(score_zero, 0.0), f"Metric failed zero overlap. Got {score_zero}"
    print("   [PASS] IoU/mAP Metric logic verified.")

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("\n[3/7] Initializing DataLoaders...")

    # get_dataloaders(debug=True) loads a small subset of the data
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Fetch one batch to verify shapes
    images, masks, depths, ids = next(iter(train_loader))

    print(f"   Images Shape: {images.shape}")  # Expect (B, 1, 128, 128)
    print(f"   Masks Shape:  {masks.shape}")  # Expect (B, 1, 128, 128)
    print(f"   Depths Shape: {depths.shape}")  # Expect (B, 1)

    assert images.shape == (Config.BATCH_SIZE, 1, 128, 128), "Invalid Image Batch Shape"
    assert masks.shape == (Config.BATCH_SIZE, 1, 128, 128), "Invalid Mask Batch Shape"
    assert depths.shape == (Config.BATCH_SIZE, 1), "Invalid Depth Batch Shape"
    print("   [PASS] Data Loading verified.")

    # -------------------------------------------------------------------------
    # 4. Model & Loss Initialization
    # -------------------------------------------------------------------------
    print("\n[4/7] Initializing Model and Loss...")

    model = ResNet34WideLinkNet().to(Config.DEVICE)
    criterion = CombinedLoss()

    # Prepare inputs
    img_tensor = images.to(Config.DEVICE, dtype=torch.float32)
    depth_tensor = depths.to(Config.DEVICE, dtype=torch.float32)
    mask_tensor = masks.to(Config.DEVICE, dtype=torch.float32)

    # Forward Pass
    outputs = model(img_tensor, depth_tensor)
    logits, aux_pred = outputs

    print(f"   Logits Shape: {logits.shape}")
    print(f"   Aux Pred Shape: {aux_pred.shape}")

    assert logits.shape == (Config.BATCH_SIZE, 1, 128, 128), "Invalid Logits Shape"
    assert aux_pred.shape == (Config.BATCH_SIZE, 1), "Invalid Aux Shape"

    # Loss Calculation
    loss = criterion(outputs, mask_tensor, depth_tensor)
    print(f"   Initial Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() > 0, "Loss must be positive"
    print("   [PASS] Model and Loss verified.")

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    print("\n[5/7] Running Training Loop (1 Epoch)...")

    trainer = Trainer(model, train_loader, val_loader, Config.DEVICE)
    trainer.fit()

    # Verify checkpoint creation
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not created!"
    print(f"   [PASS] Training complete. Checkpoint saved at: {best_model_path}")

    # -------------------------------------------------------------------------
    # 6. Inference & Threshold Optimization
    # -------------------------------------------------------------------------
    print("\n[6/7] Running Inference Engine...")

    # Load the best model
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))

    engine = InferenceEngine(model, Config.DEVICE)

    # Optimize Threshold (using validation set)
    best_threshold = engine.optimize_threshold(val_loader)
    print(f"   Optimized Threshold: {best_threshold:.4f}")
    assert 0.0 < best_threshold < 1.0, "Threshold optimization returned invalid value"

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[7/7] Generating Submission...")

    # Generate submission for test set
    sub_df = engine.generate_submission(test_loader, threshold=best_threshold)

    # Verification
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found!"
    assert len(sub_df) > 0, "Submission DataFrame is empty!"
    assert list(sub_df.columns) == ["id", "rle_mask"], "Submission columns mismatch!"

    # Check if RLE strings are valid (string or empty)
    sample_rle = sub_df.iloc[0]["rle_mask"]
    print(f"   Sample RLE (First Row): '{sample_rle}'")

    print("\n=== All Tests Passed Successfully ===")


if __name__ == "__main__":
    main()
