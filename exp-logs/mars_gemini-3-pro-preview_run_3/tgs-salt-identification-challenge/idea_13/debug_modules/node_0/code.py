import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, rle_encode, rle_decode, calculate_iou_map
from library.dataset import make_loader
from library.model import SaltUNetPlusPlus
from library.training import train_fold
from library.inference import InferenceRunner


def main():
    print("=== Starting Salt Segmentation Demo ===")

    # 1. Setup and Configuration Overrides
    # We override config values to make this demo run fast (1 epoch, small subset)
    print("Configuring environment for rapid demonstration...")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_FOLDS = 2  # We will only run fold 0
    Config.DEBUG = True

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds
    seed_everything(Config.SEED)

    # 2. Create Data Subsets for Speed
    # We read the original metadata, take a tiny slice, and save it as temp metadata
    # Then we point Config to these new files.
    print("Creating data subsets...")

    # Train Subset
    full_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    subset_train = full_train.head(16).copy()  # 16 samples
    temp_train_path = os.path.join(Config.WORKING_DIR, "temp_train.csv")
    subset_train.to_csv(temp_train_path, index=False)
    Config.TRAIN_METADATA_PATH = temp_train_path

    # Val Subset
    full_val = pd.read_csv(Config.VAL_METADATA_PATH)
    subset_val = full_val.head(8).copy()  # 8 samples
    temp_val_path = os.path.join(Config.WORKING_DIR, "temp_val.csv")
    subset_val.to_csv(temp_val_path, index=False)
    Config.VAL_METADATA_PATH = temp_val_path

    # Test Subset
    full_test = pd.read_csv(Config.TEST_METADATA_PATH)
    subset_test = full_test.head(8).copy()  # 8 samples
    temp_test_path = os.path.join(Config.WORKING_DIR, "temp_test.csv")
    subset_test.to_csv(temp_test_path, index=False)
    Config.TEST_METADATA_PATH = temp_test_path

    print(
        f"Subsets created. Train: {len(subset_train)}, Val: {len(subset_val)}, Test: {len(subset_test)}"
    )

    # 3. Verify Utility Functions
    print("\n=== Verifying Utilities ===")

    # RLE Roundtrip
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1
    rle = rle_encode(dummy_mask)
    decoded = rle_decode(rle, shape=(101, 101))
    assert np.array_equal(dummy_mask, decoded), "RLE Roundtrip failed!"
    print("RLE Encode/Decode: OK")

    # IoU Calculation
    # Perfect match
    iou_perfect = calculate_iou_map([dummy_mask], [dummy_mask], verbose=False)
    assert iou_perfect == 1.0, f"IoU Perfect Match failed: {iou_perfect}"

    # No overlap
    dummy_mask_2 = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask_2[50:60, 50:60] = 1
    iou_zero = calculate_iou_map([dummy_mask], [dummy_mask_2], verbose=False)
    # Note: calculate_iou_map calculates precision at thresholds.
    # If IoU is 0, it never passes threshold 0.5, so score is 0.
    assert iou_zero == 0.0, f"IoU Zero Match failed: {iou_zero}"
    print("IoU Metric Calculation: OK")

    # 4. Verify Model and Data Loading
    print("\n=== Verifying Data Loading & Model ===")

    # Create Loaders
    train_loader = make_loader(
        subset_train, phase="train", batch_size=Config.BATCH_SIZE, load_cached=False
    )
    val_loader = make_loader(
        subset_val, phase="val", batch_size=Config.BATCH_SIZE, load_cached=False
    )

    # Check Batch
    imgs, masks, ids = next(iter(train_loader))
    print(f"Batch Shapes -> Images: {imgs.shape}, Masks: {masks.shape}")
    assert imgs.shape == (Config.BATCH_SIZE, 3, 128, 128), "Incorrect Image Batch Shape"
    assert masks.shape == (Config.BATCH_SIZE, 1, 128, 128), "Incorrect Mask Batch Shape"

    # Initialize Model
    device = Config.DEVICE
    model = SaltUNetPlusPlus().to(device)

    # Forward Pass (Training Mode)
    model.train()
    imgs = imgs.to(device)
    outputs = model(imgs)

    # Deep Supervision check: Should return list of 4 tensors
    assert isinstance(
        outputs, list
    ), "Model in training mode should return a list (Deep Supervision)"
    assert (
        len(outputs) == 4
    ), f"Expected 4 outputs from Deep Supervision, got {len(outputs)}"
    assert outputs[0].shape == (Config.BATCH_SIZE, 1, 128, 128), "Output shape mismatch"
    print("Model Forward Pass (Deep Supervision): OK")

    # 5. Training Demonstration
    print("\n=== Running Training Demo (1 Epoch, Fold 0) ===")
    # We use the train_fold helper which handles the loop, validation, and saving
    best_score = train_fold(train_loader, val_loader, fold_idx=0)

    print(f"Training completed. Best Val mAP: {best_score}")

    # Check if checkpoint exists
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "fold_0_best.pth")
    assert os.path.exists(ckpt_path), "Checkpoint file was not created!"
    print(f"Checkpoint verified at: {ckpt_path}")

    # 6. Inference & Submission Demonstration
    print("\n=== Running Inference Demo ===")
    runner = InferenceRunner(device=device)

    # A. Optimize Threshold (Simulated)
    # We will manually generate predictions on our small val subset using the trained model
    # to demonstrate the optimize_threshold function.
    print("Generating predictions for threshold optimization...")
    model.eval()
    val_preds = []
    val_gts = []

    with torch.no_grad():
        for images, masks, _ in val_loader:
            images = images.to(device)
            # Use TTA prediction from runner
            probs = runner._predict_tta(model, images)

            # Resize masks to 101x101 for comparison
            masks_resized = torch.nn.functional.interpolate(
                masks.float(), size=(101, 101), mode="nearest"
            )

            val_preds.append(probs.cpu().numpy())
            val_gts.append(masks_resized.cpu().numpy())

    val_preds = np.concatenate(val_preds).squeeze(1)
    val_gts = np.concatenate(val_gts).squeeze(1)

    # Run optimization
    best_th = runner.optimize_threshold(val_preds, val_gts)
    print(f"Optimized Threshold: {best_th}")

    # B. Generate Submission
    # This uses the subset_test metadata we created earlier
    print("Generating submission file...")
    runner.generate_submission(threshold=best_th)

    # Verify submission
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not found!"
    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    assert len(sub_df) == len(
        subset_test
    ), f"Submission rows {len(sub_df)} != Test rows {len(subset_test)}"
    assert (
        "id" in sub_df.columns and "rle_mask" in sub_df.columns
    ), "Submission columns missing"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
