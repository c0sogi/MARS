import os
import sys
import shutil
import numpy as np
import torch
import pandas as pd
import cv2
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
# Note: We import Config first to monkey-patch it before other modules use it extensively
from library.config import Config
from library.utils import seed_everything, mask2bbox, compute_iou, calculate_map
from library.dataset import prepare_data, SIIMDataset, get_transforms
from library.model import ResNet18UNetMultiScale
from library.train import run_training
from library.predict import inference


def main():
    print("Starting SIIM-FISABIO-RSNA COVID-19 Detection Demo...")

    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print("\n[1/6] Configuring environment for demo...")

    # Monkey-patch Config for speed and isolation
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.OUTPUT_DIR = "./working/demo_execution"
    Config.MODEL_PATH = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.OUTPUT_DIR, "submission.csv")

    # Ensure clean slate for demo directory
    if os.path.exists(Config.OUTPUT_DIR):
        shutil.rmtree(Config.OUTPUT_DIR)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print(f"Output Directory: {Config.OUTPUT_DIR}")

    # =========================================================================
    # 2. Utility Verification
    # =========================================================================
    print("\n[2/6] Verifying utility functions...")

    # Test compute_iou
    box_a = [0, 0, 10, 10]
    box_b = [5, 0, 15, 10]  # Overlap is 5x10 = 50. Union is 100+100-50 = 150. IoU = 1/3
    iou = compute_iou(box_a, box_b)
    assert (
        abs(iou - 0.3333) < 1e-3
    ), f"IoU calculation failed: expected ~0.333, got {iou}"

    # Test mask2bbox
    # Create a 100x100 mask with a filled rectangle at 10,10 to 30,30
    dummy_mask = np.zeros((100, 100), dtype=np.uint8)
    dummy_mask[10:30, 10:30] = 1
    boxes = mask2bbox(dummy_mask, threshold=0.5)

    assert len(boxes) == 1, "mask2bbox should detect exactly one region"
    # cv2.boundingRect returns x, y, w, h. mask2bbox converts to x1, y1, x2, y2
    # Expected: 10, 10, 30, 30
    b = boxes[0]
    assert (
        b[0] == 10 and b[1] == 10 and b[2] == 30 and b[3] == 30
    ), f"mask2bbox coordinates incorrect: {b}"

    print("Utilities verified successfully.")

    # =========================================================================
    # 3. Data Pipeline Verification
    # =========================================================================
    print("\n[3/6] Verifying data pipeline...")

    # Run prepare_data in debug mode (processes ~50 images)
    # This will create cached .npy files in Config.OUTPUT_DIR
    train_images, train_masks, train_labels, train_ids = prepare_data(
        "train", load_cached_data=False, debug=True
    )

    assert len(train_images) > 0, "No training images loaded"
    assert train_images.shape[1:] == (
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch: {train_images.shape}"
    assert train_masks.shape[1:] == (
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Mask shape mismatch"
    assert train_labels.shape[1] == Config.NUM_CLASSES, "Label dimension mismatch"

    # Instantiate Dataset and Loader
    transforms = get_transforms("train")
    dataset = SIIMDataset(
        train_images, train_masks, train_labels, ids=train_ids, transforms=transforms
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    # Fetch one batch
    imgs, msks, lbls, idxs = next(iter(loader))

    # Verify Tensor Shapes
    # Image: (B, 3, H, W) - converted to RGB in dataset
    assert imgs.shape == (
        2,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Batch image shape incorrect: {imgs.shape}"
    # Mask: (B, 1, H, W)
    assert msks.shape == (
        2,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Batch mask shape incorrect: {msks.shape}"
    # Label: (B, 4)
    assert lbls.shape == (2, 4), f"Batch label shape incorrect: {lbls.shape}"

    print("Data pipeline verified successfully.")

    # =========================================================================
    # 4. Model Architecture Verification
    # =========================================================================
    print("\n[4/6] Verifying model architecture...")

    device = torch.device("cpu")  # Use CPU for simple shape check
    model = ResNet18UNetMultiScale(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.to(device)
    model.eval()

    with torch.no_grad():
        # Pass the batch fetched from loader
        logit_mask, logit_cls = model(imgs.to(device))

    # Check outputs
    # logit_mask: (B, 1, H, W)
    assert logit_mask.shape == (
        2,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Model mask output shape mismatch"
    # logit_cls: (B, 4)
    assert logit_cls.shape == (2, 4), "Model class output shape mismatch"

    print("Model architecture verified successfully.")

    # =========================================================================
    # 5. Training Loop Execution
    # =========================================================================
    print("\n[5/6] Executing training loop (1 Epoch, Debug Mode)...")

    # This function uses the Config we monkey-patched earlier
    run_training(debug=True)

    # Verify model checkpoint was saved
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Training failed to save model at {Config.MODEL_PATH}")

    print(f"Training complete. Model saved to {Config.MODEL_PATH}")

    # =========================================================================
    # 6. Inference & Submission
    # =========================================================================
    print("\n[6/6] Executing inference pipeline...")

    # Run inference using the trained model
    df_submission = inference(debug=True)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Inference failed to save submission at {Config.SUBMISSION_PATH}"
        )

    # Verify submission content format
    expected_cols = ["id", "PredictionString"]
    assert (
        list(df_submission.columns) == expected_cols
    ), f"Submission columns mismatch. Got {df_submission.columns}"

    # Check for study and image rows
    study_rows = df_submission[df_submission["id"].str.contains("_study")]
    image_rows = df_submission[df_submission["id"].str.contains("_image")]

    assert len(study_rows) > 0, "Submission missing study rows"
    assert len(image_rows) > 0, "Submission missing image rows"

    # Check prediction string format for a study row
    sample_pred = study_rows.iloc[0]["PredictionString"]
    parts = sample_pred.split()
    # Format: class conf 0 0 1 1 (6 parts)
    assert len(parts) >= 6, f"Invalid prediction string format: {sample_pred}"
    assert parts[2:] == [
        "0",
        "0",
        "1",
        "1",
    ], f"Study prediction must end with 1-pixel box: {sample_pred}"

    print(f"Inference complete. Submission generated with {len(df_submission)} rows.")
    print("\nAll demonstration steps completed successfully!")


if __name__ == "__main__":
    main()
