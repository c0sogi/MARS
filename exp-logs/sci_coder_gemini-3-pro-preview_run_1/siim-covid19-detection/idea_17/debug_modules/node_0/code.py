import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import cv2
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, mask2bbox, calculate_map
from library.dataset import load_data, SIIMDataset, get_transforms
from library.model import StochasticResNet34UNet
from library.engine import train_model, predict


def main():
    print("=== SIIM-FISABIO-RSNA COVID-19 Detection Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring Environment...")

    # Set a specific working directory for this execution to avoid cache conflicts
    # and ensure we can force re-processing of data.
    Config.WORKING_DIR = "./working/demo_execution_v2"
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Clean up previous run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Initialize directories and seeds
    Config.setup()

    # Set Hyperparameters for Speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 2

    # Enable Debug mode initially for Training (uses small data subset)
    Config.DEBUG = True
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utilities...")

    # Test mask2bbox
    # Create a 100x100 mask with a 20x20 square at (10, 10)
    dummy_mask = np.zeros((100, 100), dtype=np.float32)
    dummy_mask[10:30, 10:30] = 1.0
    bboxes = mask2bbox(dummy_mask, threshold=0.5)

    assert len(bboxes) == 1, "mask2bbox failed to detect object"
    # Format: x_min, y_min, x_max, y_max
    b = bboxes[0]
    # Allow 1px tolerance due to contour approximation
    assert abs(b[0] - 10) <= 1 and abs(b[1] - 10) <= 1, f"BBox coord error: {b}"
    print("    mask2bbox: PASSED")

    # Test calculate_map
    # Perfect match scenario
    pred_boxes = [np.array([[10, 10, 30, 30]])]
    pred_scores = [np.array([1.0])]
    pred_labels = [np.array([0])]
    gt_boxes = [np.array([[10, 10, 30, 30]])]
    gt_labels = [np.array([0])]

    map_score = calculate_map(
        pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, num_classes=1
    )
    assert np.isclose(
        map_score, 1.0
    ), f"calculate_map failed (Expected 1.0, got {map_score})"
    print("    calculate_map: PASSED")

    # -------------------------------------------------------------------------
    # 3. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model...")
    model = StochasticResNet34UNet()
    model.eval()

    # Dummy input: Batch=2, Channels=3, Size=512x512
    dummy_in = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    with torch.no_grad():
        study_logits, mask_logits = model(dummy_in)

    # Check shapes
    assert study_logits.shape == (
        2,
        4,
    ), f"Study logits shape mismatch: {study_logits.shape}"
    assert mask_logits.shape == (
        2,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Mask logits shape mismatch: {mask_logits.shape}"
    print("    Model Forward Pass: PASSED")

    # -------------------------------------------------------------------------
    # 4. Training Loop (Debug Mode)
    # -------------------------------------------------------------------------
    print("\n[4] Running Training (Debug Mode)...")
    # Config.DEBUG is True, so this will process 50 train/val images
    train_model()

    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError("Training failed to produce best_model.pth")
    print("    Training Complete. Model saved.")

    # -------------------------------------------------------------------------
    # 5. Inference Loop (Full Test Mode)
    # -------------------------------------------------------------------------
    print("\n[5] Running Inference...")

    # CRITICAL: Switch DEBUG to False for inference.
    # The 'predict' function in engine.py iterates the full test metadata CSV.
    # We need load_data to return the full test set to ensure alignment.
    # The test set is small (638 images), so this is fast enough.
    Config.DEBUG = False

    # Force re-loading of test data by ensuring no test cache exists
    # (Since we just created the dir, there is no test cache yet, so this is fine)
    predict()

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Inference failed to produce submission.csv")

    # -------------------------------------------------------------------------
    # 6. Validate Submission
    # -------------------------------------------------------------------------
    print("\n[6] Validating Submission...")
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    print(f"    Submission Rows: {len(sub_df)}")
    if len(sub_df) == 0:
        raise ValueError("Submission file is empty")

    required_cols = ["id", "PredictionString"]
    missing = [c for c in required_cols if c not in sub_df.columns]
    if missing:
        raise ValueError(f"Submission missing columns: {missing}")

    # Check format of a prediction string
    sample_pred = sub_df.iloc[0]["PredictionString"]
    print(f"    Sample ID: {sub_df.iloc[0]['id']}")
    print(f"    Sample Pred: {sample_pred[:50]}...")

    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    main()
