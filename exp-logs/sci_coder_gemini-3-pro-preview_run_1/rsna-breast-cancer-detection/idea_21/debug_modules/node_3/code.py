import os
import sys
import gc
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library components
import importlib
import library.utils
import library.data
import library.engine

importlib.reload(library.utils)
importlib.reload(library.data)
importlib.reload(library.engine)

from library.config import Config
from library.utils import set_seed, probabilistic_f1
from library.data import get_loaders
from library.model import SiameseEfficientNet
from library.engine import run_training, generate_submission


def main():
    # Explicitly release memory and clear tracebacks to prevent OOM (Cite debug_lesson_18, debug_lesson_15)
    if hasattr(sys, "last_traceback"):
        sys.last_traceback = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("==== Starting Demonstration Script ====")

    # 1. Setup and Configuration
    # We use debug=True and epochs=1 to ensure the script completes quickly.
    # We also reduce the debug sample size to minimal for this demonstration.
    print("[1/6] Setting up configuration...")
    Config.setup(debug=True, epochs=1)
    Config.DEBUG_SAMPLE_SIZE = 20  # Small sample size for speed
    set_seed(Config.SEED)

    # 2. Verify Metric Logic
    print("[2/6] Verifying Probabilistic F1 Metric...")
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])  # Good predictions
    score = probabilistic_f1(y_true, y_pred)

    # Check if score is valid (between 0 and 1)
    if not (0.0 <= score <= 1.0):
        raise AssertionError(f"pF1 score {score} is out of valid range [0, 1]")
    print(f"   Metric Check Passed: pF1 = {score:.4f}")

    # 3. Verify Data Loading and Shapes
    print("[3/6] Verifying Data Loading...")
    # Get loaders (this will use the small DEBUG_SAMPLE_SIZE)
    train_loader, val_loader = get_loaders(debug=True, load_cached_data=False)

    # Fetch one batch
    try:
        batch = next(iter(train_loader))
        tgt, contra, label, pids = batch
    except StopIteration:
        raise RuntimeError("DataLoader is empty. Debug sample size might be too small.")

    print(
        f"   Batch Shapes -> Target: {tgt.shape}, Contra: {contra.shape}, Label: {label.shape}"
    )

    # Assertions for data shapes
    # Expected shape: [Batch, Channels=3, Height, Width]
    if tgt.shape[1] != 3:
        raise AssertionError(
            f"Expected 3 input channels (Image+Age+Implant), got {tgt.shape[1]}"
        )
    if tgt.shape != contra.shape:
        raise AssertionError("Target and Contralateral image shapes do not match.")
    if label.shape[0] != tgt.shape[0]:
        raise AssertionError("Label batch size does not match input batch size.")

    print("   Data Loading Verification Passed.")

    # 4. Verify Model Architecture & Forward Pass
    print("[4/6] Verifying Model Forward Pass...")
    device = torch.device(Config.DEVICE)
    # Instantiate model (no pretrained weights needed for shape verification)
    model = SiameseEfficientNet(backbone_name=Config.BACKBONE, pretrained=False)
    model.to(device)
    model.eval()

    with torch.no_grad():
        tgt_dev = tgt.to(device)
        contra_dev = contra.to(device)
        logits = model(tgt_dev, contra_dev)

    print(f"   Output Logits Shape: {logits.shape}")

    # Expected output: [Batch, 1]
    if logits.shape != (tgt.shape[0], 1):
        raise AssertionError(
            f"Expected output shape {(tgt.shape[0], 1)}, got {logits.shape}"
        )

    print("   Model Verification Passed.")

    # 5. Run Training Pipeline
    print("[5/6] Running Training Pipeline (Debug Mode)...")
    # This executes the full training loop for 1 epoch on the small dataset
    run_training(debug=True, epochs=1)

    # Verify model was saved
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")
    print("   Training Pipeline Completed and Model Saved.")

    # 6. Run Submission Pipeline
    print("[6/6] Running Submission Pipeline...")
    generate_submission(debug=True)

    # Verify submission file
    submission_path = os.path.join(Config.WORKING_DIR, Config.SUBMISSION_PATH)
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    # Verify submission content format
    df_sub = pd.read_csv(submission_path)
    if "prediction_id" not in df_sub.columns or "cancer" not in df_sub.columns:
        raise ValueError("Submission file missing required columns.")

    print(f"   Submission Generated with {len(df_sub)} rows.")
    print("==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    main()
