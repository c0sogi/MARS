import os
import torch
import pandas as pd
import numpy as np
import shutil
import json

# Import from the provided library
from library.config import Config
from library.model import RSKARN
from library.loss import CascadedLoss
from library.dataset import GestureDataset
from library.trainer import Trainer
from library.inference import InferenceEngine
from library.utils import calculate_score, run_length_encoding


def main():
    print("=== Starting Demonstration Script ===")

    # ==========================================
    # 1. Setup & Configuration Override
    # ==========================================
    print("\n[1] Configuring environment for demo...")

    # Define demo-specific paths
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config parameters for speed and isolation
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "demo_model.pth")
    Config.SUBMISSION_FILE = os.path.join(DEMO_DIR, "submission.csv")

    # Reduce training load for demo
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # Create directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set seeds
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    # ==========================================
    # 2. Create Mini-Datasets
    # ==========================================
    print("\n[2] Creating mini-datasets from metadata...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Select a small subset (e.g., 5 samples each)
    # Ensure we pick samples that actually exist (metadata should be correct)
    mini_train = orig_train.head(5).copy()
    mini_val = orig_val.head(3).copy()
    mini_test = orig_test.head(3).copy()

    # Save mini metadata
    mini_train_path = os.path.join(DEMO_DIR, "mini_train.csv")
    mini_val_path = os.path.join(DEMO_DIR, "mini_val.csv")
    mini_test_path = os.path.join(DEMO_DIR, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # Point Config to mini files
    Config.TRAIN_CSV = mini_train_path
    Config.VAL_CSV = mini_val_path
    Config.TEST_CSV = mini_test_path

    print(f"    Mini-Train: {len(mini_train)} samples")
    print(f"    Mini-Val:   {len(mini_val)} samples")
    print(f"    Mini-Test:  {len(mini_test)} samples")

    # ==========================================
    # 3. Verify Model Architecture
    # ==========================================
    print("\n[3] Verifying Model Architecture...")

    model = RSKARN().to(Config.DEVICE)

    # Create dummy input: (Batch=2, Time=32, Dim=253)
    dummy_input = torch.randn(2, 32, Config.INPUT_DIM).to(Config.DEVICE)

    # Forward pass
    s1, s2, s3 = model(dummy_input)

    # Check output shapes: Should be (Batch, NumClasses, Time)
    expected_shape = (2, Config.NUM_CLASSES, 32)
    print(f"    Input shape: {dummy_input.shape}")
    print(f"    Stage 1 Output: {s1.shape}")
    print(f"    Stage 2 Output: {s2.shape}")
    print(f"    Stage 3 Output: {s3.shape}")

    assert s1.shape == expected_shape, f"Stage 1 shape mismatch: {s1.shape}"
    assert s2.shape == expected_shape, f"Stage 2 shape mismatch: {s2.shape}"
    assert s3.shape == expected_shape, f"Stage 3 shape mismatch: {s3.shape}"
    print("    Model forward pass successful.")

    # ==========================================
    # 4. Verify Loss Function
    # ==========================================
    print("\n[4] Verifying Loss Function...")

    criterion = CascadedLoss()
    # Dummy targets: (Batch=2, Time=32) with random classes
    dummy_targets = torch.randint(0, Config.NUM_CLASSES, (2, 32)).to(Config.DEVICE)

    loss, metrics = criterion(s1, s2, s3, dummy_targets)

    print(f"    Total Loss: {loss.item():.4f}")
    print(f"    Metrics: {metrics}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    print("    Loss calculation successful.")

    # ==========================================
    # 5. Verify Dataset Loading
    # ==========================================
    print("\n[5] Verifying Dataset Loading (Feature Extraction)...")

    # Initialize dataset (this will compute features and cache them)
    # Using 'train' mode to check windowing
    ds_train = GestureDataset(split="train", mode="train", load_cached_data=False)

    if len(ds_train) > 0:
        feats, labels = ds_train[0]
        print(f"    Sample Feature Shape: {feats.shape}")
        print(f"    Sample Label Shape: {labels.shape}")

        # Check dimensions
        assert (
            feats.shape[1] == Config.INPUT_DIM
        ), f"Feature dim mismatch. Expected {Config.INPUT_DIM}, got {feats.shape[1]}"
        assert (
            feats.shape[0] == Config.WINDOW_SIZE
        ), f"Window size mismatch. Expected {Config.WINDOW_SIZE}, got {feats.shape[0]}"
    else:
        print(
            "    Warning: Dataset is empty (possibly due to very short sequences in mini-set)."
        )

    # ==========================================
    # 6. Run Training Loop
    # ==========================================
    print("\n[6] Running Training Loop...")

    trainer = Trainer()
    trainer.fit()

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved."
    print("    Training completed and model saved.")

    # ==========================================
    # 7. Run Inference & Submission Generation
    # ==========================================
    print("\n[7] Running Inference on Mini-Test Set...")

    inference_engine = InferenceEngine(model_path=Config.MODEL_SAVE_PATH)
    inference_engine.generate_submission()

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created."

    # Inspect submission content
    with open(Config.SUBMISSION_FILE, "r") as f:
        lines = f.readlines()
        print(f"    Generated {len(lines)} prediction lines.")
        if len(lines) > 0:
            print(f"    Sample prediction: {lines[0].strip()}")

    # ==========================================
    # 8. Verify Metric Logic
    # ==========================================
    print("\n[8] Verifying Metric Logic (Levenshtein)...")

    # Test case:
    # GT: [1, 2, 3]
    # Pred: [1, 3] (Deletion of 2) -> Dist 1
    # Pred: [1, 2, 4] (Substitution 3->4) -> Dist 1

    gt = {"s1": [1, 2, 3]}
    pred_del = {"s1": [1, 3]}
    pred_sub = {"s1": [1, 2, 4]}

    score_del = calculate_score(pred_del, gt)
    score_sub = calculate_score(pred_sub, gt)

    # Score = Dist / Total_GT_Gestures = 1 / 3 = 0.333...
    print(f"    Score (Deletion): {score_del:.4f}")
    assert abs(score_del - (1 / 3)) < 1e-5, "Metric calculation failed for deletion."
    assert (
        abs(score_sub - (1 / 3)) < 1e-5
    ), "Metric calculation failed for substitution."

    print("    Metric logic verified.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
