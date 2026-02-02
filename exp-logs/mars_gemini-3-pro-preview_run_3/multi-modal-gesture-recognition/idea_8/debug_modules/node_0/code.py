import sys
import os
import shutil
import pandas as pd
import numpy as np
import torch

# 1. Setup paths to include the library directory
sys.path.append(os.path.abspath("./library"))

# 2. Import Library Modules
from config import Config
from utils import set_seed, compute_levenshtein_score
from data_loader import ItalianGestureDataset
from trainer import Trainer


def run_demo():
    print("=== Starting Demo Execution ===")

    # --- Configuration Overrides for Speed & Demo ---
    # Create a separate working directory for the demo to avoid clutter
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Setting up demo environment in {demo_dir}...")

    # Override Config global attributes
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.OUTPUT_DIR = os.path.join(demo_dir, "outputs")
    Config.SUBMISSION_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")
    Config.BEST_MODEL_PATH = os.path.join(Config.OUTPUT_DIR, "demo_model.pth")

    # Reduce training parameters for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.PATIENCE = 1

    # Ensure directories exist
    Config.setup_directories()

    # --- Prepare Mini Metadata ---
    # Create subsets of the metadata to run a fast cycle
    print("Creating mini-dataset metadata...")

    def create_mini_csv(src_path, dst_path, n=10):
        if os.path.exists(src_path):
            df = pd.read_csv(src_path)
            # Take top n samples
            df_mini = df.head(n)
            df_mini.to_csv(dst_path, index=False)
            return True
        return False

    mini_train_path = os.path.join(demo_dir, "train.csv")
    mini_val_path = os.path.join(demo_dir, "val.csv")
    mini_test_path = os.path.join(demo_dir, "test.csv")

    # Create mini files from original metadata
    if not create_mini_csv(Config.TRAIN_METADATA_PATH, mini_train_path, n=10):
        raise FileNotFoundError("Original train metadata not found.")
    create_mini_csv(Config.VAL_METADATA_PATH, mini_val_path, n=5)
    create_mini_csv(Config.TEST_METADATA_PATH, mini_test_path, n=5)

    # Point Config to these new mini files
    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path
    Config.TEST_METADATA_PATH = mini_test_path

    # --- Patch Dataset for Dimension Consistency ---
    # The Model expects input shape (Batch, Dim, Time) for its Conv1d layers.
    # The Dataset currently returns (Time, Dim).
    # We patch __getitem__ to transpose features to (Dim, Time).

    print("Patching Dataset to align dimensions (Time, Dim) -> (Dim, Time)...")
    original_getitem = ItalianGestureDataset.__getitem__

    def patched_getitem(self, idx):
        # Original returns: features, cls_lbl, bnd_lbl, sample_id, start
        features, cls_lbl, bnd_lbl, sample_id, start = original_getitem(self, idx)

        # features is a Tensor of shape (Time, Dim)
        # Transpose to (Dim, Time)
        if isinstance(features, torch.Tensor):
            features = features.transpose(0, 1)

        return features, cls_lbl, bnd_lbl, sample_id, start

    ItalianGestureDataset.__getitem__ = patched_getitem

    # --- Instantiate Trainer ---
    # This will initialize DataLoaders using our patched Dataset and mini metadata
    print("Initializing Trainer...")
    trainer = Trainer()

    # --- Run Training Loop ---
    # This runs training and validation for the configured epochs (1)
    print("Running Training Loop (Fit)...")
    trainer.fit()

    # --- Run Inference ---
    # This generates predictions for the test set
    print("Running Inference on Test Set...")
    trainer.predict_test()

    # --- Validation of Outputs ---
    print("Validating outputs...")
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"SUCCESS: Submission file generated at {Config.SUBMISSION_PATH}")
        # Print first few lines to verify format
        with open(Config.SUBMISSION_PATH, "r") as f:
            print("--- Submission File Head ---")
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                print(line.strip())
            print("----------------------------")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    # --- Verify Metric Logic ---
    print("Verifying Levenshtein Metric Logic...")
    # Test case:
    # GT: Sequence [1, 2, 3]
    # Pred: Sequence [1, 2] (Deletion of 3, cost 1)
    # Distance = 1. Total GT Length = 3. Score = 1/3 = 0.333...
    gt = {"s1": [1, 2, 3]}
    preds = {"s1": [1, 2]}

    score = compute_levenshtein_score(preds, gt)
    print(f"Calculated Score: {score:.4f}")

    # Allow for floating point precision issues
    assert (
        abs(score - (1.0 / 3.0)) < 1e-6
    ), f"Metric calculation failed. Expected 0.333, got {score}"

    print("=== Demo execution completed successfully ===")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)
    run_demo()
