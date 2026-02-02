import os
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, levenshtein_distance, decode_predictions
from library.model import NMD_CRCN
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Setup Temporary Workspace & Configuration Overrides
    # -------------------------------------------------------------------------
    print("\n[Step 1] Setting up temporary workspace and configuration...")

    # Define temporary paths
    base_demo_dir = "./working/demo_task"
    temp_metadata_dir = os.path.join(base_demo_dir, "metadata")
    temp_cache_dir = os.path.join(base_demo_dir, "cache")
    temp_checkpoints_dir = os.path.join(base_demo_dir, "checkpoints")
    temp_submission_dir = os.path.join(base_demo_dir, "submission")

    # Clean up if exists
    if os.path.exists(base_demo_dir):
        shutil.rmtree(base_demo_dir)

    # Create directories
    for d in [
        temp_metadata_dir,
        temp_cache_dir,
        temp_checkpoints_dir,
        temp_submission_dir,
    ]:
        os.makedirs(d, exist_ok=True)

    # Override Config to use these temporary paths and reduce workload
    Config.WORKING_DIR = base_demo_dir
    Config.METADATA_DIR = temp_metadata_dir
    Config.CACHE_DIR = temp_cache_dir
    Config.CHECKPOINTS_DIR = temp_checkpoints_dir
    Config.SUBMISSION_DIR = temp_submission_dir

    # Speed optimizations
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.PATIENCE = 1

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Epochs: {Config.NUM_EPOCHS}, Batch Size: {Config.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Create Data Subsets for Speed
    # -------------------------------------------------------------------------
    print("\n[Step 2] Creating metadata subsets...")

    original_meta_dir = "./metadata"

    # Helper to subset and save
    def subset_csv(filename, n_rows):
        src = os.path.join(original_meta_dir, filename)
        dst = os.path.join(temp_metadata_dir, filename)
        if os.path.exists(src):
            df = pd.read_csv(src)
            # Take a small subset
            df_subset = df.head(n_rows)
            df_subset.to_csv(dst, index=False)
            print(f"  -> Created {filename} with {len(df_subset)} samples.")
        else:
            raise FileNotFoundError(f"Original metadata {src} not found.")

    subset_csv("train.csv", 20)
    subset_csv("val.csv", 10)
    subset_csv("test.csv", 5)

    # -------------------------------------------------------------------------
    # 3. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying utility functions...")

    # Test Levenshtein
    seq_a = [1, 2, 3]
    seq_b = [1, 2, 3]
    dist_eq = levenshtein_distance(seq_a, seq_b)
    assert (
        dist_eq == 0
    ), f"Levenshtein distance for identical sequences should be 0, got {dist_eq}"

    seq_c = [1, 2]
    dist_diff = levenshtein_distance(seq_a, seq_c)
    assert dist_diff == 1, f"Levenshtein distance should be 1, got {dist_diff}"
    print("  -> Levenshtein distance logic verified.")

    # Test Decode Predictions
    # 0 is background. Sequence: 0, 1, 1, 0, 2, 2, 2, 0 -> [1, 2]
    raw_preds = [0, 0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0]
    decoded = decode_predictions(raw_preds, background_class=0)
    assert decoded == [1, 2], f"Decoding failed. Expected [1, 2], got {decoded}"
    print("  -> Decode predictions logic verified.")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[Step 4] Verifying Model Architecture...")

    set_seed(Config.SEED)
    model = NMD_CRCN()
    model.eval()

    # Create dummy input: (Batch=2, Time=30, Features=85)
    dummy_input = torch.randn(2, 30, Config.INPUT_DIM)
    dummy_mask = torch.ones(2, 30)

    with torch.no_grad():
        outputs = model(dummy_input, dummy_mask)

    # Check outputs
    assert "stage1" in outputs and "stage2" in outputs and "stage3" in outputs
    expected_shape = (2, 30, Config.NUM_CLASSES)

    assert (
        outputs["stage1"].shape == expected_shape
    ), f"Stage 1 shape mismatch: {outputs['stage1'].shape}"
    assert (
        outputs["stage2"].shape == expected_shape
    ), f"Stage 2 shape mismatch: {outputs['stage2'].shape}"
    assert (
        outputs["stage3"].shape == expected_shape
    ), f"Stage 3 shape mismatch: {outputs['stage3'].shape}"

    print("  -> Model forward pass and output shapes verified.")

    # -------------------------------------------------------------------------
    # 5. Run Training Pipeline
    # -------------------------------------------------------------------------
    print("\n[Step 5] Running Training Pipeline (Subset)...")

    trainer = Trainer()

    # This will load data (processing to cache first), then train for 2 epochs
    trainer.fit()

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"  -> Checkpoint created at {checkpoint_path}")
    else:
        raise FileNotFoundError("Checkpoint was not created after training.")

    # -------------------------------------------------------------------------
    # 6. Run Inference Pipeline
    # -------------------------------------------------------------------------
    print("\n[Step 6] Running Inference Pipeline (Subset)...")

    trainer.predict()

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if os.path.exists(submission_path):
        print(f"  -> Submission file created at {submission_path}")

        # Verify content format
        df_sub = pd.read_csv(submission_path)
        print("  -> Submission Head:")
        print(df_sub.head())

        assert (
            "Id" in df_sub.columns and "Sequence" in df_sub.columns
        ), "Submission columns mismatch"
        assert len(df_sub) == 5, f"Expected 5 predictions, got {len(df_sub)}"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
