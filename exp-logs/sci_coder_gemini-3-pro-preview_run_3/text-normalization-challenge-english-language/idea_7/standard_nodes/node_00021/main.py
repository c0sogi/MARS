import sys
import os
import pandas as pd
import numpy as np
import torch

# 1. Configuration Override for Performance
# Must be done before initializing other modules that rely on Config
from library.config import Config

# Optimize for A100 GPU
Config.BATCH_SIZE = 256
Config.NUM_EPOCHS = 5
Config.SOFT_FILTER_RATIO = 0.05  # Keep default to ensure stability
Config.WORK_DIR = "./working/idea_7"

# Ensure directories exist (Config does this, but good to be explicit if we changed paths)
os.makedirs(Config.WORK_DIR, exist_ok=True)

# Import Library Modules
from library.trainer import Trainer
from library.inference import CascadePredictor, generate_submission
from library.symbolic_model import SymbolicMemory
from library.data_utils import process_context


def main():
    # Set global seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    print("=== Starting Runfile Execution ===")

    # ---------------------------------------------------------
    # Step 1: Train Symbolic Model
    # ---------------------------------------------------------
    print("\n[Step 1/5] Training Symbolic Model...")
    sym_model = SymbolicMemory()
    # This aggregates stats from the full training set
    sym_model.fit(load_cached_data=True)

    # ---------------------------------------------------------
    # Step 2: Train Neural Model
    # ---------------------------------------------------------
    print("\n[Step 2/5] Training Neural Model...")
    trainer = Trainer()
    # Trains on soft-filtered data (Hard samples + 5% Easy samples)
    trainer.fit(load_cached_data=True)

    # ---------------------------------------------------------
    # Step 3: Full Validation (Cascade)
    # ---------------------------------------------------------
    print("\n[Step 3/5] Running Full Validation...")

    # Load Validation Data
    if not os.path.exists(Config.VAL_META_PATH):
        raise FileNotFoundError(
            f"Validation metadata not found at {Config.VAL_META_PATH}"
        )

    df_val = pd.read_parquet(Config.VAL_META_PATH)

    # Initialize Predictor (Loads the best neural model checkpoint)
    predictor = CascadePredictor()

    # Run Inference on Validation Set
    # Note: process_context is handled inside predict if needed, but we can do it explicitly
    # to ensure we have control for analysis later
    df_val = process_context(df_val)
    preds = predictor.predict(df_val)

    # Attach predictions
    df_val["pred"] = preds

    # Calculate Metric
    # Ensure strict string comparison
    df_val["after"] = df_val["after"].astype(str)
    df_val["pred"] = df_val["pred"].astype(str)

    correct_mask = df_val["pred"] == df_val["after"]
    accuracy = correct_mask.mean()

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {accuracy}")

    # ---------------------------------------------------------
    # Step 4: Failure Analysis
    # ---------------------------------------------------------
    print("\n[Step 4/5] Failure Analysis...")

    # Identify errors
    df_val["is_error"] = (~correct_mask).astype(int)

    # Feature: Input Length
    df_val["len_before"] = df_val["before"].str.len()

    # Calculate Correlation
    if df_val["is_error"].sum() > 0:
        corr_len = df_val["len_before"].corr(df_val["is_error"])
        print(f"Correlation between Input Length and Error: {corr_len}")
    else:
        print("No errors found in validation set. Correlation undefined.")

    # ---------------------------------------------------------
    # Step 5: Submission
    # ---------------------------------------------------------
    print("\n[Step 5/5] Submission Generation...")

    THRESHOLD = 0.9943860453286453

    if accuracy > THRESHOLD:
        print(f"Validation accuracy {accuracy} > {THRESHOLD}. Generating submission...")
        generate_submission(load_cached_data=True)
    else:
        print(
            f"Validation accuracy {accuracy} <= {THRESHOLD}. Skipping submission generation."
        )

    print("\n=== Execution Complete ===")


if __name__ == "__main__":
    main()
