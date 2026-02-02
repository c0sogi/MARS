import os
import pandas as pd
import numpy as np
import torch
from library.config import Config
from library.symbolic_stats import StatsBuilder
from library.training import run_training
from library.inference import CascadePredictor
from library.neural_data import _add_context


def main():
    # ==========================================
    # 1. Configuration Overrides for Fast Baseline
    # ==========================================
    # We enable debug mode and limit data/epochs to ensure completion within 1 hour.
    Config.DEBUG = True
    Config.DEBUG_SIZE = 150000  # Sufficient for baseline learning
    Config.NUM_EPOCHS = 3  # Fast convergence check
    Config.BATCH_SIZE = 256  # Efficient GPU utilization

    print("Configuration set for fast baseline execution.")
    print(f"Device: {Config.DEVICE}")

    # ==========================================
    # 2. Build Symbolic Statistics (Stage 1)
    # ==========================================
    print("\n--- Step 1: Building Symbolic Statistics ---")
    stats_builder = StatsBuilder()
    stats_builder.run(load_cached_data=True)

    # ==========================================
    # 3. Train Neural Model (Stage 2 & 3)
    # ==========================================
    print("\n--- Step 2: Training Neural Model ---")
    # run_training handles data preparation (filtering hard samples) and the training loop
    run_training(load_cached_data=True)

    # ==========================================
    # 4. Full Validation
    # ==========================================
    print("\n--- Step 3: Running Full Validation ---")

    # Load the full validation set (not just the neural subset)
    if not os.path.exists(Config.VAL_META):
        raise FileNotFoundError(f"Validation metadata not found at {Config.VAL_META}")

    df_val = pd.read_parquet(Config.VAL_META)

    # Preprocessing: Ensure strings and add context (prev/next)
    df_val["before"] = df_val["before"].astype(str)
    df_val["after"] = df_val["after"].astype(str)
    df_val = _add_context(df_val)

    # Initialize Predictor (Loads Symbolic Stats + Trained Neural Model)
    predictor = CascadePredictor()

    # Run Inference
    print(f"Predicting on {len(df_val)} validation samples...")
    preds_map = predictor.predict(df_val)

    # Map predictions back to DataFrame
    # Fill missing predictions with empty string (though cascade should handle all)
    df_val["pred"] = df_val["id"].map(preds_map).fillna("")

    # Calculate Accuracy
    df_val["correct"] = df_val["after"] == df_val["pred"]
    accuracy = df_val["correct"].mean()

    # Print required metric format
    print(f"Final Validation Metric: {accuracy}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\n--- Step 4: Failure Analysis ---")

    # Define Error: 1 if incorrect, 0 if correct
    df_val["error"] = (~df_val["correct"]).astype(int)

    # Generate simple features for correlation analysis
    df_val["len_input"] = df_val["before"].str.len()
    df_val["is_alnum"] = df_val["before"].str.isalnum().astype(int)
    df_val["is_upper"] = df_val["before"].str.isupper().astype(int)

    # Calculate correlations
    features = ["len_input", "is_alnum", "is_upper"]
    correlations = df_val[features + ["error"]].corr()["error"].drop("error")

    print("Correlation between Error and Input Features:")
    print(correlations)

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    THRESHOLD = 0.9943860453286453

    if accuracy > THRESHOLD:
        print(
            f"\n--- Step 5: Generating Submission (Accuracy {accuracy} > {THRESHOLD}) ---"
        )
        predictor.generate_submission(load_cached_data=True)
    else:
        print(f"\n--- Skipping Submission (Accuracy {accuracy} <= {THRESHOLD}) ---")


if __name__ == "__main__":
    main()
