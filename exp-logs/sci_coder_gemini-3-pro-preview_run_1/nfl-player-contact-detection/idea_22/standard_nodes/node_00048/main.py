import os
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from library.config import Config
from library.utils import setup_logging, seed_everything
from library.trainer import Trainer
from library.data_loader import DataLoader
from library.models import EnsemblePredictor


def main():
    # 1. Setup Environment
    setup_logging()
    seed_everything(Config.SEED)

    print(
        "Initializing Pipeline Idea 22: Vector-Decomposed Physics Ensemble with Temporal Label Smoothing"
    )

    # 2. Initialize Trainer
    trainer = Trainer()

    # Define sample size for fast baseline execution
    # Using 150,000 samples ensures the pipeline completes well within the 2-hour limit
    # while providing enough data for the ensemble to learn the physics boundaries.
    TRAIN_SAMPLE_SIZE = 150000

    # 3. Phase 1: Train Scouts
    # Scouts (LGBM, XGB) are trained on a balanced subset to learn initial boundaries.
    trainer.train_scouts(sample_size=TRAIN_SAMPLE_SIZE)

    # 4. Phase 2 & 3: Mine Hard Negatives & Train Experts
    # The trainer mines hard negatives using the scouts and trains the expert ensemble
    # with Temporal Label Smoothing on the difficult subset.
    trainer.train_experts(sample_size=TRAIN_SAMPLE_SIZE)

    # 5. Phase 4: Threshold Optimization
    # Optimize the decision threshold on a subset of the validation data for speed.
    trainer.optimize_threshold(sample_size=50000)

    # 6. Phase 5: Full Validation & Failure Analysis
    print("\n=== Running Full Validation & Failure Analysis ===")

    # Load the FULL validation set to compute the official metric
    loader = DataLoader()
    df_val = loader.load_val_data(load_cached_data=True)

    # Load the trained Expert Ensemble
    ensemble = EnsemblePredictor()
    ensemble.load_models(trainer.expert_dir)

    # Load the optimized threshold
    thresh_path = os.path.join(Config.WORKING_DIR, "best_threshold.npy")
    if os.path.exists(thresh_path):
        best_threshold = float(np.load(thresh_path)[0])
    else:
        print("Warning: Threshold file not found. Defaulting to 0.5")
        best_threshold = 0.5

    print(
        f"Inference on Validation Set ({len(df_val)} rows) using Threshold: {best_threshold:.4f}..."
    )

    # Predict probabilities
    # Note: We do not need gradients for inference
    val_probs = ensemble.predict_proba(df_val)

    # Binarize predictions
    val_preds = (val_probs >= best_threshold).astype(int)

    # Compute Final Validation Metric (MCC)
    # Ensure target is integer
    y_true = df_val["contact"].astype(int)
    mcc = matthews_corrcoef(y_true, val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {mcc}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude
    df_val["error"] = np.abs(y_true - val_probs)

    # Define features to analyze for correlation with error
    analysis_features = [
        "distance",
        "speed_p1",
        "speed_p2",
        "acceleration_p1",
        "acceleration_p2",
        "v_radial",
        "v_tangential",
        "a_radial",
        "a_tangential",
        "time_to_collision",
        "radial_acc_energy",
        "orientation_diff",
        "direction_diff",
    ]

    correlations = {}
    for feat in analysis_features:
        if feat in df_val.columns:
            # Drop NaNs for valid correlation calculation
            valid_data = df_val[[feat, "error"]].dropna()
            if not valid_data.empty:
                corr = valid_data[feat].corr(valid_data["error"])
                correlations[feat] = corr

    print("Correlation between Feature and Error Magnitude:")
    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corrs:
        print(f"{feat}: {corr:.4f}")

    # 7. Phase 6: Conditional Submission
    # Generate submission only if metric exceeds the benchmark
    TARGET_METRIC = 0.6865

    if mcc > TARGET_METRIC:
        print(
            f"\nMetric condition met ({mcc:.6f} > {TARGET_METRIC}). Generating submission..."
        )
        trainer.generate_submission()
    else:
        print(
            f"\nMetric condition not met ({mcc:.6f} <= {TARGET_METRIC}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
