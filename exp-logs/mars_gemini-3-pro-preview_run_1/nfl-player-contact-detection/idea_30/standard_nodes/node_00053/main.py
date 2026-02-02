import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library components
from library.config import Config
from library.workflow import Workflow
from library.utils import setup_logger, calc_mcc, seed_everything


def main():
    # 1. Setup & Configuration
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Setup logging
    logger = setup_logger("execution.log")
    print("Initializing OSVA-E Pipeline...")

    # 2. Hardware Optimization (GPU Detection)
    # The prompt requires automatic detection and utilization of GPU.
    if torch.cuda.is_available():
        print("GPU detected. Configuring models for GPU acceleration.")

        # Update LightGBM Parameters (Applied to both LGBMModel and LGBMDartModel)
        Config.LGBM_PARAMS.update(
            {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0}
        )

        # Update XGBoost Parameters
        Config.XGB_PARAMS.update({"tree_method": "gpu_hist", "device": "cuda"})
    else:
        print("No GPU detected. Running on CPU.")

    # 3. Fast Baseline Configuration
    # To ensure the workflow completes within the 2-hour limit, we adjust the number of estimators.
    # The default 2000 is reduced to 600, which is sufficient for the subsampled Expert/Scout datasets.
    Config.LGBM_PARAMS["n_estimators"] = 600
    Config.XGB_PARAMS["n_estimators"] = 600
    # Note: LGBMDartModel inherits LGBM_PARAMS, so it will also use 600 estimators.

    # 4. Initialize Workflow
    workflow = Workflow()

    # 5. Execution Phases

    # --- Phase 1: Train Scouts ---
    # Scouts are trained on a balanced subset to quickly learn the general decision boundary.
    print("\n=== Phase 1: Training Scouts ===")
    scouts = workflow.train_scouts(load_cached_data=True)

    # --- Phase 2: Mine Hard Negatives ---
    # Scouts scan the full training pool to find "Hard Negatives" (False Positives).
    print("\n=== Phase 2: Mining Hard Negatives ===")
    hard_neg_indices = workflow.mine_hard_negatives(scouts, load_cached_data=True)

    # --- Phase 3: Train Experts ---
    # Experts are trained on Positives + Hard Negatives + Random Anchors.
    print("\n=== Phase 3: Training Experts ===")
    experts, threshold = workflow.train_experts(hard_neg_indices, load_cached_data=True)

    # 6. Validation & Failure Analysis
    print("\n=== Phase 4: Validation & Failure Analysis ===")

    # Load the full validation dataset manually to perform detailed analysis
    df_val = workflow.factory.load_and_process_data(split="val", load_cached_data=True)

    # Generate predictions on validation set
    # Note: experts.predict returns probabilities
    val_probs = experts.predict(df_val)

    # Apply optimized threshold
    val_preds = (val_probs >= threshold).astype(int)
    y_val = df_val["contact"].values

    # Calculate MCC
    final_mcc = calc_mcc(y_val, val_preds)

    # Print the required metric string with full precision
    print(f"Final Validation Metric: {final_mcc}")

    # --- Failure Analysis ---
    # Calculate error magnitude (Absolute difference between label and probability)
    errors = np.abs(y_val - val_probs)

    # Identify feature columns used by the model
    feature_cols = experts.feature_cols

    print("\nFailure Analysis: Correlation between Features and Error Magnitude")
    correlations = {}

    # Compute correlation for each numerical feature
    for col in feature_cols:
        if col in df_val.columns:
            # We use numpy's corrcoef which returns a matrix
            try:
                corr = np.corrcoef(df_val[col].values, errors)[0, 1]
                if not np.isnan(corr):
                    correlations[col] = corr
            except Exception:
                continue

    # Sort by absolute correlation strength
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features associated with Error:")
    for name, val in sorted_corr[:10]:
        print(f"  {name}: {val:.4f}")

    # 7. Submission Generation
    # Generate submission only if the metric exceeds the specified threshold.
    TARGET_SCORE = 0.6865

    if final_mcc > TARGET_SCORE:
        print(
            f"\nValidation Score ({final_mcc}) exceeds target ({TARGET_SCORE}). Generating Submission..."
        )
        workflow.predict_test(experts, threshold, load_cached_data=True)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation Score ({final_mcc}) does not meet target ({TARGET_SCORE}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
