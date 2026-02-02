import os
import pandas as pd
import numpy as np
import warnings

# Import from provided libraries
from library.config import Config
from library.engine import Engine, RidgeStacker
from library.utils import seed_everything, calc_mae, save_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Enforce reproducibility
    seed_everything(Config.SEED)

    # Configure for Fast Baseline Execution
    # We enable DEBUG mode to limit sample size, ensuring the pipeline completes quickly within the time limit.
    # We set sample size to 1500 to ensure enough data for meaningful validation and failure analysis.
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 1500

    # Reduce training epochs for the Vision Branch to speed up execution
    Config.CNN_TRAIN_PARAMS["epochs"] = 5

    # Reduce estimators for LightGBM for speed
    if hasattr(Config, "LGB_PARAMS"):
        Config.LGB_PARAMS["n_estimators"] = 500
    if hasattr(Config, "LGBM_PARAMS"):
        Config.LGBM_PARAMS["n_estimators"] = 500

    print("Initializing Engine with Fast Baseline Configuration...")
    engine = Engine()

    # ---------------------------------------------------------
    # 2. Data Loading & Alignment
    # ---------------------------------------------------------
    # Loads and aligns Tabular and Vision data for Train, Val, and Test.
    # Note: Engine combines Train and Val for Cross-Validation.
    data = engine.load_aligned_data()

    # ---------------------------------------------------------
    # 3. Model Training (Branches A & B)
    # ---------------------------------------------------------
    # Train Tabular Branch (LightGBM)
    # Returns OOF predictions (aligned with data['y']) and Test predictions
    oof_tab, test_tab = engine.train_tabular_branch(data)

    # Train Vision Branch (EfficientNet)
    oof_vis, test_vis = engine.train_vision_branch(data)

    # ---------------------------------------------------------
    # 4. Meta-Learner Stacking
    # ---------------------------------------------------------
    print("\n--- Training Meta-Learner (Ridge Stacking) ---")
    # Stack OOF predictions from both branches
    X_stack = np.column_stack([oof_tab, oof_vis])
    X_test_stack = np.column_stack([test_tab, test_vis])
    y_all = data["y"]

    # Train Stacker on OOF predictions
    stacker = RidgeStacker()
    stacker.fit(X_stack, y_all)

    # Generate Final OOF Predictions for evaluation
    final_oof = stacker.predict(X_stack)

    # ---------------------------------------------------------
    # 5. Validation Metric (Hold-Out Set)
    # ---------------------------------------------------------
    # The Engine processes data in a combined manner for CV.
    # We explicitly extract the validation subset to report the metric on the hold-out set.

    val_meta_path = Config.VAL_METADATA_PATH
    df_val_meta = pd.read_csv(val_meta_path)

    # Identify indices in the loaded data that correspond to the validation set
    loaded_ids = data["train_ids"]
    val_ids_set = set(df_val_meta["segment_id"].values)

    val_indices = [i for i, seg_id in enumerate(loaded_ids) if seg_id in val_ids_set]

    if len(val_indices) == 0:
        print(
            "Warning: No validation samples found in loaded data. Using full OOF for metrics."
        )
        y_val = y_all
        preds_val = final_oof
        X_tab_val = data["X_tab"]
    else:
        y_val = y_all[val_indices]
        preds_val = final_oof[val_indices]
        X_tab_val = data["X_tab"].iloc[val_indices]

    # Calculate and Print Final Metric
    final_mae = calc_mae(y_val, preds_val)
    print(f"Final Validation Metric: {final_mae}")

    # ---------------------------------------------------------
    # 6. Failure Analysis
    # ---------------------------------------------------------
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude
    errors = np.abs(y_val - preds_val)

    # Correlate errors with input features to identify systematic weaknesses
    df_analysis = X_tab_val.copy()
    df_analysis["error_magnitude"] = errors

    # Compute correlations
    correlations = df_analysis.corr()["error_magnitude"].drop("error_magnitude")
    top_corr = correlations.abs().sort_values(ascending=False).head(5)

    print("Top 5 Feature Correlations with Error Magnitude:")
    for feat, corr_val in top_corr.items():
        # Print original correlation coefficient
        orig_corr = correlations[feat]
        print(f"{feat}: {orig_corr:.4f}")

    # ---------------------------------------------------------
    # 7. Submission
    # ---------------------------------------------------------
    THRESHOLD = 2250276.65

    if final_mae < THRESHOLD:
        print(f"\nMetric {final_mae} < {THRESHOLD}. Generating submission...")
        final_preds = stacker.predict(X_test_stack)
        save_submission(data["test_ids"], final_preds)
    else:
        print(
            f"\nMetric {final_mae} >= {THRESHOLD}. Threshold not met. Submission skipped."
        )


if __name__ == "__main__":
    main()
