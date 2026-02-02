import os
import shutil
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

    # Invalidate Stale Caches When Switching Execution Modes (Cite debug_lesson_4)
    if os.path.exists(Config.WORKING_DIR):
        print(
            f"Clearing working directory {Config.WORKING_DIR} to prevent stale cache usage."
        )
        shutil.rmtree(Config.WORKING_DIR)

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

    # Identify indices in the loaded data that correspond to the validation set
    val_meta_path = Config.VAL_METADATA_PATH
    df_val_meta = pd.read_csv(val_meta_path)

    loaded_ids = data["train_ids"]
    val_ids_set = set(df_val_meta["segment_id"].values)
    val_indices = [i for i, seg_id in enumerate(loaded_ids) if seg_id in val_ids_set]

    # Enforce Strict Data Isolation When Training Stacking Meta-Learners (Cite debug_lesson_3)
    if len(val_indices) == 0:
        print("Warning: No validation samples found. Training on all data.")
        stacker = RidgeStacker()
        stacker.fit(X_stack, y_all)
        preds_val = stacker.predict(X_stack)
        y_val = y_all
        X_tab_val = data["X_tab"]
    else:
        # Split data into meta-train and meta-val
        all_indices = np.arange(len(y_all))
        train_indices = np.setdiff1d(all_indices, val_indices)

        stacker = RidgeStacker()
        # Train ONLY on training set OOFs
        stacker.fit(X_stack[train_indices], y_all[train_indices])

        # Predict on Validation set OOFs (unseen by meta-learner)
        preds_val = stacker.predict(X_stack[val_indices])
        y_val = y_all[val_indices]
        X_tab_val = data["X_tab"].iloc[val_indices]

    # ---------------------------------------------------------
    # 5. Validation Metric (Hold-Out Set)
    # ---------------------------------------------------------

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
        # Refit on all data for final submission
        stacker.fit(X_stack, y_all)
        final_preds = stacker.predict(X_test_stack)
        save_submission(data["test_ids"], final_preds)
    else:
        print(
            f"\nMetric {final_mae} >= {THRESHOLD}. Threshold not met. Submission skipped."
        )


if __name__ == "__main__":
    main()
