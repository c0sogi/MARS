import sys
import os
import pandas as pd
import numpy as np
import gc

# Ensure the current directory is in the path to import local modules
sys.path.append(".")

from library.config import Config
from library.utils import set_seed, compute_mcc
from library.feature_engineering import FeatureEngineer
from library.model_engine import DualStreamGBDT


def main():
    # 1. Initialization
    set_seed(Config.SEED)

    # 2. Feature Engineering
    print("Initializing Feature Engineer...")
    fe = FeatureEngineer()

    # Load/Generate Train Data
    print("Loading Training Data...")
    (train_X_a, train_y_a, train_ids_a), (train_X_b, train_y_b, train_ids_b) = (
        fe.generate_features("train", load_cached_data=True)
    )

    # Load/Generate Validation Data
    print("Loading Validation Data...")
    (val_X_a, val_y_a, val_ids_a), (val_X_b, val_y_b, val_ids_b) = fe.generate_features(
        "validation", load_cached_data=True
    )

    # 3. Data Preparation
    # Note: We rely on the internal Targeted Majority Undersampling in the engine
    # to handle class imbalance and data volume (Cite solution_lesson_node_00060).
    # No uniform subsampling is applied here.

    # 4. Model Training
    print("Initializing Model Engine...")
    engine = DualStreamGBDT()

    # Pack data for the engine
    train_data_a = (train_X_a, train_y_a, train_ids_a)
    val_data_a = (val_X_a, val_y_a, val_ids_a)
    train_data_b = (train_X_b, train_y_b, train_ids_b)
    val_data_b = (val_X_b, val_y_b, val_ids_b)

    print("Starting Training...")
    engine.train(train_data_a, val_data_a, train_data_b, val_data_b)

    # 5. Global Validation Evaluation
    print("Performing Global Validation...")

    # Predict Stream A Validation
    preds_a = np.array([])
    if len(val_y_a) > 0:
        probs_a = engine.model_a.predict_proba(val_X_a)[:, 1]
        preds_a = (probs_a >= engine.threshold_a).astype(int)

    # Predict Stream B Validation
    preds_b = np.array([])
    if len(val_y_b) > 0:
        probs_b = engine.model_b.predict_proba(val_X_b)[:, 1]
        preds_b = (probs_b >= engine.threshold_b).astype(int)

    # Combine predictions and targets
    y_true_all = np.concatenate([val_y_a, val_y_b])
    y_pred_all = np.concatenate([preds_a, preds_b])

    final_metric = compute_mcc(y_true_all, y_pred_all)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Analyze Stream A (Player-Player)
    if len(val_y_a) > 0:
        print("Stream A (Player-Player) Error Analysis:")
        errors_a = np.abs(val_y_a - preds_a)
        # Sample for correlation calculation if dataset is large
        sample_size = min(50000, len(val_X_a))
        idx = np.random.choice(len(val_X_a), sample_size, replace=False)

        # Calculate correlation between features and error magnitude
        corrs_a = val_X_a.iloc[idx].corrwith(
            pd.Series(errors_a[idx], index=val_X_a.index[idx])
        )
        print("Top 5 Features correlated with Error:")
        print(corrs_a.abs().sort_values(ascending=False).head(5))

    # Analyze Stream B (Player-Ground)
    if len(val_y_b) > 0:
        print("\nStream B (Player-Ground) Error Analysis:")
        errors_b = np.abs(val_y_b - preds_b)
        sample_size = min(50000, len(val_X_b))
        idx = np.random.choice(len(val_X_b), sample_size, replace=False)

        corrs_b = val_X_b.iloc[idx].corrwith(
            pd.Series(errors_b[idx], index=val_X_b.index[idx])
        )
        print("Top 5 Features correlated with Error:")
        print(corrs_b.abs().sort_values(ascending=False).head(5))

    # 7. Submission Generation
    THRESHOLD = 0.6968
    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        # Load Test Data
        print("Loading Test Data...")
        (test_X_a, test_y_a, test_ids_a), (test_X_b, test_y_b, test_ids_b) = (
            fe.generate_features("test", load_cached_data=True)
        )

        test_data_a = (test_X_a, test_y_a, test_ids_a)
        test_data_b = (test_X_b, test_y_b, test_ids_b)

        engine.generate_submission(test_data_a, test_data_b)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
