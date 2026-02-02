import os
import numpy as np
import pandas as pd
import torch

from library.config import FEATURE_COLS, SUBMISSION_DIR, SAMPLE_SUBMISSION_PATH, SEED
from library.utils import seed_everything, optimize_threshold
from library.feature_engineering import FeatureEngineer
from library.trainer import Trainer


def main():
    # 1. Initialization
    seed_everything(SEED)

    # 2. Data Preparation
    print("Initializing Feature Engineer...")
    fe = FeatureEngineer()

    # Load Training Data
    print("Loading training data...")
    X_train, y_train, _ = fe.process_train(load_cached_data=True)

    # Subsample training data for speed (Fast Baseline)
    # Limit to 1M samples if larger to ensure quick turnaround
    MAX_TRAIN_SAMPLES = 1_000_000
    if len(X_train) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling training data from {len(X_train)} to {MAX_TRAIN_SAMPLES}..."
        )
        indices = np.random.choice(len(X_train), MAX_TRAIN_SAMPLES, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]

    # Load Validation Data
    print("Loading validation data...")
    X_val, y_val, _ = fe.process_val(load_cached_data=True)

    # 3. Training
    print("Initializing Trainer...")
    trainer = Trainer()

    print("Starting training loop...")
    trainer.fit(X_train, y_train, X_val, y_val)

    # 4. Validation & Threshold Optimization
    print("Performing final validation inference...")
    # Predict on validation set using the best model saved during fit
    val_probs = trainer.predict(X_val)

    # Optimize threshold
    best_thresh, final_mcc = optimize_threshold(y_val, val_probs)

    # REQUIRED: Print Final Validation Metric
    print(f"Final Validation Metric: {final_mcc}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(y_val - val_probs)

    # Extract features from the middle frame of the window for correlation analysis
    # X_val shape: (N_samples, Window_Size, N_features)
    mid_idx = X_val.shape[1] // 2
    X_val_mid = X_val[:, mid_idx, :]

    # Create DataFrame for correlation
    analysis_df = pd.DataFrame(X_val_mid, columns=FEATURE_COLS)
    analysis_df["error"] = errors

    # Compute correlation
    corr = (
        analysis_df.corr()["error"].drop("error").sort_values(ascending=False, key=abs)
    )
    print("Correlation between Error and Input Features (Top 5):")
    print(corr.head(5))

    # 6. Submission
    TARGET_SCORE = 0.62458462731896

    if final_mcc > TARGET_SCORE:
        print(
            f"\nValidation MCC ({final_mcc}) > Target ({TARGET_SCORE}). Generating submission..."
        )

        # Load Test Data
        X_test, _, test_ids = fe.process_test(load_cached_data=True)

        # Predict
        test_probs = trainer.predict(X_test)

        # Apply optimized threshold
        test_preds = (test_probs >= best_thresh).astype(int)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"contact_id": test_ids, "contact": test_preds})

        # Ensure submission directory exists
        os.makedirs(SUBMISSION_DIR, exist_ok=True)

        # Save
        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nValidation MCC ({final_mcc}) <= Target ({TARGET_SCORE}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
