import pandas as pd
import numpy as np
import os
import sys

# Import from provided libraries
from library.config import SEED, SUBMISSION_PATH, SUBMISSION_DIR
from library.utils import seed_everything, calculate_mae
from library.data_loader import create_dataset
from library.model_trainer import train_lgbm_fold, generate_predictions


def main():
    # 1. Setup
    seed_everything(SEED)
    print("Initializing pipeline...")

    # 2. Data Loading
    # Load Train, Val, and Test sets using the provided loader which handles caching
    print("Loading training data...")
    train_df = create_dataset("train", load_cached_data=True)

    print("Loading validation data...")
    val_df = create_dataset("val", load_cached_data=True)

    print("Loading test data...")
    test_df = create_dataset("test", load_cached_data=True)

    # 3. Prepare Data for Training
    # Identify feature columns (exclude metadata columns)
    feature_cols = [
        c for c in train_df.columns if c not in ["segment_id", "time_to_eruption"]
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["time_to_eruption"]

    X_val = val_df[feature_cols]
    y_val = val_df["time_to_eruption"]

    print(f"Training on {len(X_train)} samples, Validating on {len(X_val)} samples.")
    print(f"Feature count: {len(feature_cols)}")

    # 4. Model Training
    # Train using the provided fold trainer (LightGBM with Early Stopping)
    model, best_val_score = train_lgbm_fold(X_train, y_train, X_val, y_val)

    # 5. Validation & Metrics
    # Generate predictions on the full validation set
    val_preds = model.predict(X_val)

    # Calculate MAE
    final_mae = calculate_mae(y_val, val_preds)
    # Print strictly formatted metric for evaluation
    print(f"Final Validation Metric: {final_mae}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute errors
    errors = np.abs(y_val - val_preds)

    # Create a DataFrame for correlation analysis
    analysis_df = X_val.copy()
    analysis_df["abs_error"] = errors

    # Calculate correlation of features with absolute error
    correlations = analysis_df.corr()["abs_error"].drop("abs_error")

    # Sort by absolute correlation magnitude
    top_correlations = correlations.abs().sort_values(ascending=False).head(10)

    print("Top 10 features correlated with absolute error:")
    for feature, corr_val in top_correlations.items():
        # Get the actual signed correlation
        sign = correlations[feature]
        print(f"{feature}: {sign:.4f}")

    # 7. Submission Logic
    THRESHOLD = 2617304.0647319085

    if final_mae < THRESHOLD:
        print(
            f"\nValidation metric ({final_mae}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions for test set
        # generate_predictions expects a list of models
        submission_df = generate_predictions([model], test_df)

        # Ensure directory exists
        os.makedirs(SUBMISSION_DIR, exist_ok=True)

        # Save submission
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric ({final_mae}) does NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
