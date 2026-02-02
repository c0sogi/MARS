import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Import library components
from library.config import SUBMISSION_DIR, TRAIN_CONFIG, RANDOM_SEED
from library.feature_pipeline import generate_features, clean_features
from library.model import train_model, predict_model
from library.utils import log_transform, inverse_log_transform

# Set random seeds
np.random.seed(RANDOM_SEED)


def calculate_column_rmsle(y_true, y_pred_log):
    """
    Calculates RMSLE given true values (original scale) and predicted values (log scale).
    RMSLE = sqrt(mean((log(1+y_true) - y_pred_log)^2))
    """
    y_true_log = log_transform(y_true)
    return np.sqrt(mean_squared_error(y_true_log, y_pred_log))


def main():
    print("Starting pipeline execution...")

    # -------------------------------------------------------------------------
    # 1. Feature Generation
    # -------------------------------------------------------------------------
    # Generate features for Train, Validation, and Test sets.
    # The pipeline handles caching to avoid re-computation.
    # We use the full dataset (debug=False) as the size (~2000) is small enough for <2h runtime.
    train_df = generate_features(split="train", load_cached_data=True, debug=False)
    val_df = generate_features(split="val", load_cached_data=True, debug=False)
    test_df = generate_features(split="test", load_cached_data=True, debug=False)

    # -------------------------------------------------------------------------
    # 2. Feature Cleaning
    # -------------------------------------------------------------------------
    # Remove constant columns based on training data statistics
    train_df, val_df, test_df = clean_features(train_df, val_df, test_df)

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    # Train XGBoost models for each target
    models = train_model(train_df, val_df)

    # -------------------------------------------------------------------------
    # 4. Validation & Evaluation
    # -------------------------------------------------------------------------
    print("\n--- Validation Assessment ---")
    targets = TRAIN_CONFIG["target_cols"]

    # Identify feature columns (same logic as in model.py)
    exclude_cols = ["id", "file_path"] + targets
    feature_cols = [c for c in val_df.columns if c not in exclude_cols]
    X_val = val_df[feature_cols]

    rmsle_scores = []
    val_predictions_log = {}

    for target in targets:
        model = models[target]
        y_true = val_df[target]

        # Predict in log space (model output)
        y_pred_log = model.predict(X_val)
        val_predictions_log[target] = y_pred_log

        # Calculate RMSLE for this column
        score = calculate_column_rmsle(y_true, y_pred_log)
        rmsle_scores.append(score)
        print(f"Target: {target}, RMSLE: {score:.6f}")

    # Calculate final metric (mean of column-wise RMSLEs)
    final_metric = np.mean(rmsle_scores)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Failure Analysis ---")
    # We analyze the correlation between the error magnitude (in log space) and features
    # to see which features are associated with high errors.

    analysis_df = X_val.copy()

    for target in targets:
        y_true_log = log_transform(val_df[target])
        y_pred_log = val_predictions_log[target]

        # Calculate absolute error in log space (relative error proxy)
        error = np.abs(y_true_log - y_pred_log)
        analysis_df[f"error_{target}"] = error

        # Compute correlations
        correlations = analysis_df.corrwith(analysis_df[f"error_{target}"])
        # Drop the error column itself from correlations
        correlations = correlations.drop(f"error_{target}")

        print(f"\nTop feature correlations with error for {target}:")
        print(correlations.abs().sort_values(ascending=False).head(5))

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.05500532306811823

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric {final_metric} < {THRESHOLD}. Generating submission..."
        )

        # Generate predictions for test set
        # predict_model handles inverse transformation internally
        submission_df = predict_model(models, test_df)

        # Ensure output directory exists
        os.makedirs(SUBMISSION_DIR, exist_ok=True)

        # Save submission
        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

        # Preview
        print(submission_df.head())
    else:
        print(
            f"\nValidation metric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
