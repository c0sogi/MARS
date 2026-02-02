import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Import provided library modules
import library.config as config
import library.data_processor as data_processor
import library.model_trainer as model_trainer


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(config.SEED)
    print("Starting pipeline execution...")

    # 2. Data Loading
    # Loads features for Train, Validation, and Test sets.
    # Uses caching (load_cached_data=True) to speed up execution.
    print("Loading datasets...")
    train_df, val_df, test_df = data_processor.load_data(load_cached_data=True)

    print(
        f"Data loaded. Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}"
    )

    # 3. Model Training
    # Trains the LightGBM ensemble using Stratified K-Fold CV on the training set.
    print("Training LightGBM ensemble...")
    models, oof_preds, cv_scores = model_trainer.train_model_cv(train_df)

    # 4. Validation Evaluation
    # Evaluate the ensemble on the unseen hold-out validation set.
    print("Evaluating on hold-out validation set...")

    # Prepare validation features
    feature_cols = model_trainer.get_feature_columns(val_df)
    X_val = val_df[feature_cols]
    y_val = val_df["time_to_eruption"]

    # Generate predictions from each model in the ensemble
    val_preds_list = []
    for model in models:
        # LightGBM predict (CPU inference is efficient for this scale)
        preds = model.predict(X_val)
        val_preds_list.append(preds)

    # Average predictions (Ensemble)
    avg_val_preds = np.mean(val_preds_list, axis=0)

    # Ensure non-negative predictions (time cannot be negative)
    avg_val_preds = np.maximum(avg_val_preds, 0)

    # Calculate Final Metric (MAE)
    final_metric = mean_absolute_error(y_val, avg_val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nPerforming failure analysis...")
    # Create analysis dataframe
    analysis_df = val_df.copy()
    analysis_df["prediction"] = avg_val_preds
    analysis_df["abs_error"] = np.abs(
        analysis_df["time_to_eruption"] - analysis_df["prediction"]
    )

    # Calculate correlation between features and absolute error
    # We filter for numeric columns to ensure correlation calculation works
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
    correlations = (
        analysis_df[numeric_cols]
        .corrwith(analysis_df["abs_error"])
        .abs()
        .sort_values(ascending=False)
    )

    print(
        "Top 5 features correlated with prediction error (excluding target/error itself):"
    )
    # Filter out the error columns themselves to see which input features drive error
    feature_corrs = correlations.drop(
        labels=["abs_error", "time_to_eruption", "prediction"], errors="ignore"
    )
    print(feature_corrs.head(5))

    # 6. Submission Generation
    # Threshold defined in task requirements
    THRESHOLD = 2617304.0647319085

    if final_metric < THRESHOLD:
        print(f"\nValidation metric ({final_metric}) meets threshold ({THRESHOLD}).")
        print("Generating submission file...")
        model_trainer.generate_submission(models, test_df)
    else:
        print(
            f"\nValidation metric ({final_metric}) does NOT meet threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
