import os
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# Import provided library modules
from library.config import PATHS, MODEL_PARAMS
from library.data_loader import build_dataset, load_metadata
from library.model_handler import train_model, predict_model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_training():
    """
    Orchestrates the training of a 5-fold LightGBM ensemble.
    """
    print("Loading Training Data...")
    # Load the main training data (source for CV)
    X_train_full, y_train_full = build_dataset("train", load_cached_data=True)

    print("Loading Hold-out Validation Data...")
    # Load the fixed hold-out validation set
    X_holdout, y_holdout = build_dataset("val", load_cached_data=True)

    # Initialize K-Fold
    kf = KFold(
        n_splits=MODEL_PARAMS.N_FOLDS, shuffle=True, random_state=MODEL_PARAMS.SEED
    )

    models = []
    fold_metrics = []

    print(f"Starting {MODEL_PARAMS.N_FOLDS}-Fold Cross-Validation...")

    # Iterate through folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full, y_train_full)):
        print(f"\n--- Fold {fold + 1} ---")

        # Split data for this fold
        X_fold_train = X_train_full.iloc[train_idx]
        y_fold_train = y_train_full.iloc[train_idx]
        X_fold_val = X_train_full.iloc[val_idx]
        y_fold_val = y_train_full.iloc[val_idx]

        # Train model using the provided handler
        model = train_model(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
        models.append(model)

        # Evaluate on fold validation set
        preds = predict_model(model, X_fold_val)
        mae = mean_absolute_error(y_fold_val, preds)
        fold_metrics.append(mae)
        print(f"Fold {fold + 1} MAE: {mae}")

    print(f"\nAverage Fold MAE: {np.mean(fold_metrics)}")

    return models, X_holdout, y_holdout


def run_inference(models, X_holdout, y_holdout):
    """
    Evaluates the ensemble on the hold-out set and performs failure analysis.
    """
    print("\nEvaluating Ensemble on Hold-out Validation Set...")

    # Generate predictions from all models (Bagging)
    ensemble_preds = np.zeros(len(X_holdout))

    for model in models:
        preds = predict_model(model, X_holdout)
        ensemble_preds += preds

    # Average predictions
    ensemble_preds /= len(models)

    # Compute Final Metric
    final_mae = mean_absolute_error(y_holdout, ensemble_preds)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_mae}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude
    errors = np.abs(y_holdout - ensemble_preds)

    # Calculate correlation between features and error magnitude
    correlations = {}
    # Convert to numeric just in case, though features should be float
    X_numeric = X_holdout.select_dtypes(include=[np.number])

    for col in X_numeric.columns:
        try:
            # Compute Pearson correlation
            corr = np.corrcoef(X_numeric[col], errors)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr
        except Exception:
            continue

    # Sort by absolute correlation strength
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude (Systematic Error Patterns):")
    for feat, corr in sorted_corr[:5]:
        print(f"{feat}: {corr:.4f}")

    return final_mae


def generate_submission(models, threshold_mae):
    """
    Generates submission file if validation metric meets the threshold.
    """
    THRESHOLD = 2617304.0647319085

    if threshold_mae >= THRESHOLD:
        print(
            f"\nValidation MAE ({threshold_mae}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )
        return

    print(
        f"\nValidation MAE ({threshold_mae}) passed threshold. Generating submission..."
    )

    # Load Test Data
    print("Loading Test Data...")
    # build_dataset returns (X, None) for test split
    X_test, _ = build_dataset("test", load_cached_data=True)

    # Predict with Ensemble
    ensemble_preds = np.zeros(len(X_test))
    for model in models:
        preds = predict_model(model, X_test)
        ensemble_preds += preds

    ensemble_preds /= len(models)

    # Load Test Metadata to get segment_ids
    test_meta = load_metadata("test")

    # Ensure alignment: build_dataset processes files in order of metadata
    submission_df = pd.DataFrame(
        {"segment_id": test_meta["segment_id"], "time_to_eruption": ensemble_preds}
    )

    # Save Submission
    save_path = PATHS.SUBMISSION_FILE
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")

    # Print head for verification
    print("Submission head:")
    print(submission_df.head())


if __name__ == "__main__":
    # Ensure reproducibility
    np.random.seed(MODEL_PARAMS.SEED)

    # Execute Pipeline
    try:
        models, X_holdout, y_holdout = run_training()
        final_mae = run_inference(models, X_holdout, y_holdout)
        generate_submission(models, final_mae)
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        raise e
