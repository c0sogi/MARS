import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

# Import from provided library
from library.runner import run_cv_training, generate_submission
from library.feature_engine import generate_features
from library.config import WORKING_DIR, N_FOLDS, SEED
from library.utils import set_seed


def main():
    # Ensure reproducibility
    set_seed(SEED)

    print("==================================================")
    print("Step 1: Training Feature-Augmented Bagged Ensemble")
    print("==================================================")
    # Run the 5-fold CV training pipeline
    # debug=False ensures we use the full training set for maximum performance
    # load_cached_features=True allows using pre-computed features if available
    cv_mean_auc = run_cv_training(debug=False, load_cached_features=True)
    print(f"Cross-Validation Mean AUC: {cv_mean_auc}")

    print("\n==================================================")
    print("Step 2: Hold-out Validation Inference")
    print("==================================================")
    # Load the hold-out validation dataset
    # This generates/loads features for the validation split defined in metadata/val.csv
    df_val = generate_features("val", load_cached_data=True, debug=False)

    # Identify columns
    target_col = "requester_received_pizza"
    meta_cols = ["request_id", target_col]
    # Feature columns are all columns except ID and target
    feature_cols = [c for c in df_val.columns if c not in meta_cols]

    # Initialize array for aggregated predictions
    final_preds = np.zeros(len(df_val))
    fold_dir = os.path.join(WORKING_DIR, "models")

    # Perform inference using the ensemble of 5 models
    print(f"Aggregating predictions across {N_FOLDS} folds...")
    for fold in range(N_FOLDS):
        # Load artifacts for this fold
        scaler_path = os.path.join(fold_dir, f"scaler_fold_{fold}.joblib")
        model_path = os.path.join(fold_dir, f"model_fold_{fold}.joblib")

        if not os.path.exists(scaler_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifacts for fold {fold} not found.")

        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)

        # Transform validation data using the fold-specific scaler (RankGauss)
        # The ScalerWrapper handles column selection internally
        df_val_scaled = scaler.transform(df_val)
        X_val_fold = df_val_scaled[feature_cols]

        # Predict probabilities (class 1)
        # Sklearn models run on CPU, but are highly optimized
        preds = model.predict_proba(X_val_fold)[:, 1]
        final_preds += preds

    # Average predictions
    avg_preds = final_preds / N_FOLDS
    y_val = df_val[target_col].values

    # Compute and print the required metric
    val_auc = roc_auc_score(y_val, avg_preds)
    print(f"Final Validation Metric: {val_auc}")

    print("\n==================================================")
    print("Step 3: Failure Analysis")
    print("==================================================")
    # Calculate error magnitude
    errors = np.abs(y_val - avg_preds)

    # Calculate correlation between error and features
    numeric_cols = df_val.select_dtypes(include=[np.number]).columns
    correlations = {}

    for col in numeric_cols:
        if col == target_col:
            continue

        # Get feature values, filling NaNs with 0 for correlation calculation
        feat_values = df_val[col].fillna(0).values

        # Compute correlation
        # Check for constant columns to avoid division by zero
        if np.std(feat_values) > 1e-9 and np.std(errors) > 1e-9:
            corr = np.corrcoef(feat_values, errors)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr

    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features associated with Prediction Error:")
    for feat, corr in sorted_corrs[:10]:
        print(f"  {feat}: {corr:.6f}")

    print("\n==================================================")
    print("Step 4: Submission Generation")
    print("==================================================")
    threshold = 0.7141749705260098

    if val_auc > threshold:
        print(f"Validation AUC ({val_auc}) exceeds threshold ({threshold}).")
        print("Generating submission file...")
        generate_submission(debug=False, load_cached_features=True)
        print("Submission generated successfully.")
    else:
        print(f"Validation AUC ({val_auc}) does not exceed threshold ({threshold}).")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
