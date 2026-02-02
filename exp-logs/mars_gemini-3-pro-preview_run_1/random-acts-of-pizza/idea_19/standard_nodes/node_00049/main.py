import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from library.config import RANDOM_SEED, ENSEMBLE_WEIGHTS, SUBMISSION_PATH, VAL_PATH
from library.utils import seed_everything, save_submission
from library.data_loader import load_data
from library.features import FeatureEngineer
from library.engine import train_rf, train_mlp, evaluate_preds


def run():
    # 1. Setup
    seed_everything(RANDOM_SEED)
    print("Initializing Pipeline...")

    # 2. Load Data
    # The dataset size is small enough (~2k train) that we process the full dataset
    # while maintaining a fast execution time.
    train_df, val_df, test_df = load_data()

    # 3. Feature Engineering
    # Initialize engineer and process features
    # load_cached=True ensures we use pre-computed features if available to save time
    fe = FeatureEngineer()
    data_rf, data_mlp = fe.process_features(train_df, val_df, test_df, load_cached=True)

    # 4. Train Models
    # Stream A: Latent-Aligned Random Forest
    rf_preds, rf_model = train_rf(data_rf)

    # Stream B: Credibility-Gated MLP
    mlp_preds, mlp_model = train_mlp(data_mlp)

    # 5. Ensemble & Validation
    print("\n=== Ensembling ===")
    w_rf = ENSEMBLE_WEIGHTS["rf"]
    w_mlp = ENSEMBLE_WEIGHTS["mlp"]

    # Calculate Ensemble Predictions for Validation
    val_preds_ensemble = (w_rf * rf_preds["val"]) + (w_mlp * mlp_preds["val"])
    y_val = data_rf["y_val"]  # Targets are consistent across streams

    # Compute Metric
    final_val_auc = evaluate_preds(y_val, val_preds_ensemble)
    print(f"Final Validation Metric: {final_val_auc}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Create an analysis dataframe based on the validation set
    analysis_df = val_df.copy()
    analysis_df["pred"] = val_preds_ensemble
    analysis_df["target"] = y_val

    # Calculate absolute error (residual)
    analysis_df["error"] = np.abs(analysis_df["target"] - analysis_df["pred"])

    # Identify numerical columns for correlation analysis
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_exclude = ["requester_received_pizza", "pred", "target", "error"]
    feature_cols = [c for c in numeric_cols if c not in cols_to_exclude]

    correlations = {}
    for col in feature_cols:
        # Skip columns that are entirely NaN or constant
        if analysis_df[col].isnull().all() or analysis_df[col].nunique() <= 1:
            continue

        # Calculate correlation between feature value and prediction error
        # We fill NaNs with median for this specific correlation check to avoid dropping rows
        col_data = analysis_df[col].fillna(analysis_df[col].median())
        corr = col_data.corr(analysis_df["error"])
        correlations[col] = corr

    # Sort by absolute correlation to find strongest associations with error
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error (Failure Analysis):")
    for name, val in sorted_corr[:5]:
        print(f"  {name}: {val:.4f}")

    # 7. Submission
    threshold = 0.6959737721862433
    if final_val_auc > threshold:
        print(
            f"\nValidation AUC ({final_val_auc}) > Threshold ({threshold}). Generating submission..."
        )

        # Calculate Ensemble Predictions for Test
        test_preds_ensemble = (w_rf * rf_preds["test"]) + (w_mlp * mlp_preds["test"])

        # Get Request IDs
        request_ids = test_df["request_id"].values

        # Save
        save_submission(request_ids, test_preds_ensemble)
    else:
        print(
            f"\nValidation AUC ({final_val_auc}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
