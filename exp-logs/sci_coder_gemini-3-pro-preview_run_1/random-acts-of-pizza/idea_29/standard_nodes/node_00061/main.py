import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from library.config import ENSEMBLE_WEIGHTS, NN_PARAMS, RANDOM_SEED
from library.utils import set_seed
from library.data_processing import get_rf_dataset, get_nn_dataset
from library.trainer import train_rf_model, train_nn_model, generate_submission


def main():
    # 1. Setup
    set_seed(RANDOM_SEED)

    # 2. Train Models
    # We use the full dataset (no max_samples limit) because the dataset is small (2.3k rows)
    # and we need maximum performance to pass the threshold.
    # Training time will still be very fast (seconds to minutes).
    print("Starting pipeline execution...")
    rf_model, rf_val_auc = train_rf_model(load_cached_data=True)

    # Using default epochs (50) as defined in config, which is fast for this data size
    nn_model, nn_val_auc = train_nn_model(load_cached_data=True)

    # 3. Ensemble Validation
    print("\n=== Performing Ensemble Validation ===")

    # Load Validation Data for RF (Tabular)
    X_val_rf, y_val = get_rf_dataset(split="val", load_cached_data=True)

    # Load Validation Data for NN (Dataset)
    val_ds_nn = get_nn_dataset(split="val", load_cached_data=True)
    val_loader_nn = DataLoader(
        val_ds_nn, batch_size=NN_PARAMS["batch_size"], shuffle=False, num_workers=0
    )

    # Generate Predictions
    print("Generating validation predictions...")
    rf_probs = rf_model.predict_proba(X_val_rf)
    nn_probs = nn_model.predict_proba(val_loader_nn)

    # Ensure alignment
    if len(rf_probs) != len(nn_probs):
        min_len = min(len(rf_probs), len(nn_probs))
        rf_probs = rf_probs[:min_len]
        nn_probs = nn_probs[:min_len]
        y_val = y_val[:min_len]
        X_val_rf = X_val_rf.iloc[:min_len]

    # Weighted Ensemble
    w_rf = ENSEMBLE_WEIGHTS["rf"]
    w_nn = ENSEMBLE_WEIGHTS["nn"]
    ensemble_probs = (w_rf * rf_probs) + (w_nn * nn_probs)

    # Calculate Metric
    final_auc = roc_auc_score(y_val, ensemble_probs)
    print(f"Final Validation Metric: {final_auc}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude: |y_true - y_pred|
    # If y=1, error is distance from 1. If y=0, error is distance from 0.
    error_magnitude = np.abs(y_val - ensemble_probs)

    # Create a temporary dataframe for correlation analysis
    # We use X_val_rf as it contains interpretable metadata and TF-IDF features
    analysis_df = X_val_rf.copy()
    analysis_df["error_magnitude"] = error_magnitude

    # Compute correlations with error magnitude
    # We drop the error column itself after correlation
    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")

    # Get top 10 features most correlated with error (positive or negative)
    top_corrs = correlations.abs().sort_values(ascending=False).head(10)

    print("Top 10 Features Correlated with Prediction Error:")
    for feature, corr_val in top_corrs.items():
        signed_corr = correlations[feature]
        print(f"{feature}: {signed_corr:.4f}")

    # 5. Submission
    THRESHOLD = 0.7036289345758168

    if final_auc > THRESHOLD:
        print(f"\nValidation metric {final_auc} exceeds threshold {THRESHOLD}.")
        generate_submission(rf_model, nn_model, load_cached_data=True)
    else:
        print(f"\nValidation metric {final_auc} does not exceed threshold {THRESHOLD}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
