import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.utils import set_seed, save_submission
from library.feature_engineering import prepare_data
from library.rf_pipeline import train_rf_model, predict_rf
from library.mlp_pipeline import train_mlp_model, predict_mlp


def run():
    # 1. Setup and Reproducibility
    print("Initializing orchestration...")
    set_seed(Config.RANDOM_SEED)

    # 2. Data Loading & Preparation
    # load_cached_data=True ensures we use pre-computed features if available
    print("Loading and preparing data...")
    train_data, val_data, test_data = prepare_data(load_cached_data=True)

    # 3. Stream A: Random Forest Training
    print("\n--- Training Stream A: Random Forest ---")
    # Using default hyperparameters from Config via the function defaults
    rf_model = train_rf_model(
        X_train=train_data["rf_features"],
        y_train=train_data["y"],
        X_val=val_data["rf_features"],
        y_val=val_data["y"],
    )

    # 4. Stream B: MLP Training
    print("\n--- Training Stream B: Credibility-Gated MLP ---")
    # Using default hyperparameters from Config via the function defaults
    mlp_model = train_mlp_model(
        train_features=train_data["mlp_features"],
        y_train=train_data["y"],
        val_features=val_data["mlp_features"],
        y_val=val_data["y"],
    )

    # 5. Validation & Ensemble
    print("\n--- Validating Ensemble ---")
    # Get predictions
    rf_val_probs = predict_rf(rf_model, val_data["rf_features"])
    mlp_val_probs = predict_mlp(mlp_model, val_data["mlp_features"])

    # Simple Weighted Average Ensemble (0.5 / 0.5)
    ensemble_val_probs = 0.5 * rf_val_probs + 0.5 * mlp_val_probs

    # Calculate Metric
    val_auc = roc_auc_score(val_data["y"], ensemble_val_probs)
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(val_data["y"] - ensemble_val_probs)

    # Correlate errors with metadata features (MLP metadata is a good proxy for numerical features)
    # We use the metadata from the MLP features dict
    meta_features = val_data["mlp_features"]["metadata"]
    n_features = meta_features.shape[1]

    correlations = []
    for i in range(n_features):
        feat_col = meta_features[:, i]
        # Handle constant features to avoid warnings
        if np.std(feat_col) == 0:
            corr = 0
        else:
            corr, _ = pearsonr(errors, feat_col)
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Metadata Features correlated with Error:")
    for idx, corr in correlations[:5]:
        print(f"Feature Index {idx}: Correlation = {corr:.4f}")

    # 7. Submission
    threshold = 0.6959737721862433
    if val_auc > threshold:
        print(
            f"\nValidation metric ({val_auc}) meets threshold ({threshold}). Generating submission..."
        )

        # Inference on Test Set
        rf_test_probs = predict_rf(rf_model, test_data["rf_features"])
        mlp_test_probs = predict_mlp(mlp_model, test_data["mlp_features"])

        # Ensemble
        ensemble_test_probs = 0.5 * rf_test_probs + 0.5 * mlp_test_probs

        # Save
        save_submission(test_data["ids"], ensemble_test_probs)
    else:
        print(
            f"\nValidation metric ({val_auc}) did NOT meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
