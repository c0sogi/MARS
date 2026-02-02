import os
import sys
import numpy as np
import pandas as pd
import torch
import random
from sklearn.metrics import roc_auc_score

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import SEED, TARGET_COL, RAW_NUMERIC_COLS
from library.data_loader import load_dataset
from library.text_processing import generate_sbert_embeddings, generate_tfidf_features
from library.feature_engineering import prepare_rf_features, prepare_mlp_features
from library.trainer import train_rf_model, train_mlp_model, generate_submission

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def main():
    # 1. Load Data
    # Using cached data if available for speed
    train_df, val_df, test_df = load_dataset(load_cached_data=True)

    # 2. Generate Base Features (Text & Embeddings)
    # These functions handle caching internally
    sbert_data = generate_sbert_embeddings(
        train_df, val_df, test_df, load_cached_data=True
    )
    tfidf_data = generate_tfidf_features(
        train_df, val_df, test_df, load_cached_data=True
    )

    # 3. Prepare Model-Specific Features
    # RF Features (TF-IDF, Interactions, etc.)
    rf_features = prepare_rf_features(
        train_df, val_df, test_df, tfidf_data, sbert_data, load_cached_data=True
    )
    # MLP Features (Embeddings, Metadata for FiLM)
    mlp_features = prepare_mlp_features(
        train_df, val_df, test_df, sbert_data, load_cached_data=True
    )

    # 4. Train Models
    # Stream A: Random Forest
    rf_model = train_rf_model(rf_features)

    # Stream B: MLP
    mlp_trainer = train_mlp_model(mlp_features)

    # 5. Validation & Evaluation
    # Get predictions on validation set
    X_val_rf = rf_features["X_val"]
    y_val = rf_features["y_val"]

    rf_val_probs = rf_model.predict_proba(X_val_rf)
    mlp_val_probs = mlp_trainer.predict_proba(mlp_features, split_name="val")

    # Simple Average Ensemble
    ensemble_val_probs = 0.5 * rf_val_probs + 0.5 * mlp_val_probs

    # Compute Metric
    val_auc = roc_auc_score(y_val, ensemble_val_probs)
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("Performing failure analysis...")
    # Calculate absolute error
    errors = np.abs(y_val - ensemble_val_probs)

    # Prepare DataFrame for correlation
    # We use the raw numeric columns from the validation dataframe
    # Ensure we only use columns that exist in val_df
    available_cols = [c for c in RAW_NUMERIC_COLS if c in val_df.columns]

    if available_cols:
        analysis_df = val_df[available_cols].copy()

        # Fill NaNs for correlation calculation
        analysis_df = analysis_df.fillna(analysis_df.median())

        # Add error column
        analysis_df["error_magnitude"] = errors

        # Calculate correlations
        correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")

        # Sort by absolute correlation
        sorted_corrs = correlations.abs().sort_values(ascending=False)

        print("Correlation between Error Magnitude and Input Features:")
        print(sorted_corrs)
    else:
        print("No numeric columns available for failure analysis.")

    # 7. Submission Generation
    # Threshold defined in task
    THRESHOLD = 0.7135451153926904

    if val_auc > THRESHOLD:
        print("Validation metric exceeds threshold. Generating submission...")

        # RF Test Predictions
        X_test_rf = rf_features["X_test"]
        rf_test_probs = rf_model.predict_proba(X_test_rf)

        # MLP Test Predictions
        mlp_test_probs = mlp_trainer.predict_proba(mlp_features, split_name="test")

        # Generate Submission File
        generate_submission(
            test_df, rf_test_probs, mlp_test_probs, ensemble_weights=(0.5, 0.5)
        )
    else:
        print(
            f"Validation metric {val_auc} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
