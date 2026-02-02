import os
import sys
import pandas as pd
import numpy as np
import torch

# Import from provided library files
from library.config import Config
from library.utils import set_seed, calculate_roc_auc, save_submission
from library.feature_engineering import FeaturePipeline
from library.model_rf import train_rf_model
from library.model_mlp import train_mlp_model, predict_mlp


def main():
    # 1. Setup
    print("Setting up environment...")
    set_seed(Config.SEED)

    # Optimization for fast baseline execution
    # Modifying Config attributes at runtime to speed up training
    print("Optimizing hyperparameters for fast baseline execution...")
    Config.RF_N_ESTIMATORS = 100  # Reduced from 500
    Config.MLP_EPOCHS = 10  # Reduced from 50
    Config.MLP_PATIENCE = 3  # Reduced from 15

    # 2. Feature Engineering
    print("Executing Feature Engineering Pipeline...")
    pipeline = FeaturePipeline()

    # Load/Generate data
    # Returns 8 items: X_rf_train, X_rf_val, X_rf_test, X_mlp_train, X_mlp_val, X_mlp_test, y_train, y_val
    data = pipeline.process_data(load_cached_data=True)
    (
        X_rf_train,
        X_rf_val,
        X_rf_test,
        X_mlp_train,
        X_mlp_val,
        X_mlp_test,
        y_train,
        y_val,
    ) = data

    # 3. Train Models
    # Stream A: Random Forest
    print("\n--- Training Random Forest Stream ---")
    rf_model = train_rf_model(X_rf_train, y_train, X_rf_val, y_val)

    # Stream B: MLP
    print("\n--- Training MLP Stream ---")
    mlp_model, mlp_trainer = train_mlp_model(X_mlp_train, y_train, X_mlp_val, y_val)

    # 4. Validation & Ensemble
    print("\n--- Performing Ensemble Validation ---")

    # RF Predictions
    rf_val_probs = rf_model.predict_proba(X_rf_val)

    # MLP Predictions
    mlp_val_probs = predict_mlp(mlp_model, X_mlp_val)

    # Ensemble (Weighted Average)
    w_rf, w_mlp = Config.ENSEMBLE_WEIGHTS
    ensemble_val_probs = (w_rf * rf_val_probs) + (w_mlp * mlp_val_probs)

    # Metric Calculation
    val_auc = calculate_roc_auc(y_val, ensemble_val_probs)
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    print("\n--- Performing Failure Analysis ---")
    # Calculate error magnitude
    errors = np.abs(y_val - ensemble_val_probs)

    # Load raw validation data for interpretable features
    try:
        val_df = pd.read_csv(Config.VAL_DATA_PATH)

        # Select numeric columns for correlation
        numeric_cols = val_df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude target if present
        if Config.TARGET_COL in numeric_cols:
            numeric_cols.remove(Config.TARGET_COL)

        # Compute correlations
        correlations = {}
        for col in numeric_cols:
            if val_df[col].nunique() > 1:  # Skip constants
                # Fill NaNs for correlation calculation
                feat_values = val_df[col].fillna(val_df[col].median())
                corr = np.corrcoef(feat_values, errors)[0, 1]
                if not np.isnan(corr):
                    correlations[col] = corr

        # Sort by correlation (positive correlation means higher feature value -> higher error)
        sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

        print(
            "Top 5 Features correlated with Error Magnitude (Positive correlation = High feature value implies high error):"
        )
        for name, score in sorted_corr[:5]:
            print(f"  {name}: {score:.4f}")

    except Exception as e:
        print(f"Failure analysis failed: {e}")

    # 6. Submission
    threshold = 0.7135451153926904
    if val_auc > threshold:
        print(
            f"\nValidation metric ({val_auc}) meets threshold ({threshold}). Generating submission..."
        )

        # Generate Test Predictions
        rf_test_probs = rf_model.predict_proba(X_rf_test)
        mlp_test_probs = predict_mlp(mlp_model, X_mlp_test)

        ensemble_test_probs = (w_rf * rf_test_probs) + (w_mlp * mlp_test_probs)

        # Load Test IDs
        test_df = pd.read_csv(Config.TEST_DATA_PATH)
        request_ids = test_df["request_id"].tolist()

        # Save
        save_submission(request_ids, ensemble_test_probs)
    else:
        print(
            f"\nValidation metric ({val_auc}) does not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
