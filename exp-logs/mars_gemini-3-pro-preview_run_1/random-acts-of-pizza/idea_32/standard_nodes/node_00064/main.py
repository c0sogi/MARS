import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings

# Ensure the current directory is in the python path
sys.path.append(os.getcwd())

from library.config import Config
from library.data_utils import seed_everything, load_dataset, save_submission
from library.feature_engineering import FeatureEngineer
from library.training import train_rf_model, train_mlp_model, predict_ensemble

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_pipeline():
    # 1. Initialization
    print("Initializing pipeline...")
    seed_everything(Config.RANDOM_SEED)

    # 2. Feature Engineering & Data Loading
    # The FeatureEngineer handles loading raw data, caching, and generating specific feature sets
    # for both RF (sparse) and MLP (dense/embeddings).
    engineer = FeatureEngineer()
    train_data, val_data, test_data = engineer.process_data(load_cached_data=True)

    # 3. Model Training
    # Train Random Forest (Stream A)
    print("\n--- Training Random Forest ---")
    rf_model = train_rf_model(train_data)

    # Train MLP (Stream B)
    print("\n--- Training Dual-Query Gated MLP ---")
    mlp_model = train_mlp_model(train_data, val_data, device=Config.DEVICE)

    # 4. Validation
    print("\n--- Running Validation ---")
    # Generate ensemble predictions on validation set
    val_preds = predict_ensemble(rf_model, mlp_model, val_data, device=Config.DEVICE)
    val_targets = val_data["y"]

    # Calculate Metric
    val_auc = roc_auc_score(val_targets, val_preds)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Load raw validation data to get interpretable column names
    _, df_val_raw, _ = load_dataset(debug=Config.DEBUG)

    # Calculate error magnitude
    errors = np.abs(val_targets - val_preds)

    # Correlate errors with numerical features
    # We use the raw numerical columns defined in Config
    analysis_cols = Config.NUMERIC_COLS
    correlations = {}

    for col in analysis_cols:
        if col in df_val_raw.columns:
            # Handle NaNs if any (though FeatureEngineer imputes, raw df might have them)
            feat_values = df_val_raw[col].fillna(0).values
            # Ensure lengths match (debug mode might cause mismatch if not handled carefully,
            # but here we loaded consistent sets)
            if len(feat_values) == len(errors):
                corr = np.corrcoef(feat_values, errors)[0, 1]
                if not np.isnan(corr):
                    correlations[col] = corr

    # Sort and print top correlations
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Top correlations between Error and Features:")
    for name, val in sorted_corrs[:5]:
        print(f"{name}: {val:.4f}")

    # 6. Submission
    THRESHOLD = 0.7056961514236341

    if val_auc > THRESHOLD:
        print(f"\nValidation metric {val_auc} > {THRESHOLD}. Generating submission...")

        # Generate Test Predictions
        test_preds = predict_ensemble(
            rf_model, mlp_model, test_data, device=Config.DEVICE
        )

        # Load raw test dataframe to get request_ids
        _, _, df_test_raw = load_dataset(debug=Config.DEBUG)

        # Save submission
        save_submission(test_preds, df_test_raw)

    else:
        print(f"\nValidation metric {val_auc} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    run_pipeline()
