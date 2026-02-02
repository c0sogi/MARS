import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import warnings
import logging
from scipy.stats import pearsonr

# Suppress warnings and logs for clean output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Import library modules
from library.config import Config
from library.fine_tuning import train_semantic_model, set_seed
from library.regressor import train_lgbm_regressor, reconstruct_orders
from library.feature_engineering import generate_features_pipeline
from library.metrics import kendall_tau_metric
from library.inference import generate_submission_file
from library.data_loader import get_data_splits


def main():
    # 1. Configuration Overrides for Fast Baseline
    # We override default config values to ensure execution completes within the time limit.
    print("Configuring fast baseline parameters...")
    Config.NUM_FINE_TUNE_NOTEBOOKS = 5000  # Reduced from 40k for speed
    Config.LGBM_NUM_BOOST_ROUND = 500  # Reduced from 2000
    Config.LGBM_EARLY_STOPPING_ROUNDS = 50
    Config.DEBUG = False  # Ensure we run on the defined subset, not debug mode

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    # 2. Stage 1: Contrastive Fine-Tuning
    print("\n=== Stage 1: Fine-Tuning Backbone ===")
    # This will check for cached model or train a new one if needed
    train_semantic_model(load_cached_data=True)

    # 3. Stage 2: Feature Engineering & Regression Training
    print("\n=== Stage 2: Training Regressor ===")
    # This generates features (cached) and trains the LightGBM model
    train_lgbm_regressor(load_cached_data=True)

    # 4. Stage 3: Validation & Failure Analysis
    print("\n=== Stage 3: Validation & Failure Analysis ===")

    # Load validation metadata and features
    _, df_val_meta, _ = get_data_splits()

    # We reload features to ensure we have access to them for analysis
    # (train_lgbm_regressor saves them to disk)
    df_val_feats = generate_features_pipeline(
        df_val_meta, mode="val", load_cached_data=True
    )

    # Load the trained model
    if not os.path.exists(Config.LGBM_MODEL_PATH):
        raise FileNotFoundError("Trained model not found.")

    bst = lgb.Booster(model_file=Config.LGBM_MODEL_PATH)

    # Prepare features for prediction
    feature_cols = [
        c for c in df_val_feats.columns if c not in ["id", "cell_id", "target"]
    ]
    X_val = df_val_feats[feature_cols].values
    y_val = df_val_feats["target"].values

    # Predict
    val_preds = bst.predict(X_val)
    df_val_feats["pred"] = val_preds

    # --- Failure Analysis ---
    # Calculate absolute error for each prediction
    df_val_feats["error"] = np.abs(df_val_feats["target"] - df_val_feats["pred"])

    print("\nFailure Analysis (Correlation of Error with Features):")
    correlations = {}
    for col in feature_cols:
        # Calculate Pearson correlation
        corr, _ = pearsonr(df_val_feats[col], df_val_feats["error"])
        correlations[col] = corr

    # Sort and print top correlations
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corrs[:5]:
        print(f"  {feat}: {corr:.4f}")

    # --- Metric Calculation ---
    # Reconstruct full orders
    df_val_pred_orders = reconstruct_orders(df_val_feats, df_val_meta, mode="val")

    # Compute Kendall Tau
    final_metric = kendall_tau_metric(df_val_pred_orders, df_val_meta)

    # PRINT FINAL METRIC (Required Format)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Stage 4: Submission
    # Threshold check
    THRESHOLD = 0.8061

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission_file(load_cached_data=True)

        if os.path.exists(Config.SUBMISSION_PATH):
            print(f"Submission saved to {Config.SUBMISSION_PATH}")
        else:
            print("Error: Submission file was not created.")
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
