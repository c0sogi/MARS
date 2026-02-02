import sys
import os
import numpy as np
import pandas as pd
import torch

# Import provided library modules
from library import config, utils, data_loader, train, inference, models


def main():
    # 1. Setup and Configuration Overrides for Fast Baseline
    utils.seed_everything()
    utils.setup_logging()

    print("Configuring for fast baseline execution...")

    # Check for GPU and configure XGBoost/LightGBM accordingly
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("GPU detected. Configuring models to utilize GPU.")
        # Update XGBoost params for GPU acceleration
        config.XGB_PARAMS["device"] = "cuda"
        config.XGB_PARAMS["tree_method"] = "hist"
        # LightGBM GPU support often requires specific compilation.
        # We stick to CPU/Hist for LGBM to ensure stability, as it is generally fast enough.

    # Reduce estimators for speed to meet the "Fast Baseline" requirement
    # The default 2000 is high for a quick run; 500 is sufficient for a strong baseline.
    config.LGBM_PARAMS["n_estimators"] = 500
    config.XGB_PARAMS["n_estimators"] = 500

    # 2. Run Training Pipeline
    # This handles loading data, training scout, mining hard negatives, training expert, and optimizing threshold.
    print("\nStarting Training Pipeline...")
    expert_lgbm, expert_xgb, best_threshold = train.run_training_pipeline(
        load_cached_features=True, load_cached_mining=True
    )

    # 3. Validation & Failure Analysis
    print("\nPerforming Validation and Failure Analysis...")

    # Reload validation data to compute final metrics and analysis
    # We use the DatasetBuilder to ensure consistent preprocessing
    df_val = data_loader.DatasetBuilder().load_data("val", load_cached=True)

    # Generate predictions using the trained ensemble
    ensemble = models.EnsemblePredictor(expert_lgbm, expert_xgb)

    # Predict probabilities
    val_probs = ensemble.predict_proba(df_val)

    # Calculate Final MCC
    y_true = df_val["contact"].values
    final_mcc = utils.calc_mcc(y_true, val_probs, threshold=best_threshold)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_mcc}")

    # Failure Analysis
    # Calculate absolute error magnitude
    df_val["pred_prob"] = val_probs
    df_val["error"] = np.abs(df_val["contact"] - df_val["pred_prob"])

    print("\nFailure Analysis: Correlation of Error with Features")
    # Select numeric columns for correlation
    numeric_cols = df_val.select_dtypes(include=[np.number]).columns
    # Exclude target and helper cols from the correlation check
    exclude_cols = ["contact", "error", "pred_prob", "nfl_player_id_1", "step"]
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    correlations = {}
    for col in feature_cols:
        if df_val[col].std() > 0:  # Avoid constant columns
            correlations[col] = df_val[col].corr(df_val["error"])
        else:
            correlations[col] = 0.0

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features associated with Model Error:")
    for name, corr in sorted_corr[:5]:
        print(f"  {name}: {corr:.4f}")

    # 4. Submission
    # Condition: MCC > 0.6782
    TARGET_METRIC = 0.6782

    if final_mcc > TARGET_METRIC:
        print(
            f"\nValidation Metric ({final_mcc}) exceeds target ({TARGET_METRIC}). Generating submission..."
        )
        inference.create_submission(load_cached_features=True)
    else:
        print(
            f"\nValidation Metric ({final_mcc}) did not exceed target ({TARGET_METRIC}). Skipping submission."
        )


if __name__ == "__main__":
    main()
