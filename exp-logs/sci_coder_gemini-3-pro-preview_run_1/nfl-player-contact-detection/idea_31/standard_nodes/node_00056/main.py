import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

# Import Library Components
import library.config
from library.config import MODEL_PARAMS, TrainConfig
from library.data import get_data_split
from library.training import run_training_pipeline, get_feature_cols
from library.inference import generate_submission
from library.utils import setup_logging


def main():
    # 1. Setup and Configuration Overrides
    setup_logging()

    print("Configuring parameters for fast baseline execution...")
    # Reduce computational burden for the fast baseline requirement
    # Reduce number of trees
    library.config.MODEL_PARAMS["lgbm"]["n_estimators"] = 600
    library.config.MODEL_PARAMS["xgb"]["n_estimators"] = 600

    # Reduce the ratio of easy negatives (anchors) to speed up Expert training
    library.config.TrainConfig.ANCHOR_RATIO = 0.5

    # 2. Run Training Pipeline
    # This orchestrates: Data Loading -> Hard Negative Mining -> Expert Training -> Threshold Optimization
    print("Starting Training Pipeline...")
    expert_ensemble, best_thresh = run_training_pipeline(load_cached=True)

    # 3. Validation Assessment
    print("Performing Validation Assessment...")
    # Reload validation data to ensure clean evaluation
    df_val = get_data_split("val", load_cached=True)
    feature_cols = get_feature_cols(df_val)

    X_val = df_val[feature_cols]
    y_val = df_val["contact"].astype(int)

    # Generate predictions in inference mode
    val_probs = expert_ensemble.predict(X_val)
    val_preds = (val_probs >= best_thresh).astype(int)

    # Calculate and Print Metric
    mcc = matthews_corrcoef(y_val, val_preds)
    print(f"Final Validation Metric: {mcc}")

    # 4. Failure Analysis
    print("Performing Failure Analysis...")
    # Calculate error magnitude (Absolute Error)
    # y_val is binary (0/1), val_probs is [0,1]
    errors = np.abs(y_val - val_probs)

    # Calculate correlation between Error and Features
    correlations = {}
    # Convert X_val to float32 for correlation calc to save memory/time
    X_val_float = X_val.astype(np.float32)

    for col in feature_cols:
        try:
            # Handle potential NaNs in features by filling with 0 for analysis
            feat_values = X_val_float[col].fillna(0).values
            if np.std(feat_values) > 1e-6:  # Avoid constant columns
                corr = np.corrcoef(feat_values, errors)[0, 1]
                if not np.isnan(corr):
                    correlations[col] = corr
        except Exception:
            continue

    # Sort by magnitude of correlation (identifying features most tied to error)
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features associated with Prediction Error:")
    for name, val in sorted_corrs[:5]:
        print(f" - {name}: {val:.4f}")

    # 5. Conditional Submission
    THRESHOLD_MCC = 0.6865

    if mcc > THRESHOLD_MCC:
        print(
            f"Validation MCC ({mcc}) exceeds threshold ({THRESHOLD_MCC}). Generating Submission..."
        )
        generate_submission(load_cached=True)
    else:
        print(
            f"Validation MCC ({mcc}) does not exceed threshold ({THRESHOLD_MCC}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
