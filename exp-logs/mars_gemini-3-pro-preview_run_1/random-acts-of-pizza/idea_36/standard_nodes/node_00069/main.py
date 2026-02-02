import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.utils import set_seed, ensure_dir
from library.data_loader import DataLoader
from library.features import FeatureGenerator
from library.rf_stream import train_rf_model, predict_rf
from library.mlp_stream import train_mlp_model, predict_mlp

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run():
    # 1. Setup
    print("Initializing Hybrid Ensemble Pipeline...")
    set_seed(Config.RANDOM_STATE)

    # 2. Load Data
    # Uses caching if available to speed up re-runs
    df_train, df_val, df_test = DataLoader.load_data(load_cached_data=True)

    # 3. Feature Generation
    # Generates distinct feature sets for RF and MLP streams
    fg = FeatureGenerator()
    rf_feats, mlp_feats = fg.process_data(
        df_train, df_val, df_test, load_cached_data=True
    )

    # 4. Stream A: Random Forest
    print("\n--- Stream A: Random Forest ---")
    rf_model_path = os.path.join(Config.WORKING_DIR, "rf_model.joblib")

    # Train RF
    rf_model = train_rf_model(
        rf_feats["X_train"],
        rf_feats["y_train"],
        rf_feats["X_val"],
        rf_feats["y_val"],
        save_path=rf_model_path,
    )

    # Predict RF (Validation)
    rf_val_probs = predict_rf(rf_model, rf_feats["X_val"])

    # 5. Stream B: MLP
    print("\n--- Stream B: Dual-Attention Centroid MLP ---")
    mlp_model_path = os.path.join(Config.WORKING_DIR, "best_mlp.pth")

    # Train MLP
    mlp_model = train_mlp_model(mlp_feats, save_path=mlp_model_path)

    # Predict MLP (Validation)
    # Note: predict_mlp handles eval mode and no_grad internally
    mlp_val_probs = predict_mlp(mlp_model, mlp_feats, split="val")

    # 6. Ensemble & Evaluation
    print("\n--- Ensemble Evaluation ---")
    # Weighted Average Ensemble
    final_val_probs = (Config.ENSEMBLE_WEIGHT_RF * rf_val_probs) + (
        Config.ENSEMBLE_WEIGHT_MLP * mlp_val_probs
    )

    # Calculate Metric
    val_labels = rf_feats["y_val"]
    val_auc = roc_auc_score(val_labels, final_val_probs)

    # STRICT OUTPUT FORMAT REQUIRED
    print(f"Final Validation Metric: {val_auc}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(val_labels - final_val_probs)

    # Correlate errors with input numerical features from the validation dataframe
    numeric_cols = df_val.select_dtypes(include=[np.number]).columns
    correlations = {}

    for col in numeric_cols:
        # Skip columns with no variance or all NaNs if any
        if df_val[col].nunique() <= 1:
            continue

        # Handle potential missing values for correlation calculation
        series = df_val[col]
        if series.isnull().any():
            series = series.fillna(series.median())

        corr = series.corr(pd.Series(errors))
        if not np.isnan(corr):
            correlations[col] = corr

    # Print top correlations
    print("Top Feature Correlations with Prediction Error:")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, val in sorted_corr[:10]:
        print(f"{name:<50}: {val:.4f}")

    # 8. Submission
    threshold = 0.7056961514236341
    if val_auc > threshold:
        print(
            f"\nMetric ({val_auc}) > Threshold ({threshold}). Generating submission..."
        )

        # Predict Test RF
        rf_test_probs = predict_rf(rf_model, rf_feats["X_test"])

        # Predict Test MLP
        mlp_test_probs = predict_mlp(mlp_model, mlp_feats, split="test")

        # Ensemble Test Predictions
        final_test_probs = (Config.ENSEMBLE_WEIGHT_RF * rf_test_probs) + (
            Config.ENSEMBLE_WEIGHT_MLP * mlp_test_probs
        )

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {
                "request_id": df_test["request_id"],
                "requester_received_pizza": final_test_probs,
            }
        )

        # Save
        ensure_dir(Config.SUBMISSION_PATH)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric ({val_auc}) <= Threshold ({threshold}). Submission skipped.")


if __name__ == "__main__":
    run()
