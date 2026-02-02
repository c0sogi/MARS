import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from library import config, utils, feature_engineering, rf_module, nn_module


def run():
    # 1. Setup and Initialization
    print("Initializing pipeline...")
    utils.set_seed()

    # 2. Feature Generation / Loading
    # The pipeline handles caching internally.
    pipeline = feature_engineering.FeaturePipeline()
    rf_data, mlp_data = pipeline.run(load_cached_data=True)

    # 3. Stream A: Random Forest Pipeline
    # Returns val_preds, test_preds, model
    rf_val_preds, rf_test_preds, rf_model = rf_module.run_rf_pipeline(rf_data)

    # 4. Stream B: Neural Network Pipeline
    # Returns val_preds, test_preds, model
    nn_val_preds, nn_test_preds, nn_model = nn_module.run_nn_pipeline(mlp_data)

    # 5. Ensemble
    print("Ensembling predictions...")
    # Weights from config
    w_rf = config.WEIGHT_RF
    w_mlp = config.WEIGHT_MLP

    # Weighted Average for Validation
    ensemble_val_preds = (w_rf * rf_val_preds) + (w_mlp * nn_val_preds)

    # Weighted Average for Test
    ensemble_test_preds = (w_rf * rf_test_preds) + (w_mlp * nn_test_preds)

    # 6. Evaluation
    y_val = rf_data["y_val"]  # Same labels for both
    val_auc = roc_auc_score(y_val, ensemble_val_preds)

    print(f"Final Validation Metric: {val_auc}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(y_val - ensemble_val_preds)

    # Load raw validation data to get interpretable feature names
    df_val = pd.read_csv(config.VAL_DATA_PATH)

    # Select numerical columns for correlation analysis
    numeric_cols = df_val.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target and IDs if present
    exclude_cols = [
        "requester_received_pizza",
        "unix_timestamp_of_request",
        "unix_timestamp_of_request_utc",
    ]
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    correlations = {}
    for col in numeric_cols:
        # Handle NaNs in raw data for correlation calculation
        if df_val[col].isnull().any():
            feat_values = df_val[col].fillna(df_val[col].median())
        else:
            feat_values = df_val[col]

        # Calculate correlation with error
        corr = np.corrcoef(feat_values, errors)[0, 1]
        if not np.isnan(corr):
            correlations[col] = corr

    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for name, corr in sorted_corrs[:5]:
        print(f"{name}: {corr:.4f}")

    # 8. Submission
    threshold = 0.6959737721862433
    if val_auc > threshold:
        print(f"\nValidation metric {val_auc} > {threshold}. Generating submission...")

        # Load test metadata to get request_ids
        df_test = pd.read_csv(config.TEST_DATA_PATH)

        submission_df = pd.DataFrame(
            {
                "request_id": df_test["request_id"],
                "requester_received_pizza": ensemble_test_preds,
            }
        )

        # Ensure submission directory exists
        os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

        # Save submission
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric {val_auc} <= {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    run()
