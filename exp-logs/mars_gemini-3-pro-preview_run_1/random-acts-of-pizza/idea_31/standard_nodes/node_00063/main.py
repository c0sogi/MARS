import os
import numpy as np
import pandas as pd
import torch
import warnings
from sklearn.metrics import roc_auc_score
from library import config, data_loader, feature_engine, neural_net, train_eval

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run():
    # 1. Setup and Reproducibility
    print("Initializing pipeline...")
    neural_net.set_seed(config.RANDOM_STATE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Data Loading and Feature Engineering
    # This step loads data, fits the pipeline, and generates features for RF and NN streams
    print("Running feature pipeline...")
    train_res, val_res, test_res = feature_engine.run_feature_pipeline(
        load_cached_data=True
    )

    X_rf_train, data_nn_train = train_res
    X_rf_val, data_nn_val = val_res
    X_rf_test, data_nn_test = test_res

    # Extract labels (shared between streams)
    y_train = data_nn_train["labels"]
    y_val = data_nn_val["labels"]

    # 3. Model Training
    # Stream A: Random Forest
    print("\n--- Training Stream A: Random Forest ---")
    rf_model = train_eval.train_rf(X_rf_train, y_train)

    # Stream B: Neural Network
    print("\n--- Training Stream B: Dual-Query MLP ---")
    nn_model, _ = train_eval.train_nn(data_nn_train, data_nn_val, device)

    # 4. Validation and Ensemble
    print("\n--- Validating Ensemble ---")

    # Generate probabilities
    rf_probs_val = rf_model.predict_proba(X_rf_val)[:, 1]
    nn_probs_val = neural_net.predict(nn_model, data_nn_val, device=device)

    # Weighted Ensemble
    w_rf, w_nn = config.ENSEMBLE_WEIGHTS
    ensemble_probs_val = (w_rf * rf_probs_val) + (w_nn * nn_probs_val)

    # Calculate Metric
    val_auc = roc_auc_score(y_val, ensemble_probs_val)
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Load raw validation data to get interpretable feature names
    df_val = data_loader.load_dataset("val", load_cached_data=True)

    # Calculate absolute error
    errors = np.abs(y_val - ensemble_probs_val)

    # Identify numeric columns for correlation analysis
    exclude_cols = [
        "requester_received_pizza",
        "request_id",
        "source_file",
        "unix_timestamp_of_request",
        "unix_timestamp_of_request_utc",
    ]
    numeric_cols = df_val.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    correlations = {}
    for col in numeric_cols:
        # Handle potential NaNs in raw data by filling with 0 or mean for correlation check
        feat_vals = df_val[col].fillna(0).values
        if len(np.unique(feat_vals)) > 1:
            corr = np.corrcoef(feat_vals, errors)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features Correlated with Prediction Error:")
    for name, val in sorted_corr[:10]:
        print(f"{name:<50}: {val:.4f}")

    # 6. Submission Generation
    threshold = 0.7036289345758168
    if val_auc > threshold:
        print(
            f"\nValidation AUC ({val_auc}) exceeds threshold ({threshold}). Generating submission..."
        )
        train_eval.generate_submission(
            rf_model, nn_model, X_rf_test, data_nn_test, device
        )
    else:
        print(
            f"\nValidation AUC ({val_auc}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
