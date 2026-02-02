import sys
import os
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Add current directory to path to ensure library imports work correctly
sys.path.append(os.getcwd())

# Import provided library modules
from library.config import Config
from library.utils import set_seed, save_submission
from library.data_loader import DataLoader
from library.feature_engine import FeatureEngine
from library.model_rf import RandomForestStream
from library.model_nn import NeuralNetworkStream


def run():
    # 1. Setup and Configuration
    Config.setup()
    set_seed(Config.RANDOM_SEED)

    # optimize for fast baseline execution
    Config.MLP_MAX_EPOCHS = 30

    # Initialize pipeline components
    data_loader = DataLoader()
    feature_engine = FeatureEngine()
    rf_stream = RandomForestStream()
    nn_stream = NeuralNetworkStream()

    # 2. Data Loading
    # Using full dataset as it is small (~2k rows) and fits easily within time limits
    print("Loading Data...")
    df_train, df_val, df_test = data_loader.load_data(debug_size=None)

    # 3. Feature Engineering
    # Load cached features if available to speed up execution
    print("Processing Features...")
    rf_data, mlp_data = feature_engine.process_data(
        df_train, df_val, df_test, load_cached_data=True
    )

    # 4. Model Training
    # Stream A: Random Forest
    print("Training Random Forest...")
    rf_val_auc, rf_val_preds = rf_stream.train(rf_data, force_retrain=True)

    # Stream B: Dual-Query MLP
    print("Training MLP...")
    mlp_val_auc, mlp_val_preds = nn_stream.train(mlp_data, force_retrain=True)

    # 5. Ensemble and Validation
    print("Ensembling...")
    w_rf = Config.WEIGHT_RF
    w_mlp = Config.WEIGHT_MLP

    # Normalize weights
    total_weight = w_rf + w_mlp
    w_rf /= total_weight
    w_mlp /= total_weight

    # Calculate weighted ensemble predictions on validation set
    val_preds_ensemble = (w_rf * rf_val_preds) + (w_mlp * mlp_val_preds)
    y_val = rf_data["val"]["y"]

    # Compute and print final metric
    final_metric = roc_auc_score(y_val, val_preds_ensemble)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    error = np.abs(y_val - val_preds_ensemble)

    # Identify numerical columns for correlation analysis
    numeric_cols = df_val.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude non-feature columns and leakage/IDs
    exclude_cols = [
        "requester_received_pizza",
        "target",
        "prediction",
        "error",
        "request_id",
    ]
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    correlations = {}
    for col in numeric_cols:
        # Get feature values, filling NaNs with mean for correlation calculation
        feat_values = df_val[col].fillna(df_val[col].mean()).values

        # Skip constant columns
        if np.std(feat_values) > 1e-9:
            corr = np.corrcoef(feat_values, error)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr

    # Sort by absolute correlation magnitude
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with error magnitude:")
    for name, val in sorted_corr[:5]:
        print(f"{name}: {val:.4f}")

    # 7. Submission Generation
    threshold = 0.7056961514236341
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )

        # Generate predictions for test set
        rf_test_preds = rf_stream.predict(rf_data["test"]["X"])
        mlp_test_preds = nn_stream.predict(mlp_data["test"])

        # Ensemble test predictions
        test_preds_ensemble = (w_rf * rf_test_preds) + (w_mlp * mlp_test_preds)

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {
                "request_id": df_test["request_id"],
                "requester_received_pizza": test_preds_ensemble,
            }
        )

        # Save to file
        save_submission(submission_df)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
