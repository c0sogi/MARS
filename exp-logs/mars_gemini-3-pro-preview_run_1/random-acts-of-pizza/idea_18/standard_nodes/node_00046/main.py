import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.feature_engine import FeatureExtractor
from library.rf_model import RFModel
from library.mlp_model import MLPModel


def run():
    # 1. Initialization
    set_seed(Config.RANDOM_STATE)
    print("Initializing pipeline...")

    # 2. Feature Engineering
    # Loads data and generates/loads features for Train, Val, and Test
    print("Running Feature Extractor...")
    fe = FeatureExtractor()
    data = fe.run(load_cached_data=True)

    # 3. Model Training

    # --- Stream A: Random Forest ---
    print("\n=== Training Random Forest (Stream A) ===")
    rf = RFModel()
    rf.train(
        X_train=data["train"]["rf_features"],
        y_train=data["train"]["y"],
        X_val=data["val"]["rf_features"],
        y_val=data["val"]["y"],
    )

    # --- Stream B: MLP (Gated Attention) ---
    print("\n=== Training MLP (Stream B) ===")
    mlp = MLPModel()
    mlp.train(data_train=data["train"], data_val=data["val"])

    # 4. Validation & Ensemble
    print("\n=== Validating Ensemble ===")
    y_val = data["val"]["y"]

    # Get predictions
    rf_probs_val = rf.predict_proba(data["val"]["rf_features"])
    mlp_probs_val = mlp.predict_proba(data["val"])

    # Weighted Ensemble
    w_rf = Config.ENSEMBLE_WEIGHTS["rf"]
    w_mlp = Config.ENSEMBLE_WEIGHTS["mlp"]
    ensemble_probs_val = (w_rf * rf_probs_val) + (w_mlp * mlp_probs_val)

    # Metric
    val_auc = roc_auc_score(y_val, ensemble_probs_val)
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(y_val - ensemble_probs_val)

    # Load raw validation metadata for interpretable feature correlation
    df_val = pd.read_csv(Config.VAL_PATH)

    # Identify numeric columns for correlation analysis
    numeric_cols = df_val.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target and non-feature columns
    exclude_cols = ["requester_received_pizza", "request_id"]
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    correlations = {}
    for col in numeric_cols:
        # Fill NaNs with median for correlation calculation
        feat_values = df_val[col].fillna(df_val[col].median())

        # Calculate Pearson correlation
        if feat_values.std() > 0:
            corr = np.corrcoef(feat_values, errors)[0, 1]
            correlations[col] = corr
        else:
            correlations[col] = 0.0

    # Print top correlations
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Top features correlated with prediction error:")
    for name, score in sorted_corr[:5]:
        print(f"  {name}: {score:.4f}")

    # 6. Submission
    threshold = 0.6959737721862433
    if val_auc > threshold:
        print("\n=== Generating Submission ===")

        # Inference on Test Set
        rf_probs_test = rf.predict_proba(data["test"]["rf_features"])
        mlp_probs_test = mlp.predict_proba(data["test"])

        # Ensemble
        ensemble_probs_test = (w_rf * rf_probs_test) + (w_mlp * mlp_probs_test)

        # Prepare Submission DataFrame
        # We need to read the test file again to get request_ids in correct order
        df_test = pd.read_csv(Config.TEST_PATH)

        submission = pd.DataFrame(
            {
                "request_id": df_test["request_id"],
                "requester_received_pizza": ensemble_probs_test,
            }
        )

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation AUC ({val_auc}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
