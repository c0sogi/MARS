import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, print_metric, ensure_directory
from library.feature_engineering import FeatureEngineer
from library.trainer import train_rf_model, predict_rf, train_mlp_model, predict_mlp


def run():
    # 1. Initialization
    print("Initializing pipeline...")
    seed_everything(Config.RANDOM_SEED)
    warnings.filterwarnings("ignore")

    # 2. Feature Engineering
    print("\n=== Step 1: Feature Engineering ===")
    fe = FeatureEngineer()
    # Load cached data if available, otherwise process raw data
    rf_data, mlp_data = fe.process_data(load_cached_data=True)

    # 3. Model Training
    print("\n=== Step 2: Model Training ===")

    # --- Stream A: Random Forest ---
    print("Training Stream A: Interaction Random Forest...")
    rf_model = train_rf_model(rf_data, force_retrain=True)

    # --- Stream B: MLP ---
    print("Training Stream B: FiLM-Gated Dual-Attention MLP...")
    # Note: force_retrain=True ensures we train within this run's time limit context
    mlp_pipeline = train_mlp_model(mlp_data, force_retrain=True)

    # 4. Validation & Ensemble
    print("\n=== Step 3: Validation & Ensemble ===")

    # Get predictions
    rf_val_probs = predict_rf(rf_model, rf_data["X_val"])
    mlp_val_probs = predict_mlp(mlp_pipeline, mlp_data, split="val")

    # Weighted Ensemble
    # Weights are defined in Config (0.5 / 0.5)
    ensemble_val_probs = (Config.RF_WEIGHT * rf_val_probs) + (
        Config.MLP_WEIGHT * mlp_val_probs
    )

    # Calculate Metric
    y_val = rf_data["y_val"]
    val_auc = roc_auc_score(y_val, ensemble_val_probs)

    print("-" * 30)
    print(f"Final Validation Metric: {val_auc}")
    print("-" * 30)

    # 5. Failure Analysis
    print("\n=== Step 4: Failure Analysis ===")
    # Calculate error magnitude
    errors = np.abs(y_val - ensemble_val_probs)

    # We analyze correlations with dense features in X_val (Random Forest features)
    # Structure of X_val_rf: [TF-IDF (5000) | Metadata (9) | Top-K (50) | Consistency (3) | Interactions (9)]
    # We focus on Metadata, Consistency, and Interactions for meaningful analysis

    tfidf_dim = 5000
    meta_dim = 9
    topk_dim = 50
    cons_dim = 3
    inter_dim = 9

    start_dense = tfidf_dim
    end_dense = tfidf_dim + meta_dim + topk_dim + cons_dim + inter_dim

    dense_features = rf_data["X_val"][:, start_dense:end_dense]

    # Create a DataFrame for correlation calculation
    # Define column names for readability
    meta_cols = Config.NUMERIC_COLS
    # TopK cols are generic, Consistency cols are TB, TH, BH
    cons_cols = ["Cons_Title_Body", "Cons_Title_Hist", "Cons_Body_Hist"]
    # Interaction cols are generic
    inter_cols = [f"Interaction_{i}" for i in range(inter_dim)]

    # We'll just map indices to generic names for the analysis to be safe against dimension mismatches
    feature_names = (
        meta_cols + [f"TopK_{i}" for i in range(topk_dim)] + cons_cols + inter_cols
    )

    # Ensure dimensions match
    if dense_features.shape[1] == len(feature_names):
        df_analysis = pd.DataFrame(dense_features, columns=feature_names)
        df_analysis["error"] = errors

        correlations = df_analysis.corr()["error"].drop("error")
        top_correlations = correlations.abs().sort_values(ascending=False).head(10)

        print("Top 10 Features correlated with Prediction Error:")
        print(top_correlations)
    else:
        print(
            "Skipping detailed feature naming in failure analysis due to dimension mismatch."
        )
        # Fallback to simple index correlation
        corrs = []
        for i in range(dense_features.shape[1]):
            c = np.corrcoef(dense_features[:, i], errors)[0, 1]
            corrs.append((i, c))
        corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        print("Top 5 Feature Indices correlated with Error (Index, Corr):")
        for idx, val in corrs[:5]:
            print(f"Feature Index {idx}: {val:.4f}")

    # 6. Submission
    print("\n=== Step 5: Submission Generation ===")
    threshold = 0.7135451153926904

    if val_auc > threshold:
        print(
            f"Validation AUC ({val_auc:.6f}) > Threshold ({threshold:.6f}). Generating submission..."
        )

        # Predict on Test
        rf_test_probs = predict_rf(rf_model, rf_data["X_test"])
        mlp_test_probs = predict_mlp(mlp_pipeline, mlp_data, split="test")

        ensemble_test_probs = (Config.RF_WEIGHT * rf_test_probs) + (
            Config.MLP_WEIGHT * mlp_test_probs
        )

        # Load Test Metadata to get IDs
        df_test = pd.read_csv(Config.TEST_PATH)

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {
                "request_id": df_test[Config.ID_COL],
                "requester_received_pizza": ensemble_test_probs,
            }
        )

        # Save
        ensure_directory(Config.SUBMISSION_PATH)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation AUC ({val_auc:.6f}) did not meet threshold ({threshold:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    run()
