import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import warnings

# Import from provided library files
from library.config import Config
from library.utils import set_seed, ensure_dir
from library.feature_engineering import FeaturePipeline
from library.model_training import train_rf, predict_rf, train_mlp, predict_mlp

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # 1. Initialization and Configuration
    set_seed(Config.SEED)
    ensure_dir(Config.SUBMISSION_DIR)

    # 2. Data Loading and Feature Engineering
    # Initialize the pipeline and load data (computing if cache is missing)
    pipeline = FeaturePipeline()
    data = pipeline.run(load_cached_data=True)

    # 3. Stream A: Random Forest Training
    # Extract RF-specific data
    X_rf_train, y_train, X_rf_val, y_val, X_rf_test, test_ids = data["rf"]

    # Train Random Forest
    rf_model = train_rf(X_rf_train, y_train, X_rf_val, y_val)

    # Generate Validation Predictions for RF
    rf_val_preds = predict_rf(rf_model, X_rf_val)

    # 4. Stream B: Dual-Query MLP Training
    # Extract MLP-specific data
    mlp_train_data = data["mlp"]["train"]
    mlp_val_data = data["mlp"]["val"]

    # Train MLP
    mlp_model = train_mlp(mlp_train_data, mlp_val_data)

    # Generate Validation Predictions for MLP
    mlp_val_preds = predict_mlp(mlp_model, mlp_val_data)

    # 5. Ensemble Evaluation
    # Weighted average of probabilities (0.5 RF + 0.5 MLP)
    ensemble_val_preds = (Config.ENSEMBLE_WEIGHT_RF * rf_val_preds) + (
        Config.ENSEMBLE_WEIGHT_MLP * mlp_val_preds
    )

    # Calculate and print the required metric
    val_auc = roc_auc_score(y_val, ensemble_val_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    # Correlate error magnitude with input features to identify systematic failures
    errors = np.abs(y_val - ensemble_val_preds)

    # Construct feature names for the RF dataset (Metadata + Sim + TopK + TFIDF)
    # This mapping aligns with the stacking order in FeaturePipeline
    meta_cols = [
        "account_age",
        "days_since_first_post",
        "comments_total",
        "comments_raop",
        "posts_total",
        "posts_raop",
        "subs_total",
        "up_minus_down",
        "up_plus_down",
    ]
    sim_cols = ["global_sim_title", "global_sim_body"]
    topk_cols = [f"topk_{i}" for i in range(Config.TOP_K_SUBREDDITS)]
    # We only explicitly name the dense features for analysis; TFIDF cols are numerous
    dense_feature_names = meta_cols + sim_cols + topk_cols

    correlations = []
    # Analyze correlations for dense features
    for i, name in enumerate(dense_feature_names):
        if i < X_rf_val.shape[1]:
            feat_col = X_rf_val[:, i]
            # Check for constant columns to avoid division by zero in correlation
            if np.std(feat_col) > 1e-9:
                corr, _ = pearsonr(feat_col, errors)
                if not np.isnan(corr):
                    correlations.append((name, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nFailure Analysis - Top Features correlated with Error Magnitude:")
    for name, corr in correlations[:10]:
        print(f"{name}: {corr:.4f}")

    # 7. Submission Generation
    threshold = 0.7056961514236341
    if val_auc > threshold:
        # Generate Test Predictions
        rf_test_preds = predict_rf(rf_model, X_rf_test)

        mlp_test_data = data["mlp"]["test"]
        mlp_test_preds = predict_mlp(mlp_model, mlp_test_data)

        # Ensemble Test Predictions
        final_test_preds = (Config.ENSEMBLE_WEIGHT_RF * rf_test_preds) + (
            Config.ENSEMBLE_WEIGHT_MLP * mlp_test_preds
        )

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {"request_id": test_ids, "requester_received_pizza": final_test_preds}
        )

        # Save to file
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {val_auc} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
