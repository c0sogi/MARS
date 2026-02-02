import os
import sys
import numpy as np
import pandas as pd
import torch
import random
from sklearn.metrics import roc_auc_score

# Add current directory to path to ensure imports work
sys.path.append(os.getcwd())

# Import provided library modules
from library import config
from library import data_loader
from library import features
from library import model_rf
from library import model_mlp


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run():
    # 1. Initialization
    print("Initializing orchestration...")
    set_seed(config.RANDOM_STATE)

    # 2. Train Stream A: Random Forest
    print("\n" + "=" * 40)
    print("Stream A: Semantic Anchor Random Forest")
    print("=" * 40)
    # train_rf_stream returns (model, val_probs, test_probs)
    rf_model, rf_val_probs, rf_test_probs = model_rf.train_rf_stream(
        load_cached_data=True
    )

    # 3. Train Stream B: Residual Attention MLP
    print("\n" + "=" * 40)
    print("Stream B: Residual Attention MLP")
    print("=" * 40)
    # train_mlp_stream returns (model, val_probs, test_probs)
    mlp_model, mlp_val_probs, mlp_test_probs = model_mlp.train_mlp_stream(
        load_cached_data=True
    )

    # 4. Ensemble and Validation
    print("\n" + "=" * 40)
    print("Ensemble and Evaluation")
    print("=" * 40)

    # Load ground truth for validation
    _, val_df, test_df = data_loader.load_datasets(load_cached_data=True)
    y_val = val_df["requester_received_pizza"].astype(int).values

    # Calculate Ensemble Probabilities
    # Weights: RF=0.5, MLP=0.5
    w_rf = config.ENSEMBLE_WEIGHTS["rf"]
    w_mlp = config.ENSEMBLE_WEIGHTS["mlp"]

    ensemble_val_probs = (w_rf * rf_val_probs) + (w_mlp * mlp_val_probs)
    ensemble_test_probs = (w_rf * rf_test_probs) + (w_mlp * mlp_test_probs)

    # Calculate Metric
    final_auc = roc_auc_score(y_val, ensemble_val_probs)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\n" + "=" * 40)
    print("Failure Analysis")
    print("=" * 40)

    # Calculate Error (Absolute difference between target and probability)
    # Target is 0 or 1.
    errors = np.abs(y_val - ensemble_val_probs)

    # Load validation metadata features for correlation analysis
    fe = features.FeatureEngineer()
    val_meta = fe.generate_metadata_features(
        val_df, split_name="val", load_cached_data=True
    )

    # Select numeric columns for correlation
    numeric_cols = val_meta.select_dtypes(include=[np.number]).columns.tolist()

    # Compute correlations
    correlations = {}
    for col in numeric_cols:
        if val_meta[col].nunique() > 1:  # Skip constant columns
            # Fill NaNs for correlation calculation
            feat_values = val_meta[col].fillna(val_meta[col].median())
            corr = np.corrcoef(feat_values, errors)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features Correlated with Model Error:")
    for name, score in sorted_corr[:10]:
        print(f"{name:<60}: {score:.4f}")

    # 6. Submission Generation
    threshold = 0.6959737721862433

    if final_auc > threshold:
        print("\n" + "=" * 40)
        print("Generating Submission")
        print("=" * 40)

        submission_df = pd.DataFrame(
            {
                "request_id": test_df["request_id"],
                "requester_received_pizza": ensemble_test_probs,
            }
        )

        # Ensure output directory exists
        os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

        # Save
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
        print(f"Submission shape: {submission_df.shape}")

    else:
        print(
            f"\nValidation metric ({final_auc}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
