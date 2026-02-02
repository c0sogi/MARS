import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add current directory to path to ensure library imports work correctly
sys.path.append(os.getcwd())

# Import provided library modules
from library import config
from library import model_rf
from library import model_mlp
from library import data_loader

# Set global random seeds for reproducibility
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.SEED)


def run():
    print("Initializing Hybrid Ensemble Pipeline...")

    # -------------------------------------------------------------------------
    # 1. Train Stream A: Action-Profiled Random Forest
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Step 1: Training Stream A (Random Forest)")
    print("=" * 50)

    # Train RF model
    # We use load_cached_data=True to utilize any pre-computed features
    rf_results = model_rf.train_rf_model(load_cached_data=True, debug=config.DEBUG)

    print(f"Stream A (RF) Validation AUC: {rf_results['auc']}")

    # -------------------------------------------------------------------------
    # 2. Train Stream B: Hierarchical Reliability-Gated MLP
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Step 2: Training Stream B (Concatenated Attention MLP)")
    print("=" * 50)

    # Train MLP model
    mlp_results = model_mlp.train_mlp_model(load_cached_data=True, debug=config.DEBUG)

    print(f"Stream B (MLP) Validation AUC: {mlp_results['auc']}")

    # -------------------------------------------------------------------------
    # 3. Ensemble and Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Step 3: Ensembling and Final Evaluation")
    print("=" * 50)

    # Weights defined in config (0.5 / 0.5)
    w_rf = config.ENSEMBLE_WEIGHTS["rf"]
    w_mlp = config.ENSEMBLE_WEIGHTS["mlp"]

    print(f"Ensemble Weights -> RF: {w_rf}, MLP: {w_mlp}")

    # Validation Ensemble
    # Note: y_val should be identical for both models as data_loader is deterministic
    y_val = rf_results["y_val"]
    val_probs_rf = rf_results["val_probs"]
    val_probs_mlp = mlp_results["val_probs"]

    # Simple Weighted Average
    ens_val_probs = (w_rf * val_probs_rf) + (w_mlp * val_probs_mlp)

    # Calculate Final Metric
    final_auc = roc_auc_score(y_val, ens_val_probs)

    # Print EXACTLY as required
    print(f"Final Validation Metric: {final_auc}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Step 4: Failure Analysis")
    print("=" * 50)

    try:
        # Load raw validation metadata to access features for correlation analysis
        # We read directly from CSV to get all original columns
        val_df = pd.read_csv(config.VAL_PATH)

        # Verify alignment
        if len(val_df) != len(y_val):
            print(
                f"Warning: Validation DataFrame length ({len(val_df)}) does not match target length ({len(y_val)}). Skipping detailed analysis."
            )
        else:
            # Calculate absolute prediction error
            # Error = |True Label - Predicted Probability|
            errors = np.abs(y_val - ens_val_probs)

            # Select numerical columns for correlation
            numeric_cols = val_df.select_dtypes(include=[np.number]).columns.tolist()

            # Remove target and ID-like columns if present
            exclude_cols = [
                "requester_received_pizza",
                "unix_timestamp_of_request",
                "unix_timestamp_of_request_utc",
            ]
            numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

            correlations = {}
            for col in numeric_cols:
                # Handle NaNs by filling with median for correlation check
                if val_df[col].isnull().all():
                    continue

                feat_vals = val_df[col].fillna(val_df[col].median())

                # Compute correlation
                if len(feat_vals.unique()) > 1:
                    corr = np.corrcoef(feat_vals, errors)[0, 1]
                    if not np.isnan(corr):
                        correlations[col] = corr

            # Sort by absolute correlation (magnitude of relationship)
            sorted_corr = sorted(
                correlations.items(), key=lambda x: abs(x[1]), reverse=True
            )

            print(
                "Top features correlated with prediction error (systematic failure patterns):"
            )
            for name, val in sorted_corr[:5]:
                print(f"  {name:<50}: {val:.4f}")

    except Exception as e:
        print(f"An error occurred during failure analysis: {e}")

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Step 5: Submission Generation")
    print("=" * 50)

    threshold = 0.6959737721862433

    if final_auc > threshold:
        print(f"Validation metric ({final_auc}) exceeds threshold ({threshold}).")
        print("Generating submission file...")

        # Test Ensemble
        test_probs_rf = rf_results["test_probs"]
        test_probs_mlp = mlp_results["test_probs"]

        ens_test_probs = (w_rf * test_probs_rf) + (w_mlp * test_probs_mlp)

        # Load Test IDs
        test_df = pd.read_csv(config.TEST_PATH)

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {
                "request_id": test_df["request_id"],
                "requester_received_pizza": ens_test_probs,
            }
        )

        # Save
        os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved successfully to: {config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation metric ({final_auc}) did not exceed threshold ({threshold})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    run()
