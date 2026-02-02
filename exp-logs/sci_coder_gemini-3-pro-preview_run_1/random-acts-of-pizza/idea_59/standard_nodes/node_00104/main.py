import os
import numpy as np
import pandas as pd
import warnings

# Import from provided library files
from library.config import Config
from library.utils import set_seed, compute_auc
from library.data_loader import load_data
from library.features import FeatureProcessor
from library.model_rf import run_rf_pipeline
from library.model_mlp import run_mlp_pipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Reproducibility
    set_seed(Config.SEED)
    print("Initializing Hybrid Ensemble Pipeline...")

    # 2. Data Loading
    # Uses caching to speed up subsequent runs
    print("Loading datasets...")
    train_df, val_df, test_df = load_data(load_cached_data=True)

    # 3. Feature Processing
    # Generates distinct feature sets for RF (Interaction/TF-IDF) and MLP (Semantic/Gating)
    print("Processing features...")
    processor = FeatureProcessor()
    processed_data = processor.process_data(
        train_df, val_df, test_df, load_cached_data=True
    )

    # 4. Stream A: Random Forest Pipeline
    print("Training Stream A: Interaction-Projected Random Forest...")
    rf_model, rf_val_probs, rf_test_probs = run_rf_pipeline(processed_data)

    # 5. Stream B: MLP Pipeline
    print("Training Stream B: Non-Linear Orthogonal Skip-Gated MLP...")
    mlp_model, mlp_val_probs, mlp_test_probs = run_mlp_pipeline(processed_data)

    # 6. Ensemble Aggregation
    print("Aggregating Ensemble Predictions...")

    # Validation Ensemble
    if rf_val_probs is not None and mlp_val_probs is not None:
        final_val_probs = (Config.ENSEMBLE_WEIGHT_RF * rf_val_probs) + (
            Config.ENSEMBLE_WEIGHT_MLP * mlp_val_probs
        )
    else:
        final_val_probs = None
        print("Warning: Validation predictions missing from one or both models.")

    # Test Ensemble
    final_test_probs = (Config.ENSEMBLE_WEIGHT_RF * rf_test_probs) + (
        Config.ENSEMBLE_WEIGHT_MLP * mlp_test_probs
    )

    # 7. Validation Assessment & Failure Analysis
    if final_val_probs is not None:
        # Retrieve ground truth
        y_val = processed_data.get("val_y")

        if y_val is not None:
            # Compute Metric
            auc = compute_auc(y_val, final_val_probs)
            print(f"Final Validation Metric: {auc}")

            # Failure Analysis
            print("\n=== Failure Analysis ===")
            residuals = np.abs(y_val - final_val_probs)

            # Correlate residuals with numerical features in validation set
            numeric_cols = val_df.select_dtypes(include=[np.number]).columns
            correlations = {}

            for col in numeric_cols:
                # Handle NaNs by filling with 0 for correlation check
                feat_values = val_df[col].fillna(0).values
                # Check variance to avoid warnings
                if np.std(feat_values) > 1e-9 and np.std(residuals) > 1e-9:
                    corr = np.corrcoef(residuals, feat_values)[0, 1]
                    if not np.isnan(corr):
                        correlations[col] = corr

            # Sort and print top correlations
            sorted_corr = sorted(
                correlations.items(), key=lambda x: abs(x[1]), reverse=True
            )
            print("Top 5 Features Correlated with Prediction Error:")
            for name, val in sorted_corr[:5]:
                print(f"{name:<50}: {val:.4f}")
            print("========================\n")

            # 8. Submission Generation
            threshold = 0.7135451153926904

            if auc > threshold:
                print(
                    f"Validation AUC ({auc}) exceeds threshold ({threshold}). Generating submission..."
                )

                submission_df = pd.DataFrame(
                    {
                        "request_id": test_df["request_id"],
                        "requester_received_pizza": final_test_probs,
                    }
                )

                # Ensure output directory exists
                os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

                submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
                print(f"Submission saved to {Config.SUBMISSION_PATH}")
            else:
                print(
                    f"Validation AUC ({auc}) does not exceed threshold ({threshold}). Submission skipped."
                )
        else:
            print("Error: Validation labels not found.")


if __name__ == "__main__":
    main()
