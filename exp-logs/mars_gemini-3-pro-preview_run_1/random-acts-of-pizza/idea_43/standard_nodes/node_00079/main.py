import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.config import Config
from library.utils import set_seed, suppress_warnings
from library.data_loader import load_data
from library.stream_a_rf import RandomForestPipeline
from library.stream_b_mlp import MLPPipeline


def main():
    # 1. Setup and Initialization
    set_seed(Config.SEED)
    suppress_warnings()

    print("Initializing Hybrid Ensemble Solution...")

    # 2. Load Data
    # We use load_cached_data=True to speed up execution if features were previously computed
    print("Loading data...")
    train_df, val_df, test_df = load_data(load_cached_data=True, debug=Config.DEBUG)

    # 3. Execute Stream A: Consistency-Augmented Top-K Random Forest
    print("\n--- Running Stream A: Random Forest ---")
    rf_pipeline = RandomForestPipeline()
    rf_val_preds, rf_test_preds = rf_pipeline.run(
        train_df, val_df, test_df, load_cached_data=True
    )

    # 4. Execute Stream B: Unified Credibility-Gated MLP
    print("\n--- Running Stream B: Credibility-Gated MLP ---")
    mlp_pipeline = MLPPipeline()
    mlp_val_preds, mlp_test_preds = mlp_pipeline.run(
        train_df, val_df, test_df, load_cached_data=True
    )

    # 5. Ensemble Predictions
    print("\n--- Ensembling Models ---")
    # Weights are defined in Config (0.5 / 0.5)
    w_rf = Config.ENSEMBLE_WEIGHTS["rf"]
    w_mlp = Config.ENSEMBLE_WEIGHTS["mlp"]

    final_val_preds = (w_rf * rf_val_preds) + (w_mlp * mlp_val_preds)
    final_test_preds = (w_rf * rf_test_preds) + (w_mlp * mlp_test_preds)

    # 6. Validation Evaluation
    # Get ground truth labels
    y_val = val_df[Config.TARGET_COL].astype(int).values

    # Calculate AUC
    val_auc = roc_auc_score(y_val, final_val_preds)

    # Print required metric format
    print(f"Final Validation Metric: {val_auc}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(y_val - final_val_preds)

    # Identify numerical columns for correlation analysis
    numeric_cols = val_df.select_dtypes(include=[np.number]).columns.tolist()
    if Config.TARGET_COL in numeric_cols:
        numeric_cols.remove(Config.TARGET_COL)

    correlations = {}
    for col in numeric_cols:
        # Fill NaNs with 0 or median for correlation check
        feat_values = val_df[col].fillna(0).values

        # Ensure variance exists
        if len(np.unique(feat_values)) > 1:
            corr = np.corrcoef(errors, feat_values)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr

    # Sort by absolute correlation magnitude
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for name, val in sorted_corr[:5]:
        print(f"{name:<50}: {val:.4f}")

    # 8. Submission Generation
    threshold = 0.7135451153926904

    if val_auc > threshold:
        print(
            f"\nValidation AUC ({val_auc:.6f}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Create submission DataFrame
        submission = pd.DataFrame(
            {
                "request_id": test_df[Config.ID_COL],
                "requester_received_pizza": final_test_preds,
            }
        )

        # Ensure submission directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Save to CSV
        submission.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(
            f"\nValidation AUC ({val_auc:.6f}) does not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
