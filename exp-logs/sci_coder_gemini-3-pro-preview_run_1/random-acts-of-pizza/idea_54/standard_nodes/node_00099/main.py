import os
import sys
import numpy as np
import pandas as pd

# Import from the provided library files
from library.config import RANDOM_STATE, ENSEMBLE_WEIGHTS, SUBMISSION_PATH, WORKING_DIR
from library.utils import set_seed, compute_auc
from library.data_loader import load_dataset
from library.feature_engine import FeatureProcessor
from library.model_rf import train_rf_model
from library.model_mlp import train_mlp_model


def run():
    # 1. Setup
    set_seed(RANDOM_STATE)
    print("Starting execution pipeline...")

    # 2. Load Data
    # load_cached_data=True allows using pre-computed parquet files if available
    print("Loading datasets...")
    df_train, df_val, df_test = load_dataset(load_cached_data=True)
    print(
        f"Data loaded. Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}"
    )

    # 3. Feature Processing
    # This step generates the specific feature sets for both the RF and MLP streams
    print("Generating features...")
    processor = FeatureProcessor()
    rf_features, mlp_features, targets = processor.process(
        df_train, df_val, df_test, load_cached_data=True
    )
    print("Feature generation complete.")

    # 4. Train Random Forest (Stream A)
    print("\n=== Stream A: Random Forest ===")
    rf_val_preds, rf_test_preds, rf_model = train_rf_model(rf_features, targets)

    # 5. Train Topology-Aware MLP (Stream B)
    print("\n=== Stream B: Topology-Aware MLP ===")
    mlp_val_preds, mlp_test_preds, mlp_model = train_mlp_model(mlp_features, targets)

    # 6. Ensemble
    print("\n=== Ensembling ===")
    w_rf = ENSEMBLE_WEIGHTS["rf"]
    w_mlp = ENSEMBLE_WEIGHTS["mlp"]
    print(f"Weights -> RF: {w_rf}, MLP: {w_mlp}")

    final_val_preds = (w_rf * rf_val_preds) + (w_mlp * mlp_val_preds)
    final_test_preds = (w_rf * rf_test_preds) + (w_mlp * mlp_test_preds)

    # 7. Evaluation
    y_val = targets["val"]
    final_auc = compute_auc(y_val, final_val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 8. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(y_val - final_val_preds)

    # Select interpretable numerical columns from the validation dataframe for correlation
    analysis_cols = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_posts_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
        "requester_number_of_subreddits_at_request",
    ]

    correlations = []
    for col in analysis_cols:
        if col in df_val.columns:
            # Handle potential NaNs in raw data by filling with 0 for analysis
            feat_vals = df_val[col].fillna(0).values
            if len(feat_vals) == len(errors):
                # Compute correlation
                corr = np.corrcoef(feat_vals, errors)[0, 1]
                correlations.append((col, corr))

    # Sort by magnitude of correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top Feature Correlations with Prediction Error:")
    for name, val in correlations:
        print(f"{name:<50}: {val:.4f}")

    # 9. Submission
    threshold = 0.7135451153926904
    if final_auc > threshold:
        print(
            f"\nValidation metric ({final_auc}) exceeds threshold ({threshold}). Generating submission..."
        )

        submission = pd.DataFrame(
            {
                "request_id": df_test["request_id"],
                "requester_received_pizza": final_test_preds,
            }
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

        # Save
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric ({final_auc}) does not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
