import os
import sys
import numpy as np
import pandas as pd
import warnings
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.feature_engineering import FeatureEngineer
from library.dataset import create_dataloaders
from library.rf_pipeline import RFPipeline
from library.engine import MLPEngine

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run():
    # 1. Setup and Initialization
    Config.setup()
    seed_everything(Config.RANDOM_SEED)

    # 2. Feature Engineering
    # Runs the feature extraction pipeline. Uses cache if available for speed.
    print("Initializing Feature Engineering Pipeline...")
    fe = FeatureEngineer()
    data_dict = fe.run(load_cached_data=True)

    # 3. Data Preparation for MLP
    # Convert processed numpy arrays into PyTorch DataLoaders
    print("Preparing DataLoaders for MLP...")
    train_loader, val_loader, test_loader = create_dataloaders(data_dict)

    # 4. Stream A: Random Forest Pipeline
    # Trains RF, evaluates on Val, predicts on Test
    rf_pipeline = RFPipeline()
    rf_results = rf_pipeline.run(data_dict)

    # 5. Stream B: MLP Engine
    # Trains MLP, evaluates on Val, predicts on Test
    mlp_engine = MLPEngine()
    mlp_results = mlp_engine.run(train_loader, val_loader, test_loader)

    # 6. Ensemble Logic
    print("Computing Ensemble Predictions...")
    w_rf, w_mlp = Config.ENSEMBLE_WEIGHTS

    # Normalize weights
    weight_sum = w_rf + w_mlp
    w_rf /= weight_sum
    w_mlp /= weight_sum

    # Validation Ensemble
    val_probs_rf = rf_results["val_probs"]
    val_probs_mlp = mlp_results["val_probs"]

    # Safety check for shape alignment
    min_len_val = min(len(val_probs_rf), len(val_probs_mlp))
    val_probs_ensemble = (w_rf * val_probs_rf[:min_len_val]) + (
        w_mlp * val_probs_mlp[:min_len_val]
    )

    # Get validation targets
    y_val = data_dict["val"]["y"][:min_len_val]

    # 7. Validation Assessment
    final_auc = roc_auc_score(y_val, val_probs_ensemble)
    print(f"Final Validation Metric: {final_auc}")

    # 8. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Load raw validation metadata to correlate errors with interpretable features
    val_df = pd.read_csv(Config.VAL_PATH)

    # Calculate absolute error
    errors = np.abs(y_val - val_probs_ensemble)

    # Define features to analyze
    analysis_cols = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_posts_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "unix_timestamp_of_request",
    ]

    # Add derived text length feature for analysis
    val_df["text_len"] = val_df["request_text_edit_aware"].fillna("").apply(len)
    analysis_cols.append("text_len")

    print("Correlation between Prediction Error and Features:")
    for col in analysis_cols:
        if col in val_df.columns:
            # Handle potential missing values in raw metadata for correlation calculation
            feat_values = val_df[col].fillna(0).values[:min_len_val]

            if len(feat_values) == len(errors):
                corr, _ = pearsonr(feat_values, errors)
                print(f"{col:<55}: {corr:.4f}")
            else:
                print(f"{col}: Length mismatch ignored.")

    # 9. Submission Generation
    threshold = 0.7135451153926904
    if final_auc > threshold:
        print(
            f"\nValidation metric ({final_auc}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Test Ensemble
        test_probs_rf = rf_results["test_probs"]
        test_probs_mlp = mlp_results["test_probs"]

        min_len_test = min(len(test_probs_rf), len(test_probs_mlp))
        test_probs_ensemble = (w_rf * test_probs_rf[:min_len_test]) + (
            w_mlp * test_probs_mlp[:min_len_test]
        )

        # Load Test Metadata to get request_ids
        test_df = pd.read_csv(Config.TEST_PATH)

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {
                "request_id": test_df["request_id"][:min_len_test],
                "requester_received_pizza": test_probs_ensemble,
            }
        )

        # Save to file
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric ({final_auc}) does not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
