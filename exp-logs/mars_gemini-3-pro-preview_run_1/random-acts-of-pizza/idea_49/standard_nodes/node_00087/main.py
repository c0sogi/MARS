import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

from library.config import Config
from library.data_loader import DataLoader
from library.feature_engineering import FeatureEngineer
from library.models_mlp import train_model as train_mlp_model, predict as predict_mlp
from library.models_rf import train_rf_model, predict_rf

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run():
    # 1. Configuration
    # Set random seed for reproducibility
    Config.RANDOM_SEED = 42
    # Use full dataset (dataset is small enough for fast execution on A100)
    Config.MAX_SAMPLES = None
    Config.ensure_dirs()

    print("Initializing workflow...")

    # 2. Load Data
    print("Loading data...")
    dl = DataLoader()
    train_df, val_df, test_df = dl.load_dataset(load_cached_data=True)

    # Extract labels and IDs
    y_train = train_df["requester_received_pizza"].astype(int).values
    y_val = val_df["requester_received_pizza"].astype(int).values
    test_ids = test_df["request_id"].values

    # 3. Feature Engineering
    print("Engineering features...")
    fe = FeatureEngineer()
    features = fe.create_features(load_cached_data=True)

    # 4. Train MLP Model (Stream B)
    print("Training MLP Model (Stream B)...")
    # Determine input dimension for the metadata branch of the MLP
    mlp_meta_dim = features["train"]["mlp"]["metadata"].shape[1]

    mlp_model = train_mlp_model(
        features["train"]["mlp"],
        y_train,
        features["val"]["mlp"],
        y_val,
        mlp_meta_dim,
    )

    # 5. Train Random Forest Model (Stream A)
    print("Training Random Forest Model (Stream A)...")
    rf_model = train_rf_model(
        features["train"]["rf"], y_train, features["val"]["rf"], y_val
    )

    # 6. Ensemble Validation
    print("Validating Ensemble...")
    # Inference on validation set
    val_preds_mlp = predict_mlp(mlp_model, features["val"]["mlp"])
    val_preds_rf = predict_rf(rf_model, features["val"]["rf"])

    # Weighted Average Ensemble
    w_rf = Config.ENSEMBLE_WEIGHT_RF
    w_mlp = Config.ENSEMBLE_WEIGHT_MLP
    val_preds_ensemble = (w_rf * val_preds_rf) + (w_mlp * val_preds_mlp)

    # Calculate Metric
    val_auc = roc_auc_score(y_val, val_preds_ensemble)
    print(f"Final Validation Metric: {val_auc}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude (absolute difference between true label and predicted probability)
    errors = np.abs(y_val - val_preds_ensemble)

    # Define features to analyze for correlation with error
    analysis_cols = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_posts_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_number_of_subreddits_at_request",
    ]

    # Add text length as a derived feature for analysis
    val_df["text_length"] = val_df["request_text_edit_aware"].fillna("").apply(len)
    analysis_cols.append("text_length")

    print(f"{'Feature':<60} | {'Correlation with Error':<20}")
    print("-" * 85)

    for col in analysis_cols:
        if col in val_df.columns:
            # Handle potential NaNs in raw data by filling with 0 for analysis purposes
            feat_values = val_df[col].fillna(0).values
            # Ensure numeric type
            feat_values = pd.to_numeric(feat_values, errors="coerce")

            # Calculate correlation on valid entries
            valid_mask = ~np.isnan(feat_values)
            if np.sum(valid_mask) > 1:
                corr, _ = pearsonr(feat_values[valid_mask], errors[valid_mask])
                print(f"{col:<60} | {corr:.4f}")

    # 8. Submission Generation
    threshold = 0.7135451153926904

    if val_auc > threshold:
        print(
            f"\nValidation metric ({val_auc}) > threshold ({threshold}). Generating submission..."
        )

        # Test Inference
        test_preds_mlp = predict_mlp(mlp_model, features["test"]["mlp"])
        test_preds_rf = predict_rf(rf_model, features["test"]["rf"])

        # Ensemble Test Predictions
        test_preds_ensemble = (w_rf * test_preds_rf) + (w_mlp * test_preds_mlp)

        # Create submission DataFrame
        submission = pd.DataFrame(
            {"request_id": test_ids, "requester_received_pizza": test_preds_ensemble}
        )

        # Save to CSV
        submission_path = Config.SUBMISSION_PATH
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"\nValidation metric ({val_auc}) <= threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
