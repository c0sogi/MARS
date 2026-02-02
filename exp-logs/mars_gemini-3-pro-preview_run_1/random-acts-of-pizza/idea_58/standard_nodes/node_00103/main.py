import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed, load_dataset, save_submission
from library.features import get_features
from library.rf_model import MultiInteractionRF
from library.mlp_trainer import MLPTrainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Reproducibility
    set_seed(Config.RANDOM_SEED)
    Config.create_directories()

    # 2. Load and Process Data
    # get_features handles loading raw data, fitting the pipeline, and returning processed features.
    # It uses caching (load_cached_data=True) to avoid re-computing embeddings if possible.
    print("Loading and processing features...")
    (train_data, val_data, test_data) = get_features(load_cached_data=True)

    # Unpack data tuples: (mlp_feature_dict, rf_feature_array, labels)
    train_mlp, train_rf, train_y = train_data
    val_mlp, val_rf, val_y = val_data
    test_mlp, test_rf, test_y = test_data

    # 3. Train Stream A: Multi-Interaction Random Forest
    print("Training Stream A: Multi-Interaction Random Forest...")
    rf_model = MultiInteractionRF()
    # Train on training set, validate on validation set
    rf_model.train(train_rf, train_y, val_rf, val_y)

    # 4. Train Stream B: Decoupled Gated MLP
    print("Training Stream B: Decoupled Gated MLP...")
    mlp_trainer = MLPTrainer()
    # Train on training set, validate on validation set (for early stopping)
    mlp_trainer.train((train_mlp, train_y), (val_mlp, val_y))

    # 5. Validation Inference & Ensemble
    print("Running validation inference...")

    # Generate probabilities from both models
    val_pred_rf = rf_model.predict(val_rf)
    val_pred_mlp = mlp_trainer.predict(val_mlp)

    # Weighted Ensemble
    # Weights are defined in Config (0.5/0.5)
    w_rf = Config.ENSEMBLE_WEIGHT_RF
    w_mlp = Config.ENSEMBLE_WEIGHT_MLP
    val_pred_ensemble = (w_rf * val_pred_rf) + (w_mlp * val_pred_mlp)

    # Compute Metric
    val_auc = roc_auc_score(val_y, val_pred_ensemble)
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("Performing failure analysis...")
    # Load raw validation dataframe to access interpretable metadata columns
    df_val = load_dataset("val")

    # Calculate absolute error
    errors = np.abs(val_y - val_pred_ensemble)

    # Define features to analyze for correlation with error
    analysis_cols = [
        "requester_account_age_in_days_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_posts_at_request",
    ]

    # Add text length if available
    if Config.TEXT_BODY_COL in df_val.columns:
        df_val["text_length"] = (
            df_val[Config.TEXT_BODY_COL].fillna("").astype(str).apply(len)
        )
        analysis_cols.append("text_length")

    print("Correlation between Model Error and Input Features:")
    for col in analysis_cols:
        if col in df_val.columns:
            # Fill NaNs with 0 for correlation calculation
            feat_values = df_val[col].fillna(0).values
            # Ensure standard deviation is not zero to avoid NaN correlation
            if np.std(feat_values) > 0 and np.std(errors) > 0:
                corr = np.corrcoef(errors, feat_values)[0, 1]
                print(f"  {col}: {corr:.4f}")
            else:
                print(f"  {col}: Undefined (Zero Variance)")

    # 7. Submission
    # Threshold defined in task description
    THRESHOLD = 0.7135451153926904

    if val_auc > THRESHOLD:
        print(
            f"Validation metric ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Inference on Test Set
        test_pred_rf = rf_model.predict(test_rf)
        test_pred_mlp = mlp_trainer.predict(test_mlp)

        # Ensemble
        test_pred_ensemble = (w_rf * test_pred_rf) + (w_mlp * test_pred_mlp)

        # Prepare Submission DataFrame
        # Need to load test dataset to get request_ids
        df_test = load_dataset("test")
        submission_df = pd.DataFrame(
            {
                Config.ID_COL: df_test[Config.ID_COL],
                Config.TARGET_COL: test_pred_ensemble,
            }
        )

        # Save
        save_submission(submission_df)
    else:
        print(
            f"Validation metric ({val_auc}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
