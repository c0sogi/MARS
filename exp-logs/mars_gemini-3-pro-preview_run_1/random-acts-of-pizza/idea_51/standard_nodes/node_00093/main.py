import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# Import from the provided library files
from library.config import VAL_CSV, TEST_CSV, ENSEMBLE_WEIGHTS, RANDOM_STATE
from library.utils import seed_everything, save_submission, load_from_cache
from library.feature_engineering import run_feature_engineering
from library.rf_learner import run_rf_learner
from library.dataset import get_dataloaders
from library.engine import train_mlp_model, predict_mlp


def perform_failure_analysis(val_preds, val_targets):
    """
    Analyzes the correlation between prediction error and input features
    on the validation set.
    """
    print("\n" + "=" * 30)
    print("FAILURE ANALYSIS")
    print("=" * 30)

    # Load raw validation metadata for interpretable analysis
    if not os.path.exists(VAL_CSV):
        print("Validation CSV not found, skipping detailed failure analysis.")
        return

    val_df = pd.read_csv(VAL_CSV)

    # Calculate error (absolute difference)
    # Target is 0 or 1, Pred is probability [0, 1]
    errors = np.abs(val_targets - val_preds)

    # Select numerical columns for correlation
    numeric_cols = val_df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target and ID if present
    cols_to_exclude = ["requester_received_pizza", "request_id"]
    numeric_cols = [c for c in numeric_cols if c not in cols_to_exclude]

    correlations = {}
    for col in numeric_cols:
        # Handle NaNs in features by filling with mean for correlation check
        feat_values = val_df[col].fillna(val_df[col].mean())
        if len(feat_values.unique()) > 1:
            corr = np.corrcoef(errors, feat_values)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for name, score in sorted_corr[:5]:
        print(f"{name:<50}: {score:.4f}")


def main():
    # 1. Setup
    seed_everything(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Feature Engineering
    # This generates features for both RF and MLP and saves them to cache
    print("\n--- Step 1: Feature Engineering ---")
    run_feature_engineering(load_cached_data=True)

    # 3. Stream A: Random Forest
    print("\n--- Step 2: Stream A (Random Forest) ---")
    # run_rf_learner loads data internally from cache, trains, and returns preds
    rf_val_preds, rf_test_preds, rf_model = run_rf_learner()

    # 4. Stream B: MLP
    print("\n--- Step 3: Stream B (MLP) ---")
    # Load DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders()

    # Determine input dimension for metadata (from the dataset in the loader)
    # The dataset returns a dict, 'metadata' is one key.
    # We can peek at one batch or the dataset directly.
    input_metadata_dim = train_loader.dataset.metadata.shape[1]
    print(f"MLP Metadata Input Dimension: {input_metadata_dim}")

    # Train MLP
    mlp_model = train_mlp_model(
        train_loader, val_loader, input_metadata_dim, device=device
    )

    # Predict MLP
    print("Generating MLP predictions...")
    mlp_val_preds = predict_mlp(mlp_model, val_loader, device=device)
    mlp_test_preds = predict_mlp(mlp_model, test_loader, device=device)

    # 5. Ensemble & Evaluation
    print("\n--- Step 4: Ensemble & Evaluation ---")

    # Load true labels for validation
    # We can get them from the rf cache or the val csv.
    # RF learner loaded them, but didn't return them. Let's load from cache to be safe/consistent.
    rf_data = load_from_cache("features_rf.npz")
    # Handle 0-d array wrapping if present (helper logic from rf_learner)
    y_val = rf_data["y_val"]
    if isinstance(y_val, np.ndarray) and y_val.ndim == 0:
        y_val = y_val.item()

    # Weighted Ensemble
    w_rf = ENSEMBLE_WEIGHTS["rf"]
    w_mlp = ENSEMBLE_WEIGHTS["mlp"]

    final_val_preds = (w_rf * rf_val_preds) + (w_mlp * mlp_val_preds)

    # Metric
    val_auc = roc_auc_score(y_val, final_val_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    perform_failure_analysis(final_val_preds, y_val)

    # 7. Submission
    threshold = 0.7135451153926904
    if val_auc > threshold:
        print(
            f"\nValidation AUC ({val_auc}) > Threshold ({threshold}). Generating submission..."
        )

        # Ensemble Test Predictions
        final_test_preds = (w_rf * rf_test_preds) + (w_mlp * mlp_test_preds)

        # Load Test IDs
        test_df = pd.read_csv(TEST_CSV)
        request_ids = test_df["request_id"].values

        # Save
        save_submission(request_ids, final_test_preds)
    else:
        print(
            f"\nValidation AUC ({val_auc}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
