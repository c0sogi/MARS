import sys
import os
import torch
import numpy as np
import pandas as pd
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score

from library.config import Config, set_seed
from library.data_loader import get_processed_features
from library.dataset import create_dataloaders
from library.neural_net import train_mlp_model, predict_mlp
from library.tree_model import train_rf_model, predict_rf
from library.utils import save_submission


# Context manager to suppress stdout for cleaner execution
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def identify_feature_name(idx):
    """Maps feature index to feature name based on construction logic."""
    vocab_size = Config.VOCAB_SIZE_TFIDF

    if idx < vocab_size:
        return f"TF-IDF Token {idx}"

    # Offset by vocab size
    rem = idx - vocab_size

    # Order in RF Matrix:
    # [TFIDF (5000), Num (6), Ratio (1), TopK (50), Cons (2), Inter (2)]

    if rem < 6:
        names = [
            "Account Age",
            "Days Since First RAOP",
            "Requester Comments",
            "Requester Posts",
            "Upvotes Minus Downvotes",
            "Upvotes Plus Downvotes",
        ]
        return names[rem]
    rem -= 6

    if rem < 1:
        return "Upvote Ratio"
    rem -= 1

    if rem < Config.TOP_K_SUBREDDITS:
        return f"Top-K Subreddit Indicator {rem}"
    rem -= Config.TOP_K_SUBREDDITS

    if rem < 2:
        return f"Consistency Scalar {'Title' if rem==0 else 'Body'}"
    rem -= 2

    if rem < 2:
        return f"Interaction Feature {rem}"

    return f"Unknown Feature {idx}"


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Data
    # We load the dictionary for RF and general usage
    data = get_processed_features(load_cached_data=True)

    # 3. Train Random Forest (Stream A)
    # Suppress verbose output from internal function
    rf_model, _ = train_rf_model(data, verbose=False)

    # RF Validation Inference
    rf_val_probs = rf_model.predict_proba(data["rf_val"]["X"])

    # 4. Train MLP (Stream B)
    # Create dataloaders
    # Note: create_dataloaders loads data internally, but since we use caching, overhead is low.
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=Config.MLP_BATCH_SIZE, load_cached_data=True
    )

    # Determine input dimensions from a sample batch
    sample_batch = next(iter(train_loader))
    input_dims = {
        "metadata_dim": sample_batch["metadata_dense"].shape[1],
        "skip_dim": sample_batch["metadata_skip"].shape[1],
    }

    # Train MLP with suppressed logs
    with suppress_stdout():
        mlp_model = train_mlp_model(
            train_loader,
            val_loader,
            input_dims,
            device,
            epochs=Config.MLP_EPOCHS,
            patience=Config.MLP_PATIENCE,
        )

    # MLP Validation Inference
    mlp_val_probs = predict_mlp(mlp_model, val_loader, device)

    # 5. Ensemble & Metric Calculation
    y_val = data["y_val"]
    ensemble_val_probs = 0.5 * rf_val_probs + 0.5 * mlp_val_probs
    final_val_auc = roc_auc_score(y_val, ensemble_val_probs)

    print(f"Final Validation Metric: {final_val_auc}")

    # 6. Failure Analysis
    errors = np.abs(y_val - ensemble_val_probs)
    X_val = data["rf_val"]["X"]

    # Calculate correlation between error and each feature
    # We iterate through features. For efficiency with 5000+ features, we use matrix operations where possible
    # or just simple loop since N_val is small (576).

    # Center data for correlation: (x - mean_x) . (y - mean_y) / (std_x * std_y * n)
    # Using np.corrcoef is safer for individual columns

    correlations = []
    # Check top features (Metadata, Interactions) and a sample of TF-IDF/TopK
    # Actually, let's check all non-constant columns

    for i in range(X_val.shape[1]):
        col_data = X_val[:, i]
        if np.std(col_data) > 1e-9:  # Skip constant columns
            corr = np.corrcoef(col_data, errors)[0, 1]
            if not np.isnan(corr):
                correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Failure Analysis - Top 5 Features Correlated with Error:")
    for idx, corr in correlations[:5]:
        feat_name = identify_feature_name(idx)
        print(f"{feat_name} (Idx {idx}): {corr:.4f}")

    # 7. Submission
    threshold = 0.7135451153926904
    if final_val_auc > threshold:
        # RF Test Inference
        rf_test_probs = predict_rf(rf_model, data)

        # MLP Test Inference
        mlp_test_probs = predict_mlp(mlp_model, test_loader, device)

        # Ensemble
        final_test_probs = 0.5 * rf_test_probs + 0.5 * mlp_test_probs

        save_submission(data["ids_test"], final_test_probs)


if __name__ == "__main__":
    main()
