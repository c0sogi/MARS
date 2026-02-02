import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import load_data_splits
from library.features import process_and_cache_data
from library.models import MultiViewEnsemble


def main():
    # 1. Setup Environment
    set_seed(Config.SEED)
    logger = setup_logger("runfile")

    logger.info("Starting pipeline execution...")

    # 2. Load Data
    # Loading full splits as dataset size is manageable within time constraints
    train_df, val_df, test_df = load_data_splits()

    # 3. Feature Processing
    # This step generates TF-IDF, Embeddings (GPU accelerated if available), and Dense features
    # Caching is enabled to optimize runtime
    (X_train, y_train), (X_val, y_val), (X_test, _) = process_and_cache_data(
        train_df, val_df, test_df, load_cached_data=True
    )

    # 4. Model Training
    logger.info("Initializing and training MultiViewEnsemble...")
    model = MultiViewEnsemble()
    model.fit(X_train, y_train, X_val, y_val)

    # 5. Validation
    logger.info("Performing final validation inference...")
    val_probs = model.predict_proba(X_val)
    val_auc = roc_auc_score(y_val, val_probs)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    logger.info("Running failure analysis...")
    # Calculate absolute error
    errors = np.abs(y_val - val_probs)

    # Map dense feature indices to names
    dense_feature_names = Config.NUMERICAL_COLS + Config.DERIVED_NUMERICAL_COLS
    dense_matrix = X_val["dense"]

    correlations = []
    # Calculate correlation between error and each dense feature
    if dense_matrix.shape[1] == len(dense_feature_names):
        for i, feature_name in enumerate(dense_feature_names):
            feature_values = dense_matrix[:, i]
            # Check for constant values to avoid warnings
            if np.std(feature_values) > 1e-9:
                corr, _ = pearsonr(feature_values, errors)
                correlations.append((feature_name, corr))
            else:
                correlations.append((feature_name, 0.0))
    else:
        logger.warning(
            "Mismatch between dense feature matrix columns and config names. Skipping detailed feature correlation."
        )

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\n--- Failure Analysis: Top Features Correlated with Error ---")
    for name, corr in correlations[:5]:
        print(f"Feature: {name}, Correlation: {corr:.4f}")
    print("------------------------------------------------------------\n")

    # 7. Submission
    threshold = 0.6594098741904747

    if val_auc > threshold:
        logger.info(
            f"Validation AUC ({val_auc}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Inference on test set
        test_probs = model.predict_proba(X_test)

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {
                "request_id": test_df["request_id"],
                "requester_received_pizza": test_probs,
            }
        )

        # Save to disk
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
    else:
        logger.warning(
            f"Validation AUC ({val_auc}) did not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
