import os
import sys
import numpy as np
import pandas as pd
import random
import torch
import warnings
from sklearn.metrics import roc_auc_score

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.feature_extractor import HybridFeatureProcessor
from library.model_wrapper import PizzaRandomForest

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(Config.RANDOM_SEED)

    # 2. Data Processing
    # Initialize processor
    processor = HybridFeatureProcessor()

    # Process data (loads from cache if available, otherwise processes)
    # This handles loading raw data, aligning columns, generating embeddings, and feature engineering
    print("Processing data...")
    train_df, val_df, test_df = processor.process_data(load_cached_data=True)

    # 3. Prepare Features and Targets
    target_col = "requester_received_pizza"
    id_col = "request_id"

    def prepare_xy(df, is_test=False):
        # Drop ID column
        cols_to_drop = [id_col]
        y = None

        # Handle Target
        if not is_test:
            if target_col in df.columns:
                y = df[target_col].values
                cols_to_drop.append(target_col)
            else:
                raise ValueError(
                    f"Target column {target_col} missing from training/val data"
                )

        # Drop columns to get X
        # The processor ensures X contains numeric features and embeddings
        X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        return X, y

    X_train, y_train = prepare_xy(train_df)
    X_val, y_val = prepare_xy(val_df)
    X_test, _ = prepare_xy(test_df, is_test=True)

    # 4. Model Training
    model = PizzaRandomForest()
    # Train the model using the wrapper
    model.train(X_train, y_train, X_val, y_val)

    # 5. Validation & Metrics
    # Predict on validation set
    val_probs = model.predict_proba(X_val)

    # Calculate AUC
    final_val_metric = roc_auc_score(y_val, val_probs)
    print(f"Final Validation Metric: {final_val_metric}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(y_val - val_probs)

    # Calculate correlation between each feature and the error
    feature_correlations = []

    # X_val is a DataFrame, so we can iterate columns
    for col in X_val.columns:
        try:
            # Ensure column is numeric
            if pd.api.types.is_numeric_dtype(X_val[col]):
                # Calculate Pearson correlation
                corr = np.corrcoef(X_val[col], errors)[0, 1]
                if not np.isnan(corr):
                    feature_correlations.append((col, corr))
        except Exception:
            continue

    # Sort by absolute correlation
    feature_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 features correlated with prediction error:")
    for feature, corr in feature_correlations[:10]:
        print(f"{feature}: {corr:.6f}")

    # 7. Submission
    threshold = 0.648621586265928

    if final_val_metric > threshold:
        print(
            f"\nValidation metric {final_val_metric} exceeds threshold {threshold}. Generating submission..."
        )

        # Predict on test set
        test_probs = model.predict_proba(X_test)

        # Create submission DataFrame
        submission = pd.DataFrame(
            {"request_id": test_df[id_col], "requester_received_pizza": test_probs}
        )

        # Ensure directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Save submission
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric {final_val_metric} does not exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
