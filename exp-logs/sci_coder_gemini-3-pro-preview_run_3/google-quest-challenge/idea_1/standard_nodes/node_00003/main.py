import sys
import os
import numpy as np
import pandas as pd
import torch
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    SUBMISSION_PATH,
    TARGET_COLS,
    SEED,
    TEXT_COLS,
)
from library.data_loader import load_dataset, prepare_text_pairs, get_targets, get_ids
from library.feature_extractor import EmbeddingPipeline
from library.model import LinearHead
from library.utils import compute_spearman_score, save_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=SEED):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def perform_failure_analysis(y_true, y_pred, df_val):
    """
    Analyzes correlations between error magnitude and interpretable meta-features.
    """
    print("\nPerforming Failure Analysis...")

    # Calculate Mean Absolute Error per sample (averaged across all 30 targets)
    # Shape: (N,)
    mae_per_sample = np.mean(np.abs(y_true - y_pred), axis=1)

    # Create meta-features for analysis
    # We use text length as a proxy for input complexity
    q_len = (
        df_val["question_title"].fillna("") + " " + df_val["question_body"].fillna("")
    ).str.len()
    a_len = df_val["answer"].fillna("").str.len()

    analysis_df = pd.DataFrame(
        {
            "error_magnitude": mae_per_sample,
            "question_length": q_len,
            "answer_length": a_len,
        }
    )

    # Calculate Spearman correlations between error and meta-features
    correlations = analysis_df.corr(method="spearman")["error_magnitude"].drop(
        "error_magnitude"
    )

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # Identify the targets with the highest error
    mae_per_target = np.mean(np.abs(y_true - y_pred), axis=0)
    hardest_idx = np.argsort(mae_per_target)[-3:][::-1]
    print("\nTop 3 Targets with highest MAE:")
    for idx in hardest_idx:
        print(f"  {TARGET_COLS[idx]}: {mae_per_target[idx]:.4f}")


def main():
    # 1. Setup
    set_seed()
    print("Starting runfile.py execution...")

    # 2. Load Data
    print("Loading datasets...")
    train_df = load_dataset(TRAIN_DATA_PATH)
    val_df = load_dataset(VAL_DATA_PATH)
    test_df = load_dataset(TEST_DATA_PATH)

    # 3. Prepare Text Pairs
    # Using load_cached_data=True to utilize pre-computed data if available
    print("Preparing text pairs...")
    train_q, train_a = prepare_text_pairs(train_df, "train", load_cached_data=True)
    val_q, val_a = prepare_text_pairs(val_df, "val", load_cached_data=True)
    test_q, test_a = prepare_text_pairs(test_df, "test", load_cached_data=True)

    # 4. Feature Extraction
    # Initialize pipeline (automatically detects and uses GPU)
    pipeline = EmbeddingPipeline()

    print("Extracting features (Embeddings + Interactions)...")
    X_train = pipeline.get_features(train_q, train_a, "train", load_cached_data=True)
    X_val = pipeline.get_features(val_q, val_a, "val", load_cached_data=True)
    X_test = pipeline.get_features(test_q, test_a, "test", load_cached_data=True)

    # 5. Get Targets
    y_train = get_targets(train_df)
    y_val = get_targets(val_df)

    # 6. Train Model
    print("Initializing and training model...")
    model = LinearHead()
    # Fit on training data
    model.fit(X_train, y_train)

    # Save the trained model
    model.save("linear_head.joblib")

    # 7. Validation
    print("Validating...")
    y_val_pred = model.predict(X_val)

    # Compute and print the required metric
    # The metric is Mean Column-wise Spearman's Correlation
    val_score = compute_spearman_score(y_val, y_val_pred, TARGET_COLS)
    print(f"Final Validation Metric: {val_score}")

    # 8. Failure Analysis
    perform_failure_analysis(y_val, y_val_pred, val_df)

    # 9. Submission
    baseline_score = 0.3001487994080418
    if val_score > baseline_score:
        print("Generating submission...")
        y_test_pred = model.predict(X_test)
        test_ids = get_ids(test_df)

        save_submission(y_test_pred, test_ids, TARGET_COLS, SUBMISSION_PATH)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"Validation score {val_score} did not beat baseline {baseline_score}. Skipping submission."
        )


if __name__ == "__main__":
    main()
