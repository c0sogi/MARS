import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import load_datasets
from library.features import FeatureExtractor
from library.model import ToxicityClassifier
from library.evaluation import compute_score


def main():
    # 1. Setup
    set_seed(Config.SEED)
    logger = setup_logger("main")
    logger.info("Starting Runfile Execution...")

    # 2. Load Data
    # We use load_cached_data=True to utilize the parquet cache if available
    logger.info("Loading datasets...")
    train_df, val_df, test_df = load_datasets(load_cached_data=True)

    # 3. Feature Extraction
    # Extract TF-IDF features (Word + Char n-grams)
    extractor = FeatureExtractor()
    X_train, X_val, X_test = extractor.extract_features(
        train_df, val_df, test_df, load_cached_data=True
    )

    # 4. Model Training
    # Initialize and train the One-Vs-Rest Logistic Regression Classifier
    classifier = ToxicityClassifier()

    # The train method returns the mean AUC on the validation set calculated during training
    # However, for the sake of the explicit requirement to print the metric,
    # we can also re-evaluate or just use the returned value.
    # We pass the full dataframes for labels, the classifier handles extraction.
    mean_auc = classifier.train(X_train, train_df, X_val, val_df)

    # 5. Validation Metric
    # Requirement: Print the final validation metric in specific format
    print(f"Final Validation Metric: {mean_auc}")

    # 6. Failure Analysis
    logger.info("\n=== Starting Failure Analysis ===")

    # Generate predictions on validation set for analysis
    val_probs_df = classifier.predict_proba(X_val)

    # Calculate Mean Absolute Error (MAE) per sample across all 6 labels
    # We compare predicted probabilities (val_probs_df) with ground truth (val_df[target_cols])
    target_cols = Config.TARGET_COLS
    y_true = val_df[target_cols].values
    y_pred = val_probs_df[target_cols].values

    # Absolute error per label, then mean across labels for each sample
    errors = np.abs(y_true - y_pred)
    mean_errors = np.mean(errors, axis=1)

    # Compute meta-features for correlation analysis
    # We use the raw text from val_df
    val_text = val_df["comment_text"].astype(str)
    char_lengths = val_text.apply(len).values
    word_counts = val_text.apply(lambda x: len(x.split())).values

    # Calculate correlations
    corr_char, _ = pearsonr(mean_errors, char_lengths)
    corr_word, _ = pearsonr(mean_errors, word_counts)

    logger.info(f"Correlation between Error and Char Length: {corr_char:.4f}")
    logger.info(f"Correlation between Error and Word Count:  {corr_word:.4f}")

    if abs(corr_char) < 0.1 and abs(corr_word) < 0.1:
        logger.info("Result: Error is largely independent of comment length.")
    else:
        logger.info("Result: Error shows some dependency on comment length.")

    logger.info("=== Failure Analysis Complete ===\n")

    # 7. Generate Submission
    # Predict on test set and save to disk
    # Requirement: Only generate submission if metric > 0.9837638458604258
    BASELINE_METRIC = 0.9837638458604258

    if mean_auc > BASELINE_METRIC:
        logger.info(
            f"Validation Metric {mean_auc} > {BASELINE_METRIC}. Generating submission file..."
        )
        classifier.generate_submission(X_test, test_df["id"])
    else:
        logger.info(
            f"Validation Metric {mean_auc} <= {BASELINE_METRIC}. Skipping submission generation."
        )

    logger.info("Runfile Execution Complete.")


if __name__ == "__main__":
    main()
