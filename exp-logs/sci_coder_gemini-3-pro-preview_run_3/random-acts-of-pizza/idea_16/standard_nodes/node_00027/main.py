import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import (
    WORKING_DIR,
    SUBMISSION_DIR,
    TARGET_COL,
    ID_COL,
    SEED,
    DENSE_FEATURE_COLS,
    SAMPLE_SUBMISSION_PATH,
)
from library.utils import set_seed, evaluate_auc
from library.data_loader import load_dataset
from library.features import FeatureExtractor
from library.ensemble import PentViewStackingEnsemble


def main():
    # 1. Setup
    print("Initializing Regularized Pent-View Stacking Ensemble Pipeline...")
    set_seed(SEED)

    # 2. Data Loading
    # Load cached data if available to save time
    train_df, val_df, test_df = load_dataset(load_cached_data=True)

    # Extract targets
    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values

    # 3. Feature Extraction
    print("\n--- Feature Extraction ---")
    feature_extractor = FeatureExtractor()

    # Fit on training data
    feature_extractor.fit(train_df)

    # Transform all splits
    # The extractor handles caching internally
    print("Transforming Training Data...")
    X_train_dict = feature_extractor.transform(
        train_df, split_name="train", load_cached_data=True
    )

    print("Transforming Validation Data...")
    X_val_dict = feature_extractor.transform(
        val_df, split_name="val", load_cached_data=True
    )

    print("Transforming Test Data...")
    X_test_dict = feature_extractor.transform(
        test_df, split_name="test", load_cached_data=True
    )

    # 4. Model Training (Stacking)
    print("\n--- Model Training ---")
    ensemble = PentViewStackingEnsemble()

    # Step A: Train Level 1 via CV to get OOF preds and train Level 2 Meta-Learner
    print("Step 1: Cross-Validation & Meta-Learner Training")
    ensemble.fit_oof(X_train_dict, y_train)

    # Step B: Retrain Level 1 on full training data
    print("Step 2: Retraining Base Learners on Full Training Set")
    ensemble.fit_final(X_train_dict, y_train)

    # 5. Validation & Evaluation
    print("\n--- Validation ---")
    # Predict on validation set
    val_preds = ensemble.predict(X_val_dict)

    # Calculate Metric
    val_auc = evaluate_auc(y_val, val_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(y_val - val_preds)

    # We will correlate errors with the dense metadata features
    # We need to retrieve the dense features from the validation dataframe (before scaling)
    # or use the scaled metadata from X_val_dict. Using original DF is more interpretable.

    # Re-derive features to ensure we have the numeric columns used in DENSE_FEATURE_COLS
    # (The extractor computes derived cols like text length internally, so we simulate that check)
    val_analysis_df = val_df.copy()
    if "text" not in val_analysis_df.columns:
        if "request_text_edit_aware" in val_analysis_df.columns:
            val_analysis_df["text"] = (
                val_analysis_df["request_text_edit_aware"].fillna("").astype(str)
            )
        else:
            val_analysis_df["text"] = (
                val_analysis_df["request_text"].fillna("").astype(str)
            )

    if "title" not in val_analysis_df.columns:
        val_analysis_df["title"] = (
            val_analysis_df["request_title"].fillna("").astype(str)
        )

    val_analysis_df["request_text_len_char"] = val_analysis_df["text"].str.len()
    val_analysis_df["request_text_len_word"] = val_analysis_df["text"].apply(
        lambda x: len(str(x).split())
    )
    val_analysis_df["request_title_len_char"] = val_analysis_df["title"].str.len()
    val_analysis_df["request_title_len_word"] = val_analysis_df["title"].apply(
        lambda x: len(str(x).split())
    )

    print("Correlation between Error Magnitude and Features:")
    correlations = []
    for col in DENSE_FEATURE_COLS:
        if col in val_analysis_df.columns:
            # Handle potential NaNs just for analysis
            feat_values = val_analysis_df[col].fillna(0).values
            # Ensure numeric
            if np.issubdtype(feat_values.dtype, np.number):
                corr, _ = pearsonr(errors, feat_values)
                correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for col, corr in correlations:
        print(f"  {col}: {corr:.4f}")

    # 7. Submission
    THRESHOLD = 0.7085870249842536

    if val_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test
        test_preds = ensemble.predict(X_test_dict)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: test_preds})

        # Ensure directory exists
        os.makedirs(SUBMISSION_DIR, exist_ok=True)
        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

        # Save
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

        # Verify format against sample
        if os.path.exists(SAMPLE_SUBMISSION_PATH):
            sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
            print(
                f"Submission shape: {submission_df.shape}, Sample shape: {sample.shape}"
            )
            print("First 5 predictions:")
            print(submission_df.head())
    else:
        print(
            f"\nValidation AUC ({val_auc}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
