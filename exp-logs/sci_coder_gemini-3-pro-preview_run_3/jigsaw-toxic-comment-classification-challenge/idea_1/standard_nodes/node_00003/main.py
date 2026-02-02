import sys
import os
import numpy as np
import pandas as pd

# Import functions and classes from the provided library files
from library.utils import set_seed, get_logger
from library.data_processing import load_data, LABEL_COLS
from library.workflow import train_validate, generate_submission

# Initialize logger
logger = get_logger("runfile")


def main():
    # 1. Configuration
    # We use the full dataset (sample_size=None) as the NBSVM model is computationally
    # efficient (linear time complexity) and will easily complete within the time limit.
    SAMPLE_SIZE = None
    SEED = 42
    C_PARAM = 1.0
    MAX_ITER = 100

    # Set seed for reproducibility across numpy, random, etc.
    set_seed(SEED)

    logger.info("Starting pipeline execution...")

    # 2. Train and Validate
    # The train_validate function handles:
    # - Loading training and validation data (with caching)
    # - Fitting TF-IDF vectorizers (Word + Char n-grams)
    # - Training the Multi-label NBSVM model
    # - Printing initial validation scores
    model, fe = train_validate(
        load_cached_data=True,
        sample_size=SAMPLE_SIZE,
        C=C_PARAM,
        max_iter=MAX_ITER,
        seed=SEED,
    )

    # 3. Validation Metric & Failure Analysis
    # We perform a dedicated validation pass here to:
    # a) Print the metric in the exact format required by the task.
    # b) Calculate correlations for failure analysis.
    logger.info("Performing Failure Analysis and Final Validation...")

    # Load validation data (this will load from the parquet cache created in train_validate)
    val_df = load_data("val", load_cached_data=True)
    if SAMPLE_SIZE:
        val_df = val_df.iloc[:SAMPLE_SIZE]

    # Transform validation text using the fitted FeatureEngineer
    # We must use the same cache suffix logic as train_validate to hit the cache
    suffix = "val_debug" if SAMPLE_SIZE else "val"
    X_val = fe.transform(
        val_df["comment_text"], load_cached_data=True, cache_suffix=suffix
    )

    # Get predictions and true labels
    Y_val = val_df[LABEL_COLS]
    preds = model.predict_proba(X_val)

    # Calculate and print Final Validation Metric
    # model.score calculates the mean column-wise ROC AUC
    final_metric = model.score(X_val, Y_val)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between error magnitude and input features
    # We calculate the Mean Absolute Error (MAE) per sample across all 6 labels
    # Shape: (n_samples, n_labels) -> mean over axis 1 -> (n_samples,)
    errors = np.abs(Y_val.values - preds).mean(axis=1)

    # Generate meta-features (text length) to check for correlation with error
    # This helps identify if the model struggles with longer/shorter comments
    char_lengths = val_df["comment_text"].str.len().fillna(0)
    word_lengths = val_df["comment_text"].apply(lambda x: len(str(x).split()))

    # Calculate Pearson correlation coefficients
    corr_char = np.corrcoef(errors, char_lengths)[0, 1]
    corr_word = np.corrcoef(errors, word_lengths)[0, 1]

    print("-" * 40)
    print("Failure Analysis - Error Correlations:")
    print(f"Correlation (Error vs Char Length): {corr_char:.10f}")
    print(f"Correlation (Error vs Word Length): {corr_word:.10f}")
    print("-" * 40)

    # 4. Generate Submission
    # Generates predictions for the full test set and saves to ./submission/submission.csv
    generate_submission(
        model,
        fe,
        load_cached_data=True,
        sample_size=None,  # Ensure we predict on the full test set
    )

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
