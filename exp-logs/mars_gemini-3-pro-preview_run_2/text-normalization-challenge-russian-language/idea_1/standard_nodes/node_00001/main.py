import os
import sys
import pandas as pd
import numpy as np
import random
import logging

# Import from provided library
from library.config import (
    SEED,
    COL_ID,
    COL_AFTER,
    COL_BEFORE,
    COL_SENTENCE_ID,
    COL_TOKEN_ID,
    VAL_DATA_PATH,
)
from library.utils import setup_logger, save_submission
from library.data_loader import load_and_process_data
from library.model import HFBBModel


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(SEED)
    logger = setup_logger("RunFile")
    logger.info("Starting pipeline...")

    # 2. Model Initialization
    model = HFBBModel()

    # 3. Training
    # We use load_cached_data=True to use the parquet cache if available
    logger.info("Loading training data...")
    df_train = load_and_process_data(split="train", load_cached_data=True)

    logger.info("Training model...")
    model.fit(df_train)

    logger.info("Saving model...")
    model.save()

    # Free memory
    del df_train
    import gc

    gc.collect()

    # 4. Validation
    logger.info("Loading validation data...")
    df_val = load_and_process_data(split="val", load_cached_data=True)

    logger.info("Running validation inference...")
    val_preds = model.predict(df_val)

    # Calculate Metric
    # Ensure strict string comparison
    y_true = df_val[COL_AFTER].fillna("").astype(str)
    y_pred = val_preds.fillna("").astype(str)

    correct_mask = y_true == y_pred
    accuracy = correct_mask.mean()

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {accuracy}")

    # 5. Failure Analysis
    logger.info("Performing failure analysis...")

    # Create analysis dataframe
    df_analysis = df_val.copy()
    df_analysis["pred"] = y_pred
    df_analysis["is_error"] = (~correct_mask).astype(int)
    df_analysis["input_len"] = df_analysis["curr"].str.len()

    # Load raw validation metadata to get 'class' column for analysis
    # We must replicate the sorting logic from data_loader to ensure alignment
    try:
        df_meta_val = pd.read_csv(VAL_DATA_PATH)
        df_meta_val[COL_BEFORE] = df_meta_val[COL_BEFORE].fillna("").astype(str)
        df_meta_val.sort_values(by=[COL_SENTENCE_ID, COL_TOKEN_ID], inplace=True)
        df_meta_val.reset_index(drop=True, inplace=True)

        # Attach class if lengths match
        if len(df_meta_val) == len(df_analysis):
            df_analysis["class"] = df_meta_val["class"]

            # Error rate by class
            logger.info("Error Rate by Class:")
            class_errors = df_analysis.groupby("class")["is_error"].agg(
                ["count", "mean"]
            )
            class_errors.sort_values("mean", ascending=False, inplace=True)
            print(class_errors.head(10))
        else:
            logger.warning("Metadata length mismatch. Skipping class-based analysis.")

    except Exception as e:
        logger.warning(f"Could not load metadata for analysis: {e}")

    # Correlation analysis
    # Correlation between input length and error probability
    corr = df_analysis["input_len"].corr(df_analysis["is_error"])
    print(f"Correlation (Input Length vs Error): {corr}")

    # Free memory
    del df_val, df_analysis, y_true, y_pred, val_preds
    gc.collect()

    # 6. Submission
    logger.info("Loading test data...")
    df_test = load_and_process_data(split="test", load_cached_data=True)

    logger.info("Running test inference...")
    test_preds = model.predict(df_test)

    logger.info("Preparing submission...")
    submission_df = pd.DataFrame({COL_ID: df_test[COL_ID], COL_AFTER: test_preds})

    # Save submission
    save_submission(submission_df)
    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
