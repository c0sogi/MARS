import os
import pandas as pd
import numpy as np
from library.config import (
    MODEL_CACHE_PATH,
    COL_ID,
    COL_AFTER,
    COL_BEFORE,
    WORKING_DIR,
    SUBMISSION_FILE_PATH,
)
from library.utils import setup_logger, save_submission
from library.data_loader import load_and_process_data


class HFBBModel:
    """
    Hierarchical Frequency-Based Backoff Model.

    Layers:
    1. Trigram Context: (prev, curr, next) -> after
    2. Unigram Context: (curr) -> after
    3. Identity: curr -> curr
    """

    def __init__(self):
        self.trigram_map = None
        self.unigram_map = None
        self.logger = setup_logger("HFBBModel")

        # Define cache paths based on the config's base path
        # We append extensions to handle them as parquet files
        self.trigram_path = f"{MODEL_CACHE_PATH}.trigram.parquet"
        self.unigram_path = f"{MODEL_CACHE_PATH}.unigram.parquet"

    def fit(self, df):
        """
        Trains the model by aggregating counts and selecting the mode.
        """
        self.logger.info("Fitting Trigram Layer...")
        # Count occurrences of each target for every trigram context
        tri_counts = (
            df.groupby(["prev", "curr", "next", "after"])
            .size()
            .reset_index(name="count")
        )
        # Sort by count descending so the most frequent is first
        tri_counts.sort_values(by="count", ascending=False, inplace=True)
        # Drop duplicates to keep only the most frequent target for each key
        self.trigram_map = tri_counts.drop_duplicates(subset=["prev", "curr", "next"])[
            ["prev", "curr", "next", "after"]
        ]
        self.logger.info(f"Trigram Layer fitted. Unique keys: {len(self.trigram_map)}")

        self.logger.info("Fitting Unigram Layer...")
        # Count occurrences of each target for every unigram context
        uni_counts = df.groupby(["curr", "after"]).size().reset_index(name="count")
        uni_counts.sort_values(by="count", ascending=False, inplace=True)
        self.unigram_map = uni_counts.drop_duplicates(subset=["curr"])[
            ["curr", "after"]
        ]
        self.logger.info(f"Unigram Layer fitted. Unique keys: {len(self.unigram_map)}")

    def predict(self, df):
        """
        Predicts normalized text using the backoff strategy.
        Returns a Series of predictions.
        """
        self.logger.info("Starting prediction...")

        # Ensure input columns exist
        req_cols = ["prev", "curr", "next"]
        if not all(c in df.columns for c in req_cols):
            raise ValueError(f"Input dataframe must contain columns: {req_cols}")

        # 1. Trigram Lookup
        # Perform a left merge to find matches
        # Rename 'after' to 'pred_tri'
        self.logger.info("Applying Trigram Layer...")
        pred_df = (
            df[req_cols]
            .merge(self.trigram_map, on=["prev", "curr", "next"], how="left")
            .rename(columns={"after": "pred_tri"})
        )

        # 2. Unigram Lookup
        self.logger.info("Applying Unigram Layer...")
        pred_df = pred_df.merge(self.unigram_map, on=["curr"], how="left").rename(
            columns={"after": "pred_uni"}
        )

        # 3. Backoff Logic
        # Coalesce: Trigram -> Unigram -> Identity (curr)
        self.logger.info("Applying Backoff Logic...")
        predictions = (
            pred_df["pred_tri"].fillna(pred_df["pred_uni"]).fillna(pred_df["curr"])
        )

        return predictions

    def save(self):
        """
        Saves the lookup tables to Parquet.
        """
        self.logger.info(f"Saving model to {WORKING_DIR}...")
        if self.trigram_map is not None:
            self.trigram_map.to_parquet(self.trigram_path, index=False)
        if self.unigram_map is not None:
            self.unigram_map.to_parquet(self.unigram_path, index=False)
        self.logger.info("Model saved.")

    def load(self):
        """
        Loads the lookup tables from Parquet.
        Returns True if successful, False otherwise.
        """
        if os.path.exists(self.trigram_path) and os.path.exists(self.unigram_path):
            self.logger.info("Loading model from cache...")
            self.trigram_map = pd.read_parquet(self.trigram_path)
            self.unigram_map = pd.read_parquet(self.unigram_path)
            self.logger.info("Model loaded successfully.")
            return True
        return False


def run_task(load_cached_data=True, train_limit=None):
    """
    Main driver function to execute the pipeline:
    1. Load Data
    2. Train/Load Model
    3. Evaluate on Validation
    4. Generate Submission
    """
    logger = setup_logger("TaskRunner")
    model = HFBBModel()

    # ==========================================
    # 1. Model Training / Loading
    # ==========================================
    # Check if we can load a pre-trained model
    model_loaded = False
    if load_cached_data:
        model_loaded = model.load()

    if not model_loaded:
        logger.info("No cached model found or cache disabled. Training from scratch...")

        # Load Training Data
        df_train = load_and_process_data(
            split="train", load_cached_data=load_cached_data, limit=train_limit
        )

        # Fit Model
        model.fit(df_train)

        # Save Model
        model.save()

        # Free memory
        del df_train

    # ==========================================
    # 2. Validation
    # ==========================================
    logger.info("Loading validation data...")
    df_val = load_and_process_data(split="val", load_cached_data=load_cached_data)

    logger.info("Predicting on validation set...")
    val_preds = model.predict(df_val)

    # Calculate Accuracy
    # Exact string match required
    correct = (val_preds == df_val["after"]).sum()
    total = len(df_val)
    accuracy = correct / total

    print(f"Validation Accuracy: {accuracy}")

    # Free memory
    del df_val, val_preds

    # ==========================================
    # 3. Submission Generation
    # ==========================================
    logger.info("Loading test data...")
    df_test = load_and_process_data(split="test", load_cached_data=load_cached_data)

    logger.info("Predicting on test set...")
    test_preds = model.predict(df_test)

    # Prepare submission dataframe
    submission_df = pd.DataFrame({COL_ID: df_test[COL_ID], COL_AFTER: test_preds})

    logger.info(f"Saving submission to {SUBMISSION_FILE_PATH}...")
    save_submission(submission_df, filepath=SUBMISSION_FILE_PATH)

    logger.info("Task completed successfully.")
