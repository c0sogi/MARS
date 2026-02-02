import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import (
    setup_logger,
    load_or_process,
    calculate_accuracy,
    format_submission_id,
)
from library.data_processor import get_data

logger = setup_logger("NgramModel")


class HierarchicalLookupModel:
    """
    A Non-Parametric Hierarchical N-gram model for text normalization.
    It memorizes the most frequent mapping for tokens based on varying context window sizes.
    """

    def __init__(self):
        self.l1_dict = {}  # Unigram: current -> target
        self.l2_left_dict = {}  # Bigram: (prev, current) -> target
        self.l2_right_dict = {}  # Bigram: (current, next) -> target
        self.l3_dict = {}  # Trigram: (prev, current, next) -> target

    def fit(self, df: pd.DataFrame):
        """
        Learns the N-gram mappings from the provided dataframe.
        Expects df to have <BOS> and <EOS> tokens and be sorted by sentence_id, token_id.
        """
        logger.info("Fitting HierarchicalLookupModel...")

        # Ensure string types for input/output to avoid type mismatches
        df[Config.INPUT_COL] = df[Config.INPUT_COL].astype(str)
        df[Config.TARGET_COL] = df[Config.TARGET_COL].astype(str)

        # --- 1. Generate Context ---
        # Since data is sorted (BOS -> Token0 -> ... -> EOS), we can shift to get context.
        # Shift(1) gives previous token, Shift(-1) gives next token.
        logger.info("Generating context columns...")
        df["prev"] = df[Config.INPUT_COL].shift(1).fillna(Config.BOS_TOKEN)
        df["next"] = df[Config.INPUT_COL].shift(-1).fillna(Config.EOS_TOKEN)

        # --- 2. Filter Data ---
        # We only want to learn mappings for actual tokens, not the boundary markers.
        # Boundary markers serve only as context.
        mask = (df[Config.INPUT_COL] != Config.BOS_TOKEN) & (
            df[Config.INPUT_COL] != Config.EOS_TOKEN
        )
        df_clean = df[mask].copy()

        # Clean up original df reference to save memory if possible
        gc.collect()

        # --- 3. Compute Modes ---
        def get_mode_map(data, keys, target):
            """
            Groups by keys and finds the most frequent target value.
            Returns a dictionary mapping key -> best_target.
            """
            logger.info(f"Computing mode map for keys: {keys}")

            # Count occurrences of each (key..., target) tuple
            counts = data.groupby(keys + [target]).size().reset_index(name="count")

            # Sort by count descending so the most frequent appears first
            counts = counts.sort_values("count", ascending=False)

            # Drop duplicates on keys, keeping the first (most frequent)
            best = counts.drop_duplicates(subset=keys)

            # Convert to dictionary
            if len(keys) == 1:
                return dict(zip(best[keys[0]], best[target]))
            else:
                # Create tuple keys for dictionary lookup
                return dict(zip(zip(*[best[k] for k in keys]), best[target]))

        # L3: Trigram (prev, cur, next)
        self.l3_dict = get_mode_map(
            df_clean, ["prev", Config.INPUT_COL, "next"], Config.TARGET_COL
        )

        # L2 Left: Bigram (prev, cur)
        self.l2_left_dict = get_mode_map(
            df_clean, ["prev", Config.INPUT_COL], Config.TARGET_COL
        )

        # L2 Right: Bigram (cur, next)
        self.l2_right_dict = get_mode_map(
            df_clean, [Config.INPUT_COL, "next"], Config.TARGET_COL
        )

        # L1: Unigram (cur)
        self.l1_dict = get_mode_map(df_clean, [Config.INPUT_COL], Config.TARGET_COL)

        logger.info(
            f"Model fitted. Dictionary sizes - L3: {len(self.l3_dict)}, L2L: {len(self.l2_left_dict)}, L2R: {len(self.l2_right_dict)}, L1: {len(self.l1_dict)}"
        )

    def predict(self, df: pd.DataFrame) -> list:
        """
        Predicts normalized text for the given dataframe.
        Expects df to have <BOS> and <EOS> tokens and be sorted.
        Returns a list of predictions corresponding to the non-boundary rows.
        """
        logger.info("Predicting with HierarchicalLookupModel...")

        # Ensure string types
        df[Config.INPUT_COL] = df[Config.INPUT_COL].astype(str)

        # --- 1. Generate Context ---
        df["prev"] = df[Config.INPUT_COL].shift(1).fillna(Config.BOS_TOKEN)
        df["next"] = df[Config.INPUT_COL].shift(-1).fillna(Config.EOS_TOKEN)

        # --- 2. Filter Data ---
        mask = (df[Config.INPUT_COL] != Config.BOS_TOKEN) & (
            df[Config.INPUT_COL] != Config.EOS_TOKEN
        )
        df_clean = df[mask].copy()

        # --- 3. Hierarchical Lookup ---
        # We use pandas Series mapping for vectorized lookup.
        # The 'combine_first' method fills NaNs in the calling series with values from the argument.
        # Order: L3 -> L2 Left -> L2 Right -> L1 -> Identity

        logger.info("Mapping L3 (Trigram)...")
        l3_keys = list(
            zip(df_clean["prev"], df_clean[Config.INPUT_COL], df_clean["next"])
        )
        # Create Series with correct index to ensure alignment
        preds = pd.Series(l3_keys, index=df_clean.index).map(self.l3_dict)

        logger.info("Mapping L2 Left (Bigram)...")
        l2_left_keys = list(zip(df_clean["prev"], df_clean[Config.INPUT_COL]))
        preds_l2_left = pd.Series(l2_left_keys, index=df_clean.index).map(
            self.l2_left_dict
        )
        preds = preds.combine_first(preds_l2_left)

        logger.info("Mapping L2 Right (Bigram)...")
        l2_right_keys = list(zip(df_clean[Config.INPUT_COL], df_clean["next"]))
        preds_l2_right = pd.Series(l2_right_keys, index=df_clean.index).map(
            self.l2_right_dict
        )
        preds = preds.combine_first(preds_l2_right)

        logger.info("Mapping L1 (Unigram)...")
        preds_l1 = df_clean[Config.INPUT_COL].map(self.l1_dict)
        preds = preds.combine_first(preds_l1)

        logger.info("Applying Identity fallback...")
        # If still NaN (unseen token), return the original input
        preds = preds.fillna(df_clean[Config.INPUT_COL])

        return preds.tolist()

    def get_stats(self):
        """Returns the learned dictionaries."""
        return {
            "l1": self.l1_dict,
            "l2_left": self.l2_left_dict,
            "l2_right": self.l2_right_dict,
            "l3": self.l3_dict,
        }

    def load_stats(self, stats):
        """Loads dictionaries from a stats object."""
        self.l1_dict = stats["l1"]
        self.l2_left_dict = stats["l2_left"]
        self.l2_right_dict = stats["l2_right"]
        self.l3_dict = stats["l3"]


def _compute_stats_wrapper():
    """
    Internal wrapper to load training data and fit the model.
    Used by load_or_process for caching.
    """
    # Load processed training data (with BOS/EOS)
    df_train = get_data("train", load_cached_data=True)

    model = HierarchicalLookupModel()
    model.fit(df_train)

    return model.get_stats()


def train_model(load_cached_data=True):
    """
    Trains the HierarchicalLookupModel or loads cached statistics.

    Args:
        load_cached_data (bool): If True, attempts to load pre-computed stats.

    Returns:
        HierarchicalLookupModel: The trained model instance.
    """
    # Use load_or_process to handle caching of the model statistics (dictionaries)
    stats = load_or_process(
        Config.MODEL_STATS_PATH,
        _compute_stats_wrapper,
        load_cached_data=load_cached_data,
    )

    model = HierarchicalLookupModel()
    model.load_stats(stats)
    return model


def evaluate_model(model, df_val=None):
    """
    Evaluates the model on the validation set.

    Args:
        model: The trained model.
        df_val: Optional validation dataframe. If None, loads from config.

    Returns:
        float: Accuracy score.
    """
    if df_val is None:
        df_val = get_data("val", load_cached_data=True)

    logger.info("Evaluating model on validation set...")
    y_pred = model.predict(df_val)

    # Extract ground truth
    # Filter boundary tokens to match prediction output
    mask = (df_val[Config.INPUT_COL] != Config.BOS_TOKEN) & (
        df_val[Config.INPUT_COL] != Config.EOS_TOKEN
    )
    y_true = df_val.loc[mask, Config.TARGET_COL].astype(str).tolist()

    acc = calculate_accuracy(y_true, y_pred)
    logger.info(f"Validation Accuracy: {acc}")
    return acc


def generate_submission(model):
    """
    Generates predictions for the test set and saves the submission file.
    """
    logger.info("Generating submission...")
    df_test = get_data("test", load_cached_data=True)

    # Generate predictions
    y_pred = model.predict(df_test)

    # Prepare submission dataframe
    # Filter test df to get IDs (excluding boundaries)
    mask = (df_test[Config.INPUT_COL] != Config.BOS_TOKEN) & (
        df_test[Config.INPUT_COL] != Config.EOS_TOKEN
    )
    df_sub = df_test.loc[mask].copy()

    # Create submission ID: sentence_id + "_" + token_id
    # Using vectorized string concatenation for speed
    df_sub[Config.SUBMISSION_ID_COL] = (
        df_sub[Config.SENTENCE_ID_COL].astype(str)
        + "_"
        + df_sub[Config.TOKEN_ID_COL].astype(str)
    )

    # Assign predictions
    df_sub[Config.TARGET_COL] = y_pred

    # Select required columns
    submission = df_sub[[Config.SUBMISSION_ID_COL, Config.TARGET_COL]]

    # Save to CSV
    logger.info(f"Saving submission to {Config.SUBMISSION_PATH}")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
