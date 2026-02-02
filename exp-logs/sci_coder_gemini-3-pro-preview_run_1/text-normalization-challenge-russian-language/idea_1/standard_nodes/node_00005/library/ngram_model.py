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
import torch
from library.neural_model import (
    train_neural_module,
    batch_predict_neural,
    Seq2SeqModel,
    CharTokenizer,
)

logger = setup_logger("NgramModel")


class HierarchicalLookupModel:
    """
    A Hybrid Neuro-Symbolic model for text normalization.
    It uses Hierarchical N-gram lookup for frequent tokens and a Neural Seq2Seq model for rare/complex tokens.
    """

    def __init__(self):
        self.l1_dict = {}  # Unigram: current -> target
        self.l2_left_dict = {}  # Bigram: (prev, current) -> target
        self.l2_right_dict = {}  # Bigram: (current, next) -> target
        self.l3_dict = {}  # Trigram: (prev, current, next) -> target

        # Neural components
        self.neural_model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, df: pd.DataFrame):
        """
        Learns the N-gram mappings and trains the Neural fallback model.
        """
        logger.info("Fitting HierarchicalLookupModel...")

        # Ensure string types
        df[Config.INPUT_COL] = df[Config.INPUT_COL].astype(str)
        df[Config.TARGET_COL] = df[Config.TARGET_COL].astype(str)

        # --- 1. Generate Context ---
        logger.info("Generating context columns...")
        df["prev"] = df[Config.INPUT_COL].shift(1).fillna(Config.BOS_TOKEN)
        df["next"] = df[Config.INPUT_COL].shift(-1).fillna(Config.EOS_TOKEN)

        # --- 2. Filter Data ---
        mask = (df[Config.INPUT_COL] != Config.BOS_TOKEN) & (
            df[Config.INPUT_COL] != Config.EOS_TOKEN
        )
        df_clean = df[mask].copy()
        gc.collect()

        # --- 3. Compute Modes (N-gram) ---
        def get_mode_map(data, keys, target):
            counts = data.groupby(keys + [target]).size().reset_index(name="count")
            counts = counts.sort_values("count", ascending=False)
            best = counts.drop_duplicates(subset=keys)
            if len(keys) == 1:
                return dict(zip(best[keys[0]], best[target]))
            else:
                return dict(zip(zip(*[best[k] for k in keys]), best[target]))

        logger.info("Building N-gram dictionaries...")
        self.l3_dict = get_mode_map(
            df_clean, ["prev", Config.INPUT_COL, "next"], Config.TARGET_COL
        )
        self.l2_left_dict = get_mode_map(
            df_clean, ["prev", Config.INPUT_COL], Config.TARGET_COL
        )
        self.l2_right_dict = get_mode_map(
            df_clean, [Config.INPUT_COL, "next"], Config.TARGET_COL
        )
        self.l1_dict = get_mode_map(df_clean, [Config.INPUT_COL], Config.TARGET_COL)

        logger.info(f"N-gram stats - L3: {len(self.l3_dict)}, L1: {len(self.l1_dict)}")

        # --- 4. Train Neural Model ---
        logger.info("Training Neural Fallback Model...")
        self.neural_model, self.tokenizer = train_neural_module(
            df_clean, device_name=self.device
        )

    def predict(self, df: pd.DataFrame) -> list:
        """
        Predicts normalized text using N-gram lookup with Neural fallback.
        """
        logger.info("Predicting with Hybrid Model...")

        df[Config.INPUT_COL] = df[Config.INPUT_COL].astype(str)
        df["prev"] = df[Config.INPUT_COL].shift(1).fillna(Config.BOS_TOKEN)
        df["next"] = df[Config.INPUT_COL].shift(-1).fillna(Config.EOS_TOKEN)

        mask = (df[Config.INPUT_COL] != Config.BOS_TOKEN) & (
            df[Config.INPUT_COL] != Config.EOS_TOKEN
        )
        df_clean = df[mask].copy()

        # --- N-gram Lookup ---
        logger.info("Applying N-gram Lookup...")
        l3_keys = list(
            zip(df_clean["prev"], df_clean[Config.INPUT_COL], df_clean["next"])
        )
        preds = pd.Series(l3_keys, index=df_clean.index).map(self.l3_dict)

        l2_left_keys = list(zip(df_clean["prev"], df_clean[Config.INPUT_COL]))
        preds = preds.combine_first(
            pd.Series(l2_left_keys, index=df_clean.index).map(self.l2_left_dict)
        )

        l2_right_keys = list(zip(df_clean[Config.INPUT_COL], df_clean["next"]))
        preds = preds.combine_first(
            pd.Series(l2_right_keys, index=df_clean.index).map(self.l2_right_dict)
        )

        preds = preds.combine_first(df_clean[Config.INPUT_COL].map(self.l1_dict))

        # --- Neural Fallback ---
        # Identify missing predictions
        missing_mask = preds.isna()
        missing_count = missing_mask.sum()

        if missing_count > 0 and self.neural_model is not None:
            logger.info(f"Applying Neural Fallback for {missing_count} tokens...")

            # Filter: Only apply neural model to tokens containing digits or symbols (heuristic)
            # This prevents over-normalizing rare proper nouns which should be Identity.
            # We assume proper nouns don't contain digits.
            tokens_to_predict = df_clean.loc[missing_mask, Config.INPUT_COL]

            # Regex: contains digit
            digit_mask = tokens_to_predict.str.contains(r"\d", regex=True)
            neural_candidates = tokens_to_predict[digit_mask]

            if len(neural_candidates) > 0:
                logger.info(
                    f"Running Neural Model on {len(neural_candidates)} candidates (containing digits)..."
                )
                neural_preds_list = batch_predict_neural(
                    self.neural_model,
                    self.tokenizer,
                    neural_candidates.tolist(),
                    device_name=self.device,
                )

                # Update predictions
                neural_preds_series = pd.Series(
                    neural_preds_list, index=neural_candidates.index
                )
                preds.update(neural_preds_series)

        # --- Identity Fallback ---
        logger.info("Applying Identity fallback...")
        preds = preds.fillna(df_clean[Config.INPUT_COL])

        return preds.tolist()

    def get_stats(self):
        """Returns the learned dictionaries and neural model state."""
        stats = {
            "l1": self.l1_dict,
            "l2_left": self.l2_left_dict,
            "l2_right": self.l2_right_dict,
            "l3": self.l3_dict,
        }
        if self.neural_model:
            stats["neural_state"] = self.neural_model.state_dict()
            stats["tokenizer"] = self.tokenizer
            # Save hyperparams to reconstruct model
            stats["model_config"] = {
                "vocab_size": len(self.tokenizer),
                "embed_dim": 64,
                "hidden_dim": 256,
            }
        return stats

    def load_stats(self, stats):
        """Loads dictionaries and neural model."""
        self.l1_dict = stats["l1"]
        self.l2_left_dict = stats["l2_left"]
        self.l2_right_dict = stats["l2_right"]
        self.l3_dict = stats["l3"]

        if "neural_state" in stats:
            logger.info("Loading Neural Model from stats...")
            self.tokenizer = stats["tokenizer"]
            cfg = stats["model_config"]
            self.neural_model = Seq2SeqModel(
                cfg["vocab_size"], cfg["embed_dim"], cfg["hidden_dim"], self.device
            )
            self.neural_model.load_state_dict(stats["neural_state"])
            self.neural_model.to(self.device)


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
