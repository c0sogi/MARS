import pandas as pd
import numpy as np
import os
import sys
from library import config
from library import utils


class HFBB:
    """
    Hierarchical Frequency-Based Backoff (HFBB) Engine.

    Implements a 4-tier statistical lookup:
    1. Trigram (Prev, Curr, Next)
    2. Bigram Prev (Prev, Curr)
    3. Bigram Next (Curr, Next)
    4. Unigram (Curr) -> Returns Confidence Score
    """

    def __init__(self):
        self.unigram = {}
        self.bigram_prev = {}
        self.bigram_next = {}
        self.trigram = {}

    def build_stats(self, load_cached_data=True):
        """
        Builds or loads the statistical models.

        Args:
            load_cached_data (bool): If True, attempts to load from parquet cache.
                                     If False or cache missing, recomputes from train.csv.
        """
        # Paths
        uni_path = config.UNIGRAM_PATH
        bi_prev_path = config.BIGRAM_PREV_PATH
        bi_next_path = config.BIGRAM_NEXT_PATH
        tri_path = config.TRIGRAM_PATH

        # Check if all cache files exist
        all_exist = all(
            os.path.exists(p) for p in [uni_path, bi_prev_path, bi_next_path, tri_path]
        )

        if load_cached_data and all_exist:
            print("HFBB: Loading stats from cache...")
            self._load_stats()
        else:
            print("HFBB: Computing stats from training data (this may take a while)...")
            self._compute_and_save()

    def _compute_and_save(self):
        """
        Computes N-gram statistics from the training file and saves to cache.
        """
        # Load training data
        try:
            df = pd.read_csv(config.TRAIN_FILE)
        except FileNotFoundError:
            print(f"Error: Training file {config.TRAIN_FILE} not found.")
            return

        # Handle NaNs and types
        df["before"] = df["before"].fillna("").astype(str)
        df["after"] = df["after"].fillna("").astype(str)

        # Ensure data is sorted by sentence and token order
        df.sort_values(["sentence_id", "token_id"], inplace=True)

        # ---------------------------------------------------------
        # Context Generation (Vectorized)
        # ---------------------------------------------------------
        # Prev: Shift 1. If token_id is 0, it's the start of a sentence.
        df["prev"] = df["before"].shift(1).fillna("<START>")
        df.loc[df["token_id"] == 0, "prev"] = "<START>"

        # Next: Shift -1. If next row's token_id is 0, current is end of sentence.
        df["next"] = df["before"].shift(-1).fillna("<END>")
        next_token_id = df["token_id"].shift(-1).fillna(0)
        df.loc[next_token_id == 0, "next"] = "<END>"

        # ---------------------------------------------------------
        # 1. Unigram Stats (with Confidence)
        # ---------------------------------------------------------
        print("HFBB: Computing Unigrams...")
        # Count occurrences of each (before, after) pair
        uni_counts = df.groupby(["before", "after"]).size().reset_index(name="count")
        # Count total occurrences of each 'before' token
        uni_totals = df.groupby("before").size().reset_index(name="total")

        # Sort to find the mode (most frequent 'after' for each 'before')
        uni_counts.sort_values(
            ["before", "count"], ascending=[True, False], inplace=True
        )
        uni_best = uni_counts.drop_duplicates(subset=["before"])

        # Merge to calculate confidence
        uni_final = pd.merge(uni_best, uni_totals, on="before")
        uni_final["confidence"] = uni_final["count"] / uni_final["total"]

        utils.save_cache(
            uni_final[["before", "after", "confidence"]], config.UNIGRAM_PATH
        )

        # ---------------------------------------------------------
        # 2. Bigram Prev Stats
        # ---------------------------------------------------------
        print("HFBB: Computing Bigrams (Prev)...")
        # Key: (prev, before) -> after
        bi_prev_counts = (
            df.groupby(["prev", "before", "after"]).size().reset_index(name="count")
        )
        bi_prev_counts.sort_values(
            ["prev", "before", "count"], ascending=[True, True, False], inplace=True
        )
        bi_prev_best = bi_prev_counts.drop_duplicates(subset=["prev", "before"])

        utils.save_cache(
            bi_prev_best[["prev", "before", "after"]], config.BIGRAM_PREV_PATH
        )

        # ---------------------------------------------------------
        # 3. Bigram Next Stats
        # ---------------------------------------------------------
        print("HFBB: Computing Bigrams (Next)...")
        # Key: (before, next) -> after
        bi_next_counts = (
            df.groupby(["before", "next", "after"]).size().reset_index(name="count")
        )
        bi_next_counts.sort_values(
            ["before", "next", "count"], ascending=[True, True, False], inplace=True
        )
        bi_next_best = bi_next_counts.drop_duplicates(subset=["before", "next"])

        utils.save_cache(
            bi_next_best[["before", "next", "after"]], config.BIGRAM_NEXT_PATH
        )

        # ---------------------------------------------------------
        # 4. Trigram Stats
        # ---------------------------------------------------------
        print("HFBB: Computing Trigrams...")
        # Key: (prev, before, next) -> after
        tri_counts = (
            df.groupby(["prev", "before", "next", "after"])
            .size()
            .reset_index(name="count")
        )
        tri_counts.sort_values(
            ["prev", "before", "next", "count"],
            ascending=[True, True, True, False],
            inplace=True,
        )
        tri_best = tri_counts.drop_duplicates(subset=["prev", "before", "next"])

        utils.save_cache(
            tri_best[["prev", "before", "next", "after"]], config.TRIGRAM_PATH
        )

        # Load newly computed stats into memory
        self._load_stats()

    def _load_stats(self):
        """
        Loads stats from parquet files into memory (dictionaries) for fast O(1) lookup.
        """
        print("HFBB: Loading tables into memory...")

        # Unigram: dict[before] -> (after, confidence)
        uni_df = utils.load_cache(config.UNIGRAM_PATH)
        self.unigram = dict(
            zip(uni_df["before"], zip(uni_df["after"], uni_df["confidence"]))
        )

        # Bigram Prev: dict[(prev, before)] -> after
        bi_prev_df = utils.load_cache(config.BIGRAM_PREV_PATH)
        self.bigram_prev = dict(
            zip(zip(bi_prev_df["prev"], bi_prev_df["before"]), bi_prev_df["after"])
        )

        # Bigram Next: dict[(before, next)] -> after
        bi_next_df = utils.load_cache(config.BIGRAM_NEXT_PATH)
        self.bigram_next = dict(
            zip(zip(bi_next_df["before"], bi_next_df["next"]), bi_next_df["after"])
        )

        # Trigram: dict[(prev, before, next)] -> after
        tri_df = utils.load_cache(config.TRIGRAM_PATH)
        self.trigram = dict(
            zip(zip(tri_df["prev"], tri_df["before"], tri_df["next"]), tri_df["after"])
        )

        print(
            f"HFBB: Loaded - Unigram: {len(self.unigram)}, BiPrev: {len(self.bigram_prev)}, "
            f"BiNext: {len(self.bigram_next)}, Trigram: {len(self.trigram)}"
        )

    def query(self, prev, curr, next_tok):
        """
        Queries the hierarchical model for a normalization.

        Args:
            prev (str): The previous token.
            curr (str): The current token to normalize.
            next_tok (str): The next token.

        Returns:
            tuple: (prediction, confidence, level)
                   prediction (str): The normalized text.
                   confidence (float): 0.0 to 1.0 (1.0 for context matches).
                   level (str): Source of prediction ('TRIGRAM', 'BIGRAM_PREV', 'BIGRAM_NEXT', 'UNIGRAM', 'OOV').
        """
        # 1. Trigram Match
        if (prev, curr, next_tok) in self.trigram:
            return self.trigram[(prev, curr, next_tok)], 1.0, "TRIGRAM"

        # 2. Bigram Prev Match
        if (prev, curr) in self.bigram_prev:
            return self.bigram_prev[(prev, curr)], 1.0, "BIGRAM_PREV"

        # 3. Bigram Next Match
        if (curr, next_tok) in self.bigram_next:
            return self.bigram_next[(curr, next_tok)], 1.0, "BIGRAM_NEXT"

        # 4. Unigram Match
        if curr in self.unigram:
            res = self.unigram[curr]  # (after, confidence)
            return res[0], res[1], "UNIGRAM"

        # 5. Out of Vocabulary
        return curr, 0.0, "OOV"

    def evaluate(self, val_df=None):
        """
        Evaluates the HFBB model on the validation set.

        Args:
            val_df (pd.DataFrame, optional): Validation dataframe. If None, loads from config.VAL_FILE.

        Returns:
            float: Accuracy score.
        """
        if val_df is None:
            if os.path.exists(config.VAL_FILE):
                print("HFBB: Loading validation data for evaluation...")
                val_df = pd.read_csv(config.VAL_FILE)
            else:
                print("HFBB: No validation data found. Skipping evaluation.")
                return 0.0

        print("HFBB: Evaluating on validation set...")

        # Prepare data
        val_df = val_df.copy()
        val_df["before"] = val_df["before"].fillna("").astype(str)
        val_df["after"] = val_df["after"].fillna("").astype(str)

        # Ensure correct order
        val_df.sort_values(["sentence_id", "token_id"], inplace=True)

        # Generate context
        val_df["prev"] = val_df["before"].shift(1).fillna("<START>")
        val_df.loc[val_df["token_id"] == 0, "prev"] = "<START>"

        val_df["next"] = val_df["before"].shift(-1).fillna("<END>")
        next_token_id = val_df["token_id"].shift(-1).fillna(0)
        val_df.loc[next_token_id == 0, "next"] = "<END>"

        # Extract lists for fast iteration
        prevs = val_df["prev"].tolist()
        currs = val_df["before"].tolist()
        nexts = val_df["next"].tolist()
        targets = val_df["after"].tolist()

        correct_count = 0
        total = len(targets)

        # Evaluation loop
        for p, c, n, t in zip(prevs, currs, nexts, targets):
            pred, _, _ = self.query(p, c, n)
            if pred == t:
                correct_count += 1

        accuracy = correct_count / total if total > 0 else 0.0
        print(f"HFBB Validation Accuracy: {accuracy}")
        return accuracy
