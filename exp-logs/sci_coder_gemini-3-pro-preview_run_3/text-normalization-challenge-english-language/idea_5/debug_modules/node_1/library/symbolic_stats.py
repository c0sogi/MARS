import os
import pandas as pd
import numpy as np
from library.config import Config


class StatsBuilder:
    """
    Constructs hierarchical frequency maps (Trigrams, Bigrams, Unigrams)
    from the training data to serve as the symbolic 'Head' of the model.
    """

    def __init__(self):
        self.stats_files = {
            "trigram": Config.STATS_TRIGRAM,
            "bigram_left": Config.STATS_BIGRAM_LEFT,
            "bigram_right": Config.STATS_BIGRAM_RIGHT,
            "unigram": Config.STATS_UNIGRAM,
        }

    def _compute_stats(self):
        print("Computing symbolic statistics from training data...")

        # Load training data
        if not os.path.exists(Config.TRAIN_META):
            raise FileNotFoundError(
                f"Training metadata not found at {Config.TRAIN_META}"
            )

        df = pd.read_parquet(Config.TRAIN_META)

        # Ensure string types to prevent mixed-type errors
        df["before"] = df["before"].astype(str)
        df["after"] = df["after"].astype(str)

        # Generate context columns (prev, next) efficiently
        print("Generating context windows...")
        df["prev"] = df["before"].shift(1)
        df["next"] = df["before"].shift(-1)

        # Identify sentence boundaries
        # A boundary exists if the sentence_id changes compared to the adjacent row
        sent_ids = df["sentence_id"]

        # Mask for the start of a sentence (or first row)
        # shift(1) at index 0 is NaN, so (val != NaN) is True
        is_start = sent_ids != sent_ids.shift(1)

        # Mask for the end of a sentence (or last row)
        # shift(-1) at last index is NaN, so (val != NaN) is True
        is_end = sent_ids != sent_ids.shift(-1)

        # Fill boundaries with Special Tokens
        # We fillna first to ensure column consistency before assignment
        df["prev"] = df["prev"].fillna(Config.SOS_TOKEN)
        df["next"] = df["next"].fillna(Config.EOS_TOKEN)

        df.loc[is_start, "prev"] = Config.SOS_TOKEN
        df.loc[is_end, "next"] = Config.EOS_TOKEN

        # --- 1. Trigram Stats (prev, curr, next) ---
        print("Aggregating Trigrams...")
        # Count occurrences of each mapping: (prev, before, next) -> after
        trigram_counts = (
            df.groupby(["prev", "before", "next", "after"])
            .size()
            .reset_index(name="count")
        )
        # Select the most frequent 'after' for each unique context
        trigram_best = trigram_counts.sort_values(
            "count", ascending=False
        ).drop_duplicates(subset=["prev", "before", "next"])
        trigram_best[["prev", "before", "next", "after", "count"]].to_parquet(
            self.stats_files["trigram"], index=False
        )
        del trigram_counts, trigram_best

        # --- 2. Bigram Left Stats (prev, curr) ---
        print("Aggregating Left Bigrams...")
        bg_left_counts = (
            df.groupby(["prev", "before", "after"]).size().reset_index(name="count")
        )
        bg_left_best = bg_left_counts.sort_values(
            "count", ascending=False
        ).drop_duplicates(subset=["prev", "before"])
        bg_left_best[["prev", "before", "after", "count"]].to_parquet(
            self.stats_files["bigram_left"], index=False
        )
        del bg_left_counts, bg_left_best

        # --- 3. Bigram Right Stats (curr, next) ---
        print("Aggregating Right Bigrams...")
        bg_right_counts = (
            df.groupby(["before", "next", "after"]).size().reset_index(name="count")
        )
        bg_right_best = bg_right_counts.sort_values(
            "count", ascending=False
        ).drop_duplicates(subset=["before", "next"])
        bg_right_best[["before", "next", "after", "count"]].to_parquet(
            self.stats_files["bigram_right"], index=False
        )
        del bg_right_counts, bg_right_best

        # --- 4. Unigram Stats (curr) ---
        print("Aggregating Unigrams...")
        unigram_counts = (
            df.groupby(["before", "after"]).size().reset_index(name="count")
        )
        unigram_best = unigram_counts.sort_values(
            "count", ascending=False
        ).drop_duplicates(subset=["before"])
        unigram_best[["before", "after", "count"]].to_parquet(
            self.stats_files["unigram"], index=False
        )
        del unigram_counts, unigram_best

        print("Symbolic statistics computation complete.")

    def run(self, load_cached_data=True):
        """
        Executes the stats building process.
        Checks for cached files first unless load_cached_data is False.
        """
        # Ensure output directory exists
        os.makedirs(Config.IDEA_DIR, exist_ok=True)

        # Check if all files exist
        all_exist = all(os.path.exists(f) for f in self.stats_files.values())

        if load_cached_data and all_exist:
            print("Loading cached symbolic statistics (skipping computation)...")
            return

        self._compute_stats()


class HierarchicalLookup:
    """
    Provides a query interface for the hierarchical symbolic memory.
    Prioritized lookup: Trigram -> Bigram Left -> Bigram Right -> Unigram.
    """

    def __init__(self):
        self.trigram_map = {}
        self.bigram_left_map = {}
        self.bigram_right_map = {}
        self.unigram_map = {}
        self._load_stats()

    def _load_stats(self):
        print("Loading symbolic stats into memory...")

        # Load Trigrams
        if os.path.exists(Config.STATS_TRIGRAM):
            df = pd.read_parquet(Config.STATS_TRIGRAM)
            # Create a dict: key=(prev, curr, next) -> val=after
            self.trigram_map = dict(
                zip(zip(df["prev"], df["before"], df["next"]), df["after"])
            )

        # Load Bigram Left
        if os.path.exists(Config.STATS_BIGRAM_LEFT):
            df = pd.read_parquet(Config.STATS_BIGRAM_LEFT)
            self.bigram_left_map = dict(zip(zip(df["prev"], df["before"]), df["after"]))

        # Load Bigram Right
        if os.path.exists(Config.STATS_BIGRAM_RIGHT):
            df = pd.read_parquet(Config.STATS_BIGRAM_RIGHT)
            self.bigram_right_map = dict(
                zip(zip(df["before"], df["next"]), df["after"])
            )

        # Load Unigram
        if os.path.exists(Config.STATS_UNIGRAM):
            df = pd.read_parquet(Config.STATS_UNIGRAM)
            self.unigram_map = dict(zip(df["before"], df["after"]))

        print(
            f"Symbolic stats loaded. Trigrams: {len(self.trigram_map)}, Unigrams: {len(self.unigram_map)}"
        )

    def query(self, prev_token, curr_token, next_token):
        """
        Queries the hierarchy for a normalization.
        Returns the normalized string if found, else None.
        """
        # 1. Trigram (Exact Context)
        res = self.trigram_map.get((prev_token, curr_token, next_token))
        if res is not None:
            return res

        # 2. Bigram Left (Previous Context)
        res = self.bigram_left_map.get((prev_token, curr_token))
        if res is not None:
            return res

        # 3. Bigram Right (Next Context)
        res = self.bigram_right_map.get((curr_token, next_token))
        if res is not None:
            return res

        # 4. Unigram (No Context)
        res = self.unigram_map.get(curr_token)
        if res is not None:
            return res

        return None
