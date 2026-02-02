import os
import pandas as pd
import numpy as np
from library.config import Config


class SymbolicMemory:
    """
    Stage 1: Bidirectional Symbolic Memory.
    Implements a hierarchical lookup system (Trigram > Bigram > Unigram)
    based on Maximum Likelihood Estimation from the training corpus.
    """

    def __init__(self):
        # In-memory lookup tables
        # Keys are tuples (e.g., (prev, curr, next)), Values are strings (normalized text)
        self.trigram_stats = {}
        self.bigram_left_stats = {}
        self.bigram_right_stats = {}
        self.unigram_stats = {}

        # Define file paths for caching
        self.stats_dir = Config.STATS_DIR
        self.trigram_path = os.path.join(self.stats_dir, "trigram_stats.parquet")
        self.bigram_left_path = os.path.join(
            self.stats_dir, "bigram_left_stats.parquet"
        )
        self.bigram_right_path = os.path.join(
            self.stats_dir, "bigram_right_stats.parquet"
        )
        self.unigram_path = os.path.join(self.stats_dir, "unigram_stats.parquet")

    def build_stats(self, df=None, load_cached_data=True):
        """
        Constructs or loads the N-gram statistics.

        Args:
            df (pd.DataFrame): Training data containing 'sentence_id', 'before', 'after'.
                               Required if load_cached_data is False or cache is missing.
            load_cached_data (bool): If True, attempts to load from disk first.
        """
        # Ensure stats directory exists
        os.makedirs(self.stats_dir, exist_ok=True)

        # Check if all cache files exist
        cache_exists = (
            os.path.exists(self.trigram_path)
            and os.path.exists(self.bigram_left_path)
            and os.path.exists(self.bigram_right_path)
            and os.path.exists(self.unigram_path)
        )

        if load_cached_data and cache_exists:
            print("Loading symbolic stats from cache...")
            self._load_stats()
        else:
            print("Computing symbolic stats from scratch...")
            if df is None:
                raise ValueError("DataFrame 'df' is required when cache is not used.")
            self._compute_and_save_stats(df)

        print(
            f"Symbolic Memory Ready. "
            f"Trigrams: {len(self.trigram_stats)}, "
            f"Bigrams(L): {len(self.bigram_left_stats)}, "
            f"Bigrams(R): {len(self.bigram_right_stats)}, "
            f"Unigrams: {len(self.unigram_stats)}"
        )

    def _compute_and_save_stats(self, df):
        """
        Internal method to compute MLE stats and save to Parquet.
        """
        # Work on a copy to avoid modifying the input
        # Ensure string types
        df = df.copy()
        df["before"] = df["before"].astype(str)
        df["after"] = df["after"].astype(str)

        # 1. Generate Context Columns (prev, next)
        print("Generating context columns...")

        # Shift to get candidates
        df["prev"] = df["before"].shift(1)
        df["next"] = df["before"].shift(-1)

        # Handle Sentence Boundaries
        # If sentence_id changed from previous row, current 'prev' is invalid -> SOS
        # We assume df is sorted by sentence_id, token_id (standard in this dataset)

        # Identify start of sentences
        # (sentence_id != prev_sentence_id)
        # We fill NA in shift with a dummy value (-1) to handle the first row
        sent_id_shift = df["sentence_id"].shift(1).fillna(-1)
        is_start = df["sentence_id"] != sent_id_shift

        # Identify end of sentences
        # (sentence_id != next_sentence_id)
        sent_id_shift_next = df["sentence_id"].shift(-1).fillna(-1)
        is_end = df["sentence_id"] != sent_id_shift_next

        # Apply Special Tokens
        df.loc[is_start, "prev"] = Config.SOS_TOKEN
        df.loc[is_end, "next"] = Config.EOS_TOKEN

        # Fill any remaining NaNs (e.g. if single-token sentences exist, logic holds)
        df["prev"] = df["prev"].fillna(Config.SOS_TOKEN)
        df["next"] = df["next"].fillna(Config.EOS_TOKEN)

        # 2. Compute Stats (MLE)
        # Helper function to extract best mapping
        def extract_best_mapping(group_cols, save_path):
            print(f"Aggregating stats for columns: {group_cols}...")
            # Count occurrences of each mapping
            # group_cols includes the input features + the target 'after'
            counts = df.groupby(group_cols + ["after"]).size().reset_index(name="count")

            # Sort by count descending
            counts = counts.sort_values("count", ascending=False)

            # Drop duplicates on input features, keeping the one with highest count
            best_mapping = counts.drop_duplicates(subset=group_cols)

            # Save to parquet
            best_mapping.to_parquet(save_path, index=False)
            return best_mapping

        # Trigram: (prev, curr, next) -> after
        df_tri = extract_best_mapping(["prev", "before", "next"], self.trigram_path)
        self.trigram_stats = dict(
            zip(zip(df_tri["prev"], df_tri["before"], df_tri["next"]), df_tri["after"])
        )

        # Bigram Left: (prev, curr) -> after
        df_bi_l = extract_best_mapping(["prev", "before"], self.bigram_left_path)
        self.bigram_left_stats = dict(
            zip(zip(df_bi_l["prev"], df_bi_l["before"]), df_bi_l["after"])
        )

        # Bigram Right: (curr, next) -> after
        df_bi_r = extract_best_mapping(["before", "next"], self.bigram_right_path)
        self.bigram_right_stats = dict(
            zip(zip(df_bi_r["before"], df_bi_r["next"]), df_bi_r["after"])
        )

        # Unigram: (curr) -> after
        df_uni = extract_best_mapping(["before"], self.unigram_path)
        self.unigram_stats = dict(zip(df_uni["before"], df_uni["after"]))

    def _load_stats(self):
        """
        Internal method to load stats from Parquet files into memory.
        """
        # Load Trigrams
        df_tri = pd.read_parquet(self.trigram_path)
        # Create dictionary: (prev, curr, next) -> after
        self.trigram_stats = dict(
            zip(zip(df_tri["prev"], df_tri["before"], df_tri["next"]), df_tri["after"])
        )

        # Load Bigrams Left
        df_bi_l = pd.read_parquet(self.bigram_left_path)
        self.bigram_left_stats = dict(
            zip(zip(df_bi_l["prev"], df_bi_l["before"]), df_bi_l["after"])
        )

        # Load Bigrams Right
        df_bi_r = pd.read_parquet(self.bigram_right_path)
        self.bigram_right_stats = dict(
            zip(zip(df_bi_r["before"], df_bi_r["next"]), df_bi_r["after"])
        )

        # Load Unigrams
        df_uni = pd.read_parquet(self.unigram_path)
        self.unigram_stats = dict(zip(df_uni["before"], df_uni["after"]))

    def query(self, prev_token, curr_token, next_token):
        """
        Queries the symbolic memory for a normalized form.

        Priority:
        1. Trigram (prev, curr, next)
        2. Bigram Left (prev, curr)
        3. Bigram Right (curr, next)
        4. Unigram (curr)

        Args:
            prev_token (str): The preceding token (or <sos>).
            curr_token (str): The token to normalize.
            next_token (str): The following token (or <eos>).

        Returns:
            str or None: The normalized text if found, else None.
        """
        # Ensure inputs are strings
        p, c, n = str(prev_token), str(curr_token), str(next_token)

        # 1. Trigram Check
        res = self.trigram_stats.get((p, c, n))
        if res is not None:
            return res

        # 2. Bigram Left Check
        res = self.bigram_left_stats.get((p, c))
        if res is not None:
            return res

        # 3. Bigram Right Check
        res = self.bigram_right_stats.get((c, n))
        if res is not None:
            return res

        # 4. Unigram Check
        res = self.unigram_stats.get(c)
        if res is not None:
            return res

        # No match found
        return None
