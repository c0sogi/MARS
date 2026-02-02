import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed


class HFBBModel:
    """
    Hierarchical Frequency Back-Off (HFBB) Model.

    Implements a 4-level statistical lookup:
    1. Trigram (Prev, Curr, Next)
    2. Bigram Prev (Prev, Curr)
    3. Bigram Next (Curr, Next)
    4. Unigram (Curr) - Includes Confidence Score
    """

    def __init__(self):
        self.unigram_map = None
        self.bigram_prev_map = None
        self.bigram_next_map = None
        self.trigram_map = None

        # Special tokens for context boundaries
        self.SOS = "<SOS>"
        self.EOS = "<EOS>"
        self.SEP = " "  # Separator for keys

    def build(self, load_cached_data=True):
        """
        Orchestrates loading from cache or fitting from scratch.

        Args:
            load_cached_data (bool): If True, attempts to load parquet files from cache.
        """
        # Check if all cache files exist
        cache_exists = (
            os.path.exists(Config.HFBB_UNIGRAM_PATH)
            and os.path.exists(Config.HFBB_BIGRAM_PREV_PATH)
            and os.path.exists(Config.HFBB_BIGRAM_NEXT_PATH)
            and os.path.exists(Config.HFBB_TRIGRAM_PATH)
        )

        if load_cached_data and cache_exists:
            print("HFBB: Loading models from cache...")
            self.load()
        else:
            print("HFBB: Cache not found or ignore_cache set. Fitting from scratch...")
            self.fit()

    def fit(self):
        """
        Computes frequency maps from the training data and saves them to cache.
        """
        # Ensure output directory exists
        os.makedirs(Config.HFBB_CACHE_DIR, exist_ok=True)

        print("HFBB: Loading training data...")
        # Load data
        df = pd.read_csv(Config.TRAIN_FILE)

        # Fill NAs to ensure string operations work
        df["before"] = df["before"].fillna("").astype(str)
        df["after"] = df["after"].fillna("").astype(str)

        print("HFBB: Generating context columns...")
        # Generate prev and next tokens respecting sentence boundaries
        # Group by sentence_id to ensure we don't shift across sentences
        # Using transform to keep index alignment
        df["prev"] = df.groupby("sentence_id")["before"].shift(1).fillna(self.SOS)
        df["next"] = df.groupby("sentence_id")["before"].shift(-1).fillna(self.EOS)

        # ---------------------------------------------------------
        # 1. Unigram: Token -> (Mode After, Confidence)
        # ---------------------------------------------------------
        print("HFBB: Building Unigram layer...")
        # Count occurrences of each (before, after) pair
        unigram_counts = (
            df.groupby(["before", "after"]).size().reset_index(name="count")
        )

        # Get total counts per 'before' token
        token_totals = df.groupby("before").size().reset_index(name="total")

        # Find the mode (most frequent 'after') for each 'before'
        # Sort by count descending and drop duplicates to keep top 1
        unigram_modes = unigram_counts.sort_values(
            "count", ascending=False
        ).drop_duplicates("before")

        # Merge with totals to calculate confidence
        unigram_final = pd.merge(unigram_modes, token_totals, on="before")
        unigram_final["confidence"] = unigram_final["count"] / unigram_final["total"]

        # Save Unigram
        self.unigram_map = unigram_final.set_index("before")[["after", "confidence"]]
        self.unigram_map.to_parquet(Config.HFBB_UNIGRAM_PATH)
        print(f"HFBB: Saved Unigram map ({len(self.unigram_map)} entries)")

        # ---------------------------------------------------------
        # 2. Bigram Prev: "Prev Curr" -> Mode After
        # ---------------------------------------------------------
        print("HFBB: Building Bigram Prev layer...")
        # Create key
        df["key_bi_prev"] = df["prev"] + self.SEP + df["before"]

        # Find mode
        bi_prev_counts = (
            df.groupby(["key_bi_prev", "after"]).size().reset_index(name="count")
        )
        bi_prev_modes = bi_prev_counts.sort_values(
            "count", ascending=False
        ).drop_duplicates("key_bi_prev")

        # Save
        self.bigram_prev_map = bi_prev_modes.set_index("key_bi_prev")[["after"]]
        self.bigram_prev_map.to_parquet(Config.HFBB_BIGRAM_PREV_PATH)
        print(f"HFBB: Saved Bigram Prev map ({len(self.bigram_prev_map)} entries)")

        # ---------------------------------------------------------
        # 3. Bigram Next: "Curr Next" -> Mode After
        # ---------------------------------------------------------
        print("HFBB: Building Bigram Next layer...")
        df["key_bi_next"] = df["before"] + self.SEP + df["next"]

        bi_next_counts = (
            df.groupby(["key_bi_next", "after"]).size().reset_index(name="count")
        )
        bi_next_modes = bi_next_counts.sort_values(
            "count", ascending=False
        ).drop_duplicates("key_bi_next")

        self.bigram_next_map = bi_next_modes.set_index("key_bi_next")[["after"]]
        self.bigram_next_map.to_parquet(Config.HFBB_BIGRAM_NEXT_PATH)
        print(f"HFBB: Saved Bigram Next map ({len(self.bigram_next_map)} entries)")

        # ---------------------------------------------------------
        # 4. Trigram: "Prev Curr Next" -> Mode After
        # ---------------------------------------------------------
        print("HFBB: Building Trigram layer...")
        df["key_tri"] = df["prev"] + self.SEP + df["before"] + self.SEP + df["next"]

        tri_counts = df.groupby(["key_tri", "after"]).size().reset_index(name="count")
        tri_modes = tri_counts.sort_values("count", ascending=False).drop_duplicates(
            "key_tri"
        )

        self.trigram_map = tri_modes.set_index("key_tri")[["after"]]
        self.trigram_map.to_parquet(Config.HFBB_TRIGRAM_PATH)
        print(f"HFBB: Saved Trigram map ({len(self.trigram_map)} entries)")

        # Cleanup
        del df
        import gc

        gc.collect()

    def load(self):
        """
        Loads the maps from Parquet files.
        """
        print("HFBB: Loading Unigram...")
        self.unigram_map = pd.read_parquet(Config.HFBB_UNIGRAM_PATH)

        print("HFBB: Loading Bigram Prev...")
        self.bigram_prev_map = pd.read_parquet(Config.HFBB_BIGRAM_PREV_PATH)

        print("HFBB: Loading Bigram Next...")
        self.bigram_next_map = pd.read_parquet(Config.HFBB_BIGRAM_NEXT_PATH)

        print("HFBB: Loading Trigram...")
        self.trigram_map = pd.read_parquet(Config.HFBB_TRIGRAM_PATH)

        print("HFBB: All models loaded.")

    def query(self, token, prev_token=None, next_token=None):
        """
        Queries the hierarchical model for a normalization.

        Args:
            token (str): The target token to normalize.
            prev_token (str, optional): The previous token. Defaults to <SOS> if None.
            next_token (str, optional): The next token. Defaults to <EOS> if None.

        Returns:
            dict: {
                'pred': str (normalized text),
                'source': str ('TRIGRAM', 'BIGRAM_PREV', 'BIGRAM_NEXT', 'UNIGRAM', 'NONE'),
                'confidence': float (0.0 to 1.0)
            }
        """
        token = str(token)
        prev_token = str(prev_token) if prev_token is not None else self.SOS
        next_token = str(next_token) if next_token is not None else self.EOS

        # 1. Trigram Check
        tri_key = f"{prev_token}{self.SEP}{token}{self.SEP}{next_token}"
        if self.trigram_map is not None and tri_key in self.trigram_map.index:
            return {
                "pred": self.trigram_map.at[tri_key, "after"],
                "source": "TRIGRAM",
                "confidence": 1.0,  # Context matches are treated as max confidence
            }

        # 2. Bigram Prev Check
        bi_prev_key = f"{prev_token}{self.SEP}{token}"
        if (
            self.bigram_prev_map is not None
            and bi_prev_key in self.bigram_prev_map.index
        ):
            return {
                "pred": self.bigram_prev_map.at[bi_prev_key, "after"],
                "source": "BIGRAM_PREV",
                "confidence": 1.0,
            }

        # 3. Bigram Next Check
        bi_next_key = f"{token}{self.SEP}{next_token}"
        if (
            self.bigram_next_map is not None
            and bi_next_key in self.bigram_next_map.index
        ):
            return {
                "pred": self.bigram_next_map.at[bi_next_key, "after"],
                "source": "BIGRAM_NEXT",
                "confidence": 1.0,
            }

        # 4. Unigram Check
        if self.unigram_map is not None and token in self.unigram_map.index:
            # Handle potential duplicate index issues (though fit() drops duplicates)
            # using .at might return array if duplicates exist, so we ensure scalar
            res = self.unigram_map.loc[token]
            if isinstance(res, pd.DataFrame):
                # Fallback if somehow duplicates exist, take first
                res = res.iloc[0]

            return {
                "pred": res["after"],
                "source": "UNIGRAM",
                "confidence": float(res["confidence"]),
            }

        # 5. OOV
        return {"pred": None, "source": "NONE", "confidence": 0.0}
