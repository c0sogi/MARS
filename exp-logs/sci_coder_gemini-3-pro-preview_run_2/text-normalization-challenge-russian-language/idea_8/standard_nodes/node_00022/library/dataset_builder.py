import os
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from library.config import Config
from library.utils import set_seed


class InMemoryHFBB:
    """
    A lightweight, in-memory implementation of the HFBB logic
    optimized for the Jackknife process using Pandas vectorized operations.
    Unlike the main HFBBModel, this does not read/write to disk and is designed
    for rapid fitting on DataFrame subsets.
    """

    def __init__(self):
        self.unigram = None
        self.bigram_prev = None
        self.bigram_next = None
        self.trigram = None
        self.SEP = " "
        self.SOS = "<SOS>"
        self.EOS = "<EOS>"

    def fit(self, df):
        """
        Fits the hierarchical statistical models on the provided DataFrame.
        Expects df to have 'before', 'after', 'prev', 'next' columns.
        """
        # 1. Unigram: Token -> (Mode After, Confidence)
        uni_counts = df.groupby(["before", "after"]).size().reset_index(name="count")
        uni_totals = df.groupby("before").size().reset_index(name="total")
        # Get Mode (most frequent 'after')
        uni_modes = uni_counts.sort_values("count", ascending=False).drop_duplicates(
            "before"
        )
        # Merge for confidence calculation
        uni_final = pd.merge(uni_modes, uni_totals, on="before")
        uni_final["confidence"] = uni_final["count"] / uni_final["total"]
        self.unigram = uni_final.set_index("before")[["after", "confidence"]]

        # 2. Bigram Prev: "Prev Curr" -> Mode After
        df["key_bp"] = df["prev"] + self.SEP + df["before"]
        bp_counts = df.groupby(["key_bp", "after"]).size().reset_index(name="count")
        bp_modes = bp_counts.sort_values("count", ascending=False).drop_duplicates(
            "key_bp"
        )
        self.bigram_prev = bp_modes.set_index("key_bp")[["after"]]

        # 3. Bigram Next: "Curr Next" -> Mode After
        df["key_bn"] = df["before"] + self.SEP + df["next"]
        bn_counts = df.groupby(["key_bn", "after"]).size().reset_index(name="count")
        bn_modes = bn_counts.sort_values("count", ascending=False).drop_duplicates(
            "key_bn"
        )
        self.bigram_next = bn_modes.set_index("key_bn")[["after"]]

        # 4. Trigram: "Prev Curr Next" -> Mode After
        df["key_tri"] = df["prev"] + self.SEP + df["before"] + self.SEP + df["next"]
        tri_counts = df.groupby(["key_tri", "after"]).size().reset_index(name="count")
        tri_modes = tri_counts.sort_values("count", ascending=False).drop_duplicates(
            "key_tri"
        )
        self.trigram = tri_modes.set_index("key_tri")[["after"]]

        # Cleanup temporary key columns to save memory
        df.drop(columns=["key_bp", "key_bn", "key_tri"], inplace=True, errors="ignore")

    def predict(self, df):
        """
        Performs vectorized prediction on the provided DataFrame.
        Returns a DataFrame with ['pred', 'confidence', 'source'] aligned with input df index.
        """
        # Prepare keys for lookup
        df = df.copy()
        df["key_bp"] = df["prev"] + self.SEP + df["before"]
        df["key_bn"] = df["before"] + self.SEP + df["next"]
        df["key_tri"] = df["prev"] + self.SEP + df["before"] + self.SEP + df["next"]

        # Initialize results container
        results = pd.DataFrame(index=df.index)
        results["pred"] = None
        results["confidence"] = 0.0
        results["source"] = "NONE"

        # 1. Trigram Lookup (Highest Priority)
        tri_match = df.join(self.trigram, on="key_tri", rsuffix="_tri")
        mask_tri = tri_match["after"].notna()
        results.loc[mask_tri, "pred"] = tri_match.loc[mask_tri, "after"]
        results.loc[mask_tri, "confidence"] = 1.0
        results.loc[mask_tri, "source"] = "TRIGRAM"

        # Identify remaining unpredicted tokens
        remaining = ~mask_tri

        # 2. Bigram Prev Lookup
        if remaining.any():
            bp_match = df.loc[remaining].join(
                self.bigram_prev, on="key_bp", rsuffix="_bp"
            )
            mask_bp = bp_match["after"].notna()
            global_idx = bp_match.loc[mask_bp].index
            results.loc[global_idx, "pred"] = bp_match.loc[mask_bp, "after"]
            results.loc[global_idx, "confidence"] = 1.0
            results.loc[global_idx, "source"] = "BIGRAM_PREV"
            remaining = remaining & (~results["pred"].notna())

        # 3. Bigram Next Lookup
        if remaining.any():
            bn_match = df.loc[remaining].join(
                self.bigram_next, on="key_bn", rsuffix="_bn"
            )
            mask_bn = bn_match["after"].notna()
            global_idx = bn_match.loc[mask_bn].index
            results.loc[global_idx, "pred"] = bn_match.loc[mask_bn, "after"]
            results.loc[global_idx, "confidence"] = 1.0
            results.loc[global_idx, "source"] = "BIGRAM_NEXT"
            remaining = remaining & (~results["pred"].notna())

        # 4. Unigram Lookup (Lowest Priority, includes Confidence)
        if remaining.any():
            uni_match = df.loc[remaining].join(
                self.unigram, on="before", rsuffix="_uni"
            )
            mask_uni = uni_match["after"].notna()
            global_idx = uni_match.loc[mask_uni].index
            results.loc[global_idx, "pred"] = uni_match.loc[mask_uni, "after"]
            results.loc[global_idx, "confidence"] = uni_match.loc[
                mask_uni, "confidence"
            ]
            results.loc[global_idx, "source"] = "UNIGRAM"

        return results


class DatasetBuilder:
    """
    Constructs the Curriculum-Enriched Residual dataset for the Transformer.
    Uses Jackknifing to identify failure cases of the HFBB model.
    """

    def __init__(self):
        self.SOS = "<SOS>"
        self.EOS = "<EOS>"
        self.SEP = "<SEP>"  # Separator for input string formatting

    def _add_context(self, df):
        """
        Adds 'prev' and 'next' token columns respecting sentence boundaries.
        """
        # Ensure string types
        df["before"] = df["before"].fillna("").astype(str)
        df["after"] = df["after"].fillna("").astype(str)

        # Sort to ensure sequential processing
        df = df.sort_values(["sentence_id", "token_id"]).copy()

        # Shift to get context
        df["prev"] = df["before"].shift(1).fillna(self.SOS)
        df["next"] = df["before"].shift(-1).fillna(self.EOS)

        # Mask boundaries where sentence_id changes
        # If sentence_id[i] != sentence_id[i-1], prev[i] is SOS
        sentence_change_prev = df["sentence_id"] != df["sentence_id"].shift(1)
        df.loc[sentence_change_prev, "prev"] = self.SOS

        # If sentence_id[i] != sentence_id[i+1], next[i] is EOS
        sentence_change_next = df["sentence_id"] != df["sentence_id"].shift(-1)
        df.loc[sentence_change_next, "next"] = self.EOS

        return df

    def _filter_residuals(self, df, preds):
        """
        Identifies residuals (errors/ambiguities) and anchors (correct semiotics).

        Logic:
        1. Residuals: (Pred != Actual) OR (Pred == Actual AND Conf <= Threshold AND Semiotic)
        2. Anchors: (Pred == Actual AND Conf > Threshold AND Semiotic)
        """
        df = df.copy()
        df["pred"] = preds["pred"]
        df["confidence"] = preds["confidence"]

        # Identify Semiotics (Digits or Latin chars)
        # Using vectorized regex for speed
        df["is_semiotic"] = df["before"].str.contains(r"[\d|a-zA-Z]", regex=True)

        # Conditions
        is_correct = df["pred"] == df["after"]
        is_high_conf = df["confidence"] > Config.HFBB_CONFIDENCE_THRESHOLD

        # Residuals
        # 1. Hard Errors
        cond_error = ~is_correct
        # 2. Ambiguous Semiotics (Correct but low confidence, e.g. "2" -> "two" or "second"?)
        cond_ambiguous = is_correct & (~is_high_conf) & df["is_semiotic"]

        mask_residual = cond_error | cond_ambiguous

        # Anchors (Correct, High Confidence, Semiotic)
        mask_anchor = is_correct & is_high_conf & df["is_semiotic"]

        residuals = df[mask_residual].copy()
        anchors = df[mask_anchor].copy()

        return residuals, anchors

    def build_dataset(self, load_cached_data=True):
        """
        Main method to build or load the dataset.

        Args:
            load_cached_data (bool): Whether to load from parquet cache if available.

        Returns:
            tuple: (train_df, val_df) formatted for the Transformer.
        """
        Config.setup_dirs()

        # Check cache
        if (
            load_cached_data
            and os.path.exists(Config.RESIDUAL_TRAIN_PATH)
            and os.path.exists(Config.RESIDUAL_VAL_PATH)
        ):
            print("DatasetBuilder: Loading cached datasets...")
            train_df = pd.read_parquet(Config.RESIDUAL_TRAIN_PATH)
            val_df = pd.read_parquet(Config.RESIDUAL_VAL_PATH)
            return train_df, val_df

        print("DatasetBuilder: Building dataset from scratch...")
        set_seed()

        # Load raw metadata
        print("DatasetBuilder: Loading metadata...")
        full_train_df = pd.read_csv(Config.TRAIN_FILE)
        full_val_df = pd.read_csv(Config.VAL_FILE)

        # Add context columns
        print("DatasetBuilder: Generating context...")
        full_train_df = self._add_context(full_train_df)
        full_val_df = self._add_context(full_val_df)

        # --- Phase A: Train Set Generation (Jackknife) ---
        print(
            f"DatasetBuilder: Starting {Config.N_FOLDS}-Fold Jackknife on Training Set..."
        )

        collected_residuals = []
        collected_anchors = []

        # Use GroupKFold to prevent sentence leakage between folds
        gkf = GroupKFold(n_splits=Config.N_FOLDS)
        groups = full_train_df["sentence_id"].values

        fold = 1
        for train_idx, holdout_idx in gkf.split(full_train_df, groups=groups):
            print(f"  Processing Fold {fold}/{Config.N_FOLDS}...")

            # Split data
            fold_train = full_train_df.iloc[train_idx]
            fold_holdout = full_train_df.iloc[holdout_idx]

            # Train temporary HFBB model on fold_train
            hfbb = InMemoryHFBB()
            hfbb.fit(fold_train)

            # Predict on holdout set
            preds = hfbb.predict(fold_holdout)

            # Identify residuals and anchors
            res, anc = self._filter_residuals(fold_holdout, preds)

            collected_residuals.append(res)
            collected_anchors.append(anc)

            # Memory cleanup
            del hfbb, preds, fold_train, fold_holdout
            gc.collect()
            fold += 1

        # Combine results from all folds
        all_residuals = pd.concat(collected_residuals, ignore_index=True)
        all_anchors = pd.concat(collected_anchors, ignore_index=True)

        print(
            f"DatasetBuilder: Found {len(all_residuals)} residuals and {len(all_anchors)} potential anchors."
        )

        # Sample Anchors to prevent catastrophic forgetting
        n_anchors = int(len(all_anchors) * Config.ANCHOR_RATIO)
        if n_anchors > 0:
            sampled_anchors = all_anchors.sample(n=n_anchors, random_state=Config.SEED)
        else:
            sampled_anchors = pd.DataFrame(columns=all_anchors.columns)

        print(f"DatasetBuilder: Sampled {len(sampled_anchors)} anchors.")

        # Combine to form final training set
        final_train = pd.concat([all_residuals, sampled_anchors], ignore_index=True)

        # Format for Transformer: "Prev <SEP> Target <SEP> Next"
        final_train["input_text"] = (
            final_train["prev"]
            + self.SEP
            + final_train["before"]
            + self.SEP
            + final_train["next"]
        )
        final_train["target_text"] = final_train["after"]

        # Shuffle
        final_train = (
            final_train[["input_text", "target_text"]]
            .sample(frac=1.0, random_state=Config.SEED)
            .reset_index(drop=True)
        )

        # --- Phase B: Validation Set Generation ---
        print("DatasetBuilder: Generating Validation Set...")

        # Train HFBB on the FULL training set (this mimics inference time state)
        full_hfbb = InMemoryHFBB()
        full_hfbb.fit(full_train_df)

        # Predict on validation set
        val_preds = full_hfbb.predict(full_val_df)

        # Filter residuals (errors + ambiguous)
        val_residuals, val_anchors = self._filter_residuals(full_val_df, val_preds)

        # Sample anchors for validation as well to maintain distribution consistency
        n_val_anchors = int(len(val_anchors) * Config.ANCHOR_RATIO)
        if n_val_anchors > 0:
            sampled_val_anchors = val_anchors.sample(
                n=n_val_anchors, random_state=Config.SEED
            )
        else:
            sampled_val_anchors = pd.DataFrame(columns=val_anchors.columns)

        final_val = pd.concat([val_residuals, sampled_val_anchors], ignore_index=True)

        final_val["input_text"] = (
            final_val["prev"]
            + self.SEP
            + final_val["before"]
            + self.SEP
            + final_val["next"]
        )
        final_val["target_text"] = final_val["after"]

        final_val = (
            final_val[["input_text", "target_text"]]
            .sample(frac=1.0, random_state=Config.SEED)
            .reset_index(drop=True)
        )

        print(f"DatasetBuilder: Final Train Size: {len(final_train)}")
        print(f"DatasetBuilder: Final Val Size: {len(final_val)}")

        # Save to cache
        print("DatasetBuilder: Saving to cache...")
        final_train.to_parquet(Config.RESIDUAL_TRAIN_PATH)
        final_val.to_parquet(Config.RESIDUAL_VAL_PATH)

        return final_train, final_val
