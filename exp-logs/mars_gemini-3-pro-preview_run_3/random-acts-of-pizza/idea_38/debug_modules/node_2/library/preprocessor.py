import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from library.config import (
    METADATA_DIR,
    CACHE_DIR,
    SEED,
    N_FOLDS,
    ID_COL,
    TARGET_COL,
    TEXT_COLS,
    METADATA_COLS,
    SUBREDDIT_COL,
)
from library.utils import get_logger, save_to_cache, load_from_cache

logger = get_logger("preprocessor")


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Handles text concatenation and cleaning.
    """

    def __init__(self, text_cols=TEXT_COLS):
        self.text_cols = text_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Create a copy to avoid SettingWithCopy warnings
        X_out = X.copy()

        # Fill NaNs with empty string
        for col in self.text_cols:
            X_out[col] = X_out[col].fillna("").astype(str)

        # Concatenate title and body
        # Assuming order: [Title, Body] based on config
        # We join them with a space.
        X_out["text_concat"] = X_out[self.text_cols].apply(
            lambda x: " ".join(x), axis=1
        )

        return X_out[["text_concat"]]


class CommunityTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Generates Community Generosity Score using Target Encoding on Subreddits.
    Supports nested cross-validation to prevent leakage in training data.
    """

    def __init__(self, subreddit_col=SUBREDDIT_COL, target_col=TARGET_COL):
        self.subreddit_col = subreddit_col
        self.target_col = target_col
        self.subreddit_map_ = {}
        self.global_mean_ = 0.0

    def fit(self, X, y=None):
        """
        Computes global scores for subreddits based on X and y.
        X must contain the subreddit list column.
        y must be the target series (or present in X if y is None).
        """
        if y is None:
            if self.target_col not in X.columns:
                raise ValueError("Target column must be provided in X or y.")
            y = X[self.target_col]

        # Calculate global mean
        self.global_mean_ = y.mean()

        # Create a temporary dataframe for explosion
        temp_df = pd.DataFrame({"subreddit": X[self.subreddit_col], "target": y.values})

        # Explode the list of subreddits
        # Rows with empty lists become NaNs in 'subreddit' column after explode (if empty list)
        # or just disappear depending on pandas version/settings.
        # We drop NaNs to be safe.
        exploded = temp_df.explode("subreddit")
        exploded = exploded.dropna(subset=["subreddit"])

        if exploded.empty:
            self.subreddit_map_ = {}
            return self

        # Group by subreddit and calculate mean target
        grouped = exploded.groupby("subreddit")["target"].mean()
        self.subreddit_map_ = grouped.to_dict()

        return self

    def transform(self, X):
        """
        Applies the learned subreddit scores to X.
        """
        # Explode
        temp_series = X[self.subreddit_col].explode()

        # Map scores
        # Map using the learned dictionary, fill unknown with global mean
        mapped_scores = temp_series.map(self.subreddit_map_).fillna(self.global_mean_)

        # If a user had an empty list, they generated a NaN in explode (or were dropped).
        # We need to aggregate back to the original index.
        # We group by index (level=0) and take the mean.
        # For users with no subreddits, this results in NaN. We fill those with global mean.

        # Note: mapped_scores index matches X index (duplicated for multiple subreddits)
        agg_scores = mapped_scores.groupby(level=0).mean()

        # Reindex to ensure we have a score for every row in X
        final_scores = agg_scores.reindex(X.index, fill_value=self.global_mean_)

        return pd.DataFrame({"community_generosity_score": final_scores})

    def fit_transform_nested(self, X, y, n_folds=N_FOLDS, seed=SEED):
        """
        Performs nested cross-validation to generate leakage-free scores for training data.
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

        # Placeholder for results
        scores = np.zeros(len(X))

        # Ensure index alignment
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        for train_idx, val_idx in kf.split(X, y):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train = y.iloc[train_idx]

            # Fit on fold train
            encoder = CommunityTargetEncoder(self.subreddit_col, self.target_col)
            encoder.fit(X_fold_train, y_fold_train)

            # Transform fold val
            val_scores = encoder.transform(X_fold_val)

            # Store
            scores[val_idx] = val_scores["community_generosity_score"].values

        return pd.DataFrame({"community_generosity_score": scores}, index=X.index)


class MetadataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Handles numerical metadata: Selection, Imputation, Scaling.
    """

    def __init__(self, metadata_cols=METADATA_COLS):
        # We filter out the computed score column if it's in the list,
        # as it is handled separately, but we will append it later.
        # Ideally, METADATA_COLS in config contains static cols.
        self.cols = [c for c in metadata_cols if c != "community_generosity_score"]
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        # Select columns
        X_sel = X[self.cols]

        # Fit imputer
        self.imputer.fit(X_sel)
        X_imputed = self.imputer.transform(X_sel)

        # Fit scaler
        self.scaler.fit(X_imputed)
        return self

    def transform(self, X):
        X_sel = X[self.cols]
        X_imputed = self.imputer.transform(X_sel)
        X_scaled = self.scaler.transform(X_imputed)

        return pd.DataFrame(X_scaled, columns=self.cols, index=X.index)


class Preprocessor:
    """
    Orchestrates the data processing pipeline.
    """

    def __init__(self):
        self.train_path = os.path.join(METADATA_DIR, "train.parquet")
        self.val_path = os.path.join(METADATA_DIR, "val.parquet")
        self.test_path = os.path.join(METADATA_DIR, "test.parquet")

    def _load_raw(self):
        train_df = pd.read_parquet(self.train_path)
        val_df = pd.read_parquet(self.val_path)
        test_df = pd.read_parquet(self.test_path)
        return train_df, val_df, test_df

    def run(self, load_cached_data=True):
        """
        Main execution method.
        Returns:
            train_df, val_df, test_df (Processed DataFrames)
        """
        # Check cache
        if load_cached_data:
            logger.info("Checking cache for processed data...")
            train_proc = load_from_cache("train_processed.parquet")
            val_proc = load_from_cache("val_processed.parquet")
            test_proc = load_from_cache("test_processed.parquet")

            if (
                train_proc is not None
                and val_proc is not None
                and test_proc is not None
            ):
                logger.info("Loaded processed data from cache.")
                return train_proc, val_proc, test_proc
            else:
                logger.info("Cache miss. Processing from scratch...")

        # Load raw
        train_raw, val_raw, test_raw = self._load_raw()

        # --- 1. Text Processing ---
        logger.info("Processing text...")
        text_cleaner = TextCleaner()
        train_text = text_cleaner.transform(train_raw)
        val_text = text_cleaner.transform(val_raw)
        test_text = text_cleaner.transform(test_raw)

        # --- 2. Community Generosity Score ---
        logger.info("Generating Community Generosity Scores...")

        # A. Train: Nested CV
        comm_encoder_nested = CommunityTargetEncoder()
        train_comm_score = comm_encoder_nested.fit_transform_nested(
            train_raw, train_raw[TARGET_COL], n_folds=N_FOLDS
        )

        # B. Val: Fit on Train, Transform Val
        comm_encoder_val = CommunityTargetEncoder()
        comm_encoder_val.fit(train_raw, train_raw[TARGET_COL])
        val_comm_score = comm_encoder_val.transform(val_raw)

        # C. Test: Fit on Train + Val (Full Train), Transform Test
        # Concatenate train and val for fitting
        full_train_raw = pd.concat([train_raw, val_raw], ignore_index=True)
        comm_encoder_test = CommunityTargetEncoder()
        comm_encoder_test.fit(full_train_raw, full_train_raw[TARGET_COL])
        test_comm_score = comm_encoder_test.transform(test_raw)

        # --- 3. Metadata Processing ---
        logger.info("Processing metadata...")

        # A. Train/Val: Fit on Train, Transform both
        meta_proc_train = MetadataPreprocessor()
        meta_proc_train.fit(train_raw)
        train_meta = meta_proc_train.transform(train_raw)
        val_meta = meta_proc_train.transform(val_raw)

        # B. Test: Fit on Train + Val, Transform Test
        meta_proc_test = MetadataPreprocessor()
        meta_proc_test.fit(full_train_raw)
        test_meta = meta_proc_test.transform(test_raw)

        # --- 4. Assembly ---
        logger.info("Assembling final datasets...")

        def assemble(raw_df, text_df, comm_df, meta_df, include_target=False):
            # Start with ID
            out = pd.DataFrame({ID_COL: raw_df[ID_COL]})

            # Add Target if requested
            if include_target and TARGET_COL in raw_df.columns:
                out[TARGET_COL] = raw_df[TARGET_COL].values

            # Add Text
            out = pd.concat([out, text_df.reset_index(drop=True)], axis=1)

            # Add Community Score
            out = pd.concat([out, comm_df.reset_index(drop=True)], axis=1)

            # Add Metadata
            out = pd.concat([out, meta_df.reset_index(drop=True)], axis=1)

            # Pass through original subreddit list for sparse behavioral bagger
            # We convert list to string representation or keep as object if parquet supports it.
            # Parquet supports lists.
            out[SUBREDDIT_COL] = raw_df[SUBREDDIT_COL].reset_index(drop=True)

            return out

        train_processed = assemble(
            train_raw, train_text, train_comm_score, train_meta, include_target=True
        )
        val_processed = assemble(
            val_raw, val_text, val_comm_score, val_meta, include_target=True
        )
        test_processed = assemble(
            test_raw, test_text, test_comm_score, test_meta, include_target=False
        )

        # --- 5. Save to Cache ---
        logger.info("Saving to cache...")
        save_to_cache(train_processed, "train_processed.parquet")
        save_to_cache(val_processed, "val_processed.parquet")
        save_to_cache(test_processed, "test_processed.parquet")

        return train_processed, val_processed, test_processed
