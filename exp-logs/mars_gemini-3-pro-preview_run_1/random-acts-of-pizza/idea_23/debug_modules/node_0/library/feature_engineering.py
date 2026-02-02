import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from library import config, data_loader


class MetadataEngineer:
    """
    Extracts 'Full-Spectrum' metadata including raw magnitudes, engineered ratios,
    and text meta-features.
    """

    def __init__(self):
        pass

    def transform(self, df):
        """
        Generates metadata features for the provided dataframe.
        """
        df_out = pd.DataFrame(index=df.index)

        # --- 1. Text Meta-Features ---
        # Use edit-aware text if available, else standard text
        text_col = config.TEXT_COL_BODY
        if text_col not in df.columns:
            # Fallback if specific column missing
            text_col = "request_text"

        if text_col in df.columns:
            texts = df[text_col].fillna("").astype(str)

            # Character Count
            df_out["meta_text_len_chars"] = texts.apply(len)

            # Word Count
            df_out["meta_text_len_words"] = texts.apply(lambda x: len(x.split()))

            # Caps Ratio
            def calc_caps_ratio(s):
                if len(s) == 0:
                    return 0.0
                return sum(1 for c in s if c.isupper()) / len(s)

            df_out["meta_caps_ratio"] = texts.apply(calc_caps_ratio)
        else:
            # Fill with 0 if text column completely missing
            df_out["meta_text_len_chars"] = 0
            df_out["meta_text_len_words"] = 0
            df_out["meta_caps_ratio"] = 0.0

        # --- 2. Engineered Ratios ---
        # Upvote Ratio: U / (U+D)
        # We use 'requester_upvotes_plus_downvotes_at_request' (Sum) and 'minus' (Diff)
        # U = (Sum + Diff) / 2
        # Ratio = U / Sum = (Sum + Diff) / (2 * Sum) = 0.5 * (1 + Diff/Sum)

        col_plus = "requester_upvotes_plus_downvotes_at_request"
        col_minus = "requester_upvotes_minus_downvotes_at_request"

        if col_plus in df.columns and col_minus in df.columns:
            plus = df[col_plus].fillna(0)
            minus = df[col_minus].fillna(0)

            # Calculate ratio, handling division by zero
            # If plus (total votes) is 0, we assume neutral ratio (0.5)
            ratio = 0.5 * (1 + minus / plus)
            df_out["meta_upvote_ratio"] = ratio.fillna(0.5)
        else:
            df_out["meta_upvote_ratio"] = 0.5

        # --- 3. Raw Numeric Columns ---
        # Include raw numeric columns defined in config
        for col in config.NUMERIC_COLS:
            if col in df.columns:
                # Fill NaNs with median or 0. Using 0 as safe default for sparse counts.
                df_out[col] = df[col].fillna(0)
            else:
                df_out[col] = 0

        return df_out


class MultiObjectiveTargetEncoder:
    """
    Implements Multi-Objective Bayesian Target Encoding on subreddit history.
    Calculates Generosity, Engagement, and Toxicity scores.
    Uses Stratified K-Fold for Train (OOF) and Global Maps for Test.
    """

    def __init__(self, folds=5, smoothing=10.0):
        self.folds = folds
        self.smoothing = smoothing
        self.target_map = config.TE_TARGET_MAP
        self.global_maps = {}  # {target_name: {subreddit: score}}
        self.global_priors = {}  # {target_name: global_mean}

    def _compute_stats(self, df, target_col):
        """
        Computes count and sum of target variable for each subreddit.
        """
        # Filter rows where target is valid
        df_clean = df.dropna(subset=[target_col])

        # Explode the history list column to get (subreddit, target) pairs
        # Ensure the column exists
        if config.HISTORY_COL not in df_clean.columns:
            return pd.DataFrame(), df_clean[target_col].mean()

        exploded = df_clean[[config.HISTORY_COL, target_col]].explode(
            config.HISTORY_COL
        )

        # Group by subreddit
        stats = exploded.groupby(config.HISTORY_COL)[target_col].agg(["count", "sum"])

        # Global mean (prior)
        global_mean = df_clean[target_col].mean()

        return stats, global_mean

    def _smooth(self, stats, global_mean):
        """
        Applies Bayesian smoothing to subreddit stats.
        Score = (sum + alpha * global_mean) / (count + alpha)
        """
        if stats.empty:
            return pd.Series(dtype=float).to_dict()

        smoothed = (stats["sum"] + self.smoothing * global_mean) / (
            stats["count"] + self.smoothing
        )
        return smoothed.to_dict()

    def fit(self, df_train):
        """
        Computes global maps on the full training set (used for Val/Test).
        """
        for name, target_col in self.target_map.items():
            if target_col not in df_train.columns:
                # Skip if target column missing (e.g. engagement/toxicity might be missing in some cleaned versions)
                # But typically present in train.json
                continue

            stats, global_mean = self._compute_stats(df_train, target_col)
            mapping = self._smooth(stats, global_mean)

            self.global_maps[name] = mapping
            self.global_priors[name] = global_mean

    def _transform_subset(self, df, maps, priors):
        """
        Applies mappings to a dataframe.
        Aggregates subreddit scores per user (Mean, Max).
        """
        result = pd.DataFrame(index=df.index)

        # Initialize columns with priors (default value)
        for name in maps.keys():
            result[f"te_{name}_mean"] = priors.get(name, 0.0)
            result[f"te_{name}_max"] = priors.get(name, 0.0)

        if len(df) == 0:
            return result

        # Explode history
        if config.HISTORY_COL not in df.columns:
            return result

        exploded = df[config.HISTORY_COL].explode()

        # If exploded is empty (e.g. all lists were empty), return defaults
        if exploded.empty:
            return result

        for name, mapping in maps.items():
            prior = priors.get(name, 0.0)

            # Map subreddits to scores
            # Subreddits not in map get the global prior
            mapped_values = exploded.map(mapping).fillna(prior)

            # Group back by index (user)
            # Note: Users with empty history lists are dropped by explode.
            # We must reindex result at the end to include them.
            grouped = mapped_values.groupby(mapped_values.index)

            agg_mean = grouped.mean()
            agg_max = grouped.max()

            # Update result dataframe
            # fillna(prior) handles users who were dropped (empty history)
            result[f"te_{name}_mean"] = agg_mean
            result[f"te_{name}_max"] = agg_max

            result[f"te_{name}_mean"] = result[f"te_{name}_mean"].fillna(prior)
            result[f"te_{name}_max"] = result[f"te_{name}_max"].fillna(prior)

        return result

    def transform_train(self, df_train):
        """
        Generates Out-Of-Fold (OOF) features for the training set to prevent leakage.
        """
        skf = StratifiedKFold(
            n_splits=self.folds, shuffle=True, random_state=config.RANDOM_SEED
        )

        # Prepare output container
        feature_names = []
        for name in self.target_map.keys():
            feature_names.extend([f"te_{name}_mean", f"te_{name}_max"])

        oof_features = pd.DataFrame(index=df_train.index, columns=feature_names)

        # Target for stratification
        y = df_train[config.TARGET_COL].astype(int)

        for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, y)):
            train_sub = df_train.iloc[train_idx]
            val_sub = df_train.iloc[val_idx]

            # Compute maps on this fold's training subset
            fold_maps = {}
            fold_priors = {}

            for name, t_col in self.target_map.items():
                if t_col in train_sub.columns:
                    stats, mean = self._compute_stats(train_sub, t_col)
                    fold_maps[name] = self._smooth(stats, mean)
                    fold_priors[name] = mean

            # Transform this fold's validation subset
            feats = self._transform_subset(val_sub, fold_maps, fold_priors)

            # Assign to OOF dataframe
            oof_features.iloc[val_idx] = feats

        return oof_features.astype(float)

    def transform_test(self, df_test):
        """
        Transforms test/validation data using the global maps learned in fit().
        """
        return self._transform_subset(df_test, self.global_maps, self.global_priors)


def generate_features(load_cached_data=True):
    """
    Main function to generate features.
    Orchestrates loading, Metadata Engineering, Target Encoding, and Caching.
    """
    # Cache paths
    cache_file_train = os.path.join(config.WORKING_DIR, "features_train.parquet")
    cache_file_val = os.path.join(config.WORKING_DIR, "features_val.parquet")
    cache_file_test = os.path.join(config.WORKING_DIR, "features_test.parquet")

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(cache_file_train)
            and os.path.exists(cache_file_val)
            and os.path.exists(cache_file_test)
        ):
            print("Loading engineered features from cache...")
            try:
                X_train = pd.read_parquet(cache_file_train)
                X_val = pd.read_parquet(cache_file_val)
                X_test = pd.read_parquet(cache_file_test)
                return X_train, X_val, X_test
            except Exception as e:
                print(f"Cache load failed: {e}. Regenerating...")

    print("Generating features from scratch...")

    # 2. Load Raw Data
    df_train, df_val, df_test = data_loader.load_tabular_data(
        load_cached_data=load_cached_data
    )

    # 3. Metadata Engineering (Row-wise)
    me = MetadataEngineer()
    meta_train = me.transform(df_train)
    meta_val = me.transform(df_val)
    meta_test = me.transform(df_test)

    # 4. Multi-Objective Target Encoding
    # Initialize
    te = MultiObjectiveTargetEncoder(
        folds=config.TE_FOLDS, smoothing=config.TE_SMOOTHING_ALPHA
    )

    # Fit Global Maps (used for Val and Test)
    te.fit(df_train)

    # Transform Train (OOF)
    print("Generating OOF Target Encodings for Train...")
    te_train = te.transform_train(df_train)

    # Transform Val & Test (Global)
    print("Generating Target Encodings for Val/Test...")
    te_val = te.transform_test(df_val)
    te_test = te.transform_test(df_test)

    # 5. Concatenate
    X_train = pd.concat([meta_train, te_train], axis=1)
    X_val = pd.concat([meta_val, te_val], axis=1)
    X_test = pd.concat([meta_test, te_test], axis=1)

    # 6. Save to Cache
    print("Saving features to cache...")
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    X_train.to_parquet(cache_file_train)
    X_val.to_parquet(cache_file_val)
    X_test.to_parquet(cache_file_test)

    return X_train, X_val, X_test
