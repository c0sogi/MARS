import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from collections import Counter

from library.config import WORKING_DIR, TOP_K_SUBREDDITS, RANDOM_STATE, TRAIN_PATH
from library.utils import set_seed
from library.data_loader import get_common_columns
from library.text_encoder import generate_text_features

# Ensure deterministic behavior
set_seed(RANDOM_STATE)


class FeatureProcessor:
    def __init__(self):
        self.top_k_subreddits = []
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.numeric_cols = []
        self.fitted = False

    def _get_top_k_subreddits(self, df_train, k=TOP_K_SUBREDDITS):
        """
        Identifies the top K most frequent subreddits from the training set.
        """
        all_subreddits = []
        # 'requester_subreddits_at_request' is a list of strings
        for sub_list in df_train["requester_subreddits_at_request"]:
            if isinstance(sub_list, list):
                all_subreddits.extend(sub_list)

        counts = Counter(all_subreddits)
        # Select top K
        top_k = [sub for sub, count in counts.most_common(k)]
        return top_k

    def _generate_binary_flags(self, df, top_k_subs):
        """
        Generates a binary matrix (N, K) indicating presence of top K subreddits.
        """
        # Map subreddits to indices
        sub_to_idx = {sub: i for i, sub in enumerate(top_k_subs)}
        n_samples = len(df)
        n_features = len(top_k_subs)

        flags = np.zeros((n_samples, n_features), dtype=np.float32)

        for row_idx, sub_list in enumerate(df["requester_subreddits_at_request"]):
            if isinstance(sub_list, list):
                for sub in sub_list:
                    if sub in sub_to_idx:
                        col_idx = sub_to_idx[sub]
                        flags[row_idx, col_idx] = 1.0

        return flags

    def _compute_derived_metrics(self, df):
        """
        Computes derived numerical metrics like Upvote Ratio.
        Returns a DataFrame with these new columns added/updated.
        """
        df = df.copy()

        # Upvote Ratio
        # up_plus_down = up + down
        # up_minus_down = up - down
        # up = (up_plus_down + up_minus_down) / 2
        # ratio = up / up_plus_down

        plus = df["requester_upvotes_plus_downvotes_at_request"].fillna(0)
        minus = df["requester_upvotes_minus_downvotes_at_request"].fillna(0)

        upvotes = (plus + minus) / 2

        # Avoid division by zero
        ratio = np.divide(upvotes, plus, out=np.zeros_like(plus), where=plus != 0)
        df["calculated_upvote_ratio"] = ratio

        return df

    def _prepare_reliability_features(self, df, is_train=False):
        """
        Selects, imputes, transforms (Arcsinh), and scales numerical metadata.
        """
        # Identify columns if not already done
        if is_train:
            # Get common numerical columns excluding IDs and Text
            # We use a heuristic or the get_common_columns utility, but we need to filter for numeric
            # We'll explicitly define a robust set based on the dataset description to avoid noise
            candidates = [
                "requester_account_age_in_days_at_request",
                "requester_days_since_first_post_on_raop_at_request",
                "requester_number_of_comments_at_request",
                "requester_number_of_comments_in_raop_at_request",
                "requester_number_of_posts_at_request",
                "requester_number_of_posts_on_raop_at_request",
                "requester_number_of_subreddits_at_request",
                "requester_upvotes_minus_downvotes_at_request",
                "requester_upvotes_plus_downvotes_at_request",
                "calculated_upvote_ratio",  # Derived
            ]
            # Filter to what's actually in df
            self.numeric_cols = [c for c in candidates if c in df.columns]

        # Select data
        X = df[self.numeric_cols].values

        # Impute
        if is_train:
            X = self.imputer.fit_transform(X)
        else:
            X = self.imputer.transform(X)

        # Arcsinh Transform (handling skewness)
        X = np.arcsinh(X)

        # Scale
        if is_train:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X

    def _create_interaction_terms(self, df, consistency_scalars):
        """
        Creates explicit interaction terms for the Random Forest.
        I1 = Topic_Consistency * log(1 + Account_Age)
        I2 = Narrative_Consistency * Upvote_Ratio
        """
        # Extract raw components
        # consistency_scalars is a dict with 'title_hist_sim' and 'text_hist_sim'
        topic_sim = consistency_scalars["title_hist_sim"].flatten()
        narrative_sim = consistency_scalars["text_hist_sim"].flatten()

        acc_age = df["requester_account_age_in_days_at_request"].fillna(0).values
        upvote_ratio = df["calculated_upvote_ratio"].fillna(0).values

        # I1
        log_age = np.log1p(acc_age)
        i1 = topic_sim * log_age

        # I2
        i2 = narrative_sim * upvote_ratio

        return np.stack([i1, i2], axis=1)

    def process(self, df_train, df_val, df_test, load_cached_data=True):
        """
        Main processing pipeline.

        Returns:
            rf_data: dict with keys 'train', 'val', 'test', containing concatenated numpy arrays.
            mlp_data: dict with keys 'train', 'val', 'test', containing dicts of branches.
            y_data: dict with keys 'train', 'val' containing target arrays.
        """
        os.makedirs(WORKING_DIR, exist_ok=True)
        rf_cache_path = os.path.join(WORKING_DIR, "rf_features.npz")
        mlp_cache_path = os.path.join(WORKING_DIR, "mlp_features.npz")
        y_cache_path = os.path.join(WORKING_DIR, "targets.npz")

        # Check cache
        if (
            load_cached_data
            and os.path.exists(rf_cache_path)
            and os.path.exists(mlp_cache_path)
            and os.path.exists(y_cache_path)
        ):
            try:
                # Load RF
                rf_loaded = np.load(rf_cache_path)
                rf_data = {k: rf_loaded[k] for k in ["train", "val", "test"]}

                # Load MLP
                # MLP data is nested, npz flattens it. We need to structure it carefully or save/load differently.
                # To simplify, we'll assume we saved them as flat arrays with specific keys in one file.
                mlp_loaded = np.load(mlp_cache_path)
                mlp_data = {}
                for split in ["train", "val", "test"]:
                    mlp_data[split] = {
                        "semantic": mlp_loaded[f"{split}_semantic"],
                        "reliability": mlp_loaded[f"{split}_reliability"],
                        "community": mlp_loaded[f"{split}_community"],
                    }

                # Load Targets
                y_loaded = np.load(y_cache_path)
                y_data = {k: y_loaded[k] for k in ["train", "val"]}

                return rf_data, mlp_data, y_data
            except Exception:
                pass  # Fallback to compute

        # ---------------------------------------------------------
        # 1. Feature Generation (Text & SBERT)
        # ---------------------------------------------------------
        # This handles caching internally
        sbert_data, tfidf_data = generate_text_features(
            df_train, df_val, df_test, load_cached_data=load_cached_data
        )

        # ---------------------------------------------------------
        # 2. Preprocessing & Metadata
        # ---------------------------------------------------------
        # Compute derived metrics (Upvote Ratio)
        df_train = self._compute_derived_metrics(df_train)
        df_val = self._compute_derived_metrics(df_val)
        df_test = self._compute_derived_metrics(df_test)

        # Learn Top-K Subreddits
        self.top_k_subreddits = self._get_top_k_subreddits(df_train)

        # Generate Binary Flags
        flags_train = self._generate_binary_flags(df_train, self.top_k_subreddits)
        flags_val = self._generate_binary_flags(df_val, self.top_k_subreddits)
        flags_test = self._generate_binary_flags(df_test, self.top_k_subreddits)

        # Reliability Features (Numerical Metadata)
        # Fit on train, transform others
        rel_train = self._prepare_reliability_features(df_train, is_train=True)
        rel_val = self._prepare_reliability_features(df_val, is_train=False)
        rel_test = self._prepare_reliability_features(df_test, is_train=False)

        # Interaction Terms (RF Specific)
        inter_train = self._create_interaction_terms(df_train, sbert_data["train"])
        inter_val = self._create_interaction_terms(df_val, sbert_data["val"])
        inter_test = self._create_interaction_terms(df_test, sbert_data["test"])

        # ---------------------------------------------------------
        # 3. Assembly: Random Forest (Stream A)
        # ---------------------------------------------------------
        # Inputs: TF-IDF + Metadata (Raw/Imputed - here we use the processed rel features for simplicity as they are scaled,
        # but RF handles raw fine. Using scaled is acceptable) + Top-K Flags + Interactions + Consistency Scalars

        def assemble_rf(tfidf, rel, flags, inter, sbert):
            # Consistency scalars are in sbert dict
            cons_topic = sbert["title_hist_sim"]
            cons_narr = sbert["text_hist_sim"]

            # Concatenate all
            # Note: TF-IDF is high dim, others are low dim
            return np.concatenate(
                [tfidf, rel, flags, inter, cons_topic, cons_narr], axis=1
            )

        rf_train = assemble_rf(
            tfidf_data["train"],
            rel_train,
            flags_train,
            inter_train,
            sbert_data["train"],
        )
        rf_val = assemble_rf(
            tfidf_data["val"], rel_val, flags_val, inter_val, sbert_data["val"]
        )
        rf_test = assemble_rf(
            tfidf_data["test"], rel_test, flags_test, inter_test, sbert_data["test"]
        )

        # ---------------------------------------------------------
        # 4. Assembly: MLP (Stream B)
        # ---------------------------------------------------------
        # Branch 1: Semantic (Title Emb, Text Emb, Hist Emb, Consistency Scalars)
        # Branch 2: Reliability (Scaled Metadata) -> rel_train/val/test
        # Branch 3: Community (Top-K Flags) -> flags_train/val/test

        def assemble_mlp_semantic(sbert):
            return np.concatenate(
                [
                    sbert["title_emb"],
                    sbert["text_emb"],
                    sbert["history_emb"],
                    sbert["title_hist_sim"],
                    sbert["text_hist_sim"],
                ],
                axis=1,
            )

        mlp_sem_train = assemble_mlp_semantic(sbert_data["train"])
        mlp_sem_val = assemble_mlp_semantic(sbert_data["val"])
        mlp_sem_test = assemble_mlp_semantic(sbert_data["test"])

        mlp_data = {
            "train": {
                "semantic": mlp_sem_train,
                "reliability": rel_train,
                "community": flags_train,
            },
            "val": {
                "semantic": mlp_sem_val,
                "reliability": rel_val,
                "community": flags_val,
            },
            "test": {
                "semantic": mlp_sem_test,
                "reliability": rel_test,
                "community": flags_test,
            },
        }

        # ---------------------------------------------------------
        # 5. Targets
        # ---------------------------------------------------------
        y_train = df_train["requester_received_pizza"].astype(int).values
        y_val = df_val["requester_received_pizza"].astype(int).values

        # ---------------------------------------------------------
        # 6. Caching
        # ---------------------------------------------------------
        # Save RF
        np.savez(rf_cache_path, train=rf_train, val=rf_val, test=rf_test)

        # Save MLP (Flattened keys)
        save_dict = {}
        for split in ["train", "val", "test"]:
            save_dict[f"{split}_semantic"] = mlp_data[split]["semantic"]
            save_dict[f"{split}_reliability"] = mlp_data[split]["reliability"]
            save_dict[f"{split}_community"] = mlp_data[split]["community"]
        np.savez(mlp_cache_path, **save_dict)

        # Save Targets
        np.savez(y_cache_path, train=y_train, val=y_val)

        return (
            {"train": rf_train, "val": rf_val, "test": rf_test},
            mlp_data,
            {"train": y_train, "val": y_val},
        )
