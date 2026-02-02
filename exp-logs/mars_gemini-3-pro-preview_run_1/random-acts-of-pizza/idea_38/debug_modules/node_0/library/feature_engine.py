import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from library.config import Config


class MetadataProcessor:
    """
    Handles processing of numerical metadata and text-derived meta-features.
    Generates two versions: one for Random Forest (raw/imputed) and one for MLP (arcsinh/scaled).
    """

    def __init__(self, cache_dir=Config.WORKING_DIR):
        self.cache_dir = cache_dir
        self.numeric_cols = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_comments_in_raop_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_posts_on_raop_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request",
        ]

    def _extract_base_features(self, df):
        """Extracts raw numericals and engineers text meta-features."""
        # Select numeric columns, filling missing cols with 0 if they don't exist (safety)
        data = pd.DataFrame()
        for col in self.numeric_cols:
            if col in df.columns:
                data[col] = df[col]
            else:
                data[col] = 0.0

        # Engineer Text Features
        text_col = Config.TEXT_COL_BODY
        texts = df[text_col].fillna("").astype(str)

        data["text_len_chars"] = texts.apply(len)
        data["text_word_count"] = texts.apply(lambda x: len(x.split()))

        def get_caps_ratio(s):
            if len(s) == 0:
                return 0.0
            return sum(1 for c in s if c.isupper()) / len(s)

        data["text_caps_ratio"] = texts.apply(get_caps_ratio)

        return data

    def process(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Main processing pipeline.
        Returns:
            rf_features: (train, val, test) dictionary of numpy arrays
            mlp_features: (train, val, test) dictionary of numpy arrays
        """
        os.makedirs(self.cache_dir, exist_ok=True)

        # Define cache paths
        rf_cache_path = os.path.join(self.cache_dir, "rf_features.npz")
        mlp_cache_path = os.path.join(self.cache_dir, "mlp_features.npz")

        # Check cache
        if (
            load_cached_data
            and os.path.exists(rf_cache_path)
            and os.path.exists(mlp_cache_path)
        ):
            rf_data = np.load(rf_cache_path)
            mlp_data = np.load(mlp_cache_path)
            return (
                (rf_data["train"], rf_data["val"], rf_data["test"]),
                (mlp_data["train"], mlp_data["val"], mlp_data["test"]),
            )

        # 1. Extract Base Features
        train_base = self._extract_base_features(train_df)
        val_base = self._extract_base_features(val_df)
        test_base = self._extract_base_features(test_df)

        # 2. Imputation (Median from Train)
        # We use a simple approach: calculate medians on train, apply to all
        medians = train_base.median()
        train_filled = train_base.fillna(medians)
        val_filled = val_base.fillna(medians)
        test_filled = test_base.fillna(medians)

        # 3. Prepare RF Features (Raw/Imputed)
        rf_train = train_filled.values.astype(np.float32)
        rf_val = val_filled.values.astype(np.float32)
        rf_test = test_filled.values.astype(np.float32)

        # 4. Prepare MLP Features (Arcsinh + StandardScaler)
        # Apply Arcsinh to handle skew and zeros
        train_log = np.arcsinh(train_filled)
        val_log = np.arcsinh(val_filled)
        test_log = np.arcsinh(test_filled)

        # Standard Scaling (Fit on Train)
        scaler = StandardScaler()
        mlp_train = scaler.fit_transform(train_log)
        mlp_val = scaler.transform(val_log)
        mlp_test = scaler.transform(test_log)

        # 5. Save to Cache
        np.savez(rf_cache_path, train=rf_train, val=rf_val, test=rf_test)
        np.savez(mlp_cache_path, train=mlp_train, val=mlp_val, test=mlp_test)

        return (rf_train, rf_val, rf_test), (mlp_train, mlp_val, mlp_test)


class HistoryProcessor:
    """
    Handles subreddit history features.
    Generates Top-K binary flags for the Random Forest.
    """

    def __init__(self, top_k=Config.TOP_K_SUBREDDITS, cache_dir=Config.WORKING_DIR):
        self.top_k = top_k
        self.cache_dir = cache_dir

    def process(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Generates binary indicator matrices for top-k subreddits.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, "history_topk.npz")

        if load_cached_data and os.path.exists(cache_path):
            data = np.load(cache_path)
            return data["train"], data["val"], data["test"]

        # 1. Identify Top-K Subreddits from Train
        all_subreddits = []
        col_name = "requester_subreddits_at_request"

        # Iterate safely
        for sub_list in train_df[col_name]:
            if isinstance(sub_list, (list, np.ndarray)):
                all_subreddits.extend(sub_list)

        counts = Counter(all_subreddits)
        top_k_subs = [sub for sub, _ in counts.most_common(self.top_k)]

        # Map sub to index
        sub_to_idx = {sub: i for i, sub in enumerate(top_k_subs)}

        # 2. Transform function
        def transform(df):
            N = len(df)
            K = len(top_k_subs)
            matrix = np.zeros((N, K), dtype=np.float32)

            for i, sub_list in enumerate(df[col_name]):
                if isinstance(sub_list, (list, np.ndarray)):
                    for sub in sub_list:
                        if sub in sub_to_idx:
                            matrix[i, sub_to_idx[sub]] = 1.0
            return matrix

        train_mat = transform(train_df)
        val_mat = transform(val_df)
        test_mat = transform(test_df)

        # 3. Save
        np.savez(cache_path, train=train_mat, val=val_mat, test=test_mat)

        return train_mat, val_mat, test_mat


class PrototypeComputer:
    """
    Computes Semantic Prototype Scores.
    Calculates centroids for Success/Fail Requests and Histories on Train.
    Computes cosine similarity for all samples against these centroids.
    """

    def __init__(self, cache_dir=Config.WORKING_DIR):
        self.cache_dir = cache_dir

    def _compute_mean_history(self, hist_emb, mask):
        """
        Collapses (N, T, D) history embeddings to (N, D) by averaging valid tokens.
        """
        # hist_emb: (N, T, D), mask: (N, T)
        # Sum over T
        # Expand mask to (N, T, 1)
        mask_expanded = mask[:, :, np.newaxis].astype(np.float32)

        sum_emb = np.sum(hist_emb * mask_expanded, axis=1)  # (N, D)
        count = np.sum(mask_expanded, axis=1)  # (N, 1)

        # Avoid div by zero
        count = np.maximum(count, 1.0)

        return sum_emb / count

    def process(
        self,
        train_req,
        train_hist,
        train_hist_mask,
        train_y,
        val_req,
        val_hist,
        val_hist_mask,
        test_req,
        test_hist,
        test_hist_mask,
        load_cached_data=True,
    ):
        """
        Computes the 4 prototype scores for each dataset.
        Returns: (train_scores, val_scores, test_scores)
        Each is (N, 4).
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, "prototype_scores.npz")

        if load_cached_data and os.path.exists(cache_path):
            data = np.load(cache_path)
            return data["train"], data["val"], data["test"]

        # 1. Prepare Mean History Vectors
        train_hist_mean = self._compute_mean_history(train_hist, train_hist_mask)
        val_hist_mean = self._compute_mean_history(val_hist, val_hist_mask)
        test_hist_mean = self._compute_mean_history(test_hist, test_hist_mask)

        # 2. Compute Centroids (Train Only)
        # Ensure train_y is boolean or 0/1
        y = np.array(train_y).astype(bool)

        # Request Centroids
        req_pos_centroid = np.mean(train_req[y], axis=0).reshape(1, -1)
        req_neg_centroid = np.mean(train_req[~y], axis=0).reshape(1, -1)

        # History Centroids
        # Handle edge case where no positive/negative samples exist (unlikely but safe)
        if np.sum(y) > 0:
            hist_pos_centroid = np.mean(train_hist_mean[y], axis=0).reshape(1, -1)
        else:
            hist_pos_centroid = np.zeros((1, train_hist_mean.shape[1]))

        if np.sum(~y) > 0:
            hist_neg_centroid = np.mean(train_hist_mean[~y], axis=0).reshape(1, -1)
        else:
            hist_neg_centroid = np.zeros((1, train_hist_mean.shape[1]))

        # 3. Compute Similarities
        def get_scores(req_emb, hist_emb_mean):
            # Cosine similarity returns (N, 1)
            s1 = cosine_similarity(req_emb, req_pos_centroid)
            s2 = cosine_similarity(req_emb, req_neg_centroid)
            s3 = cosine_similarity(hist_emb_mean, hist_pos_centroid)
            s4 = cosine_similarity(hist_emb_mean, hist_neg_centroid)
            return np.hstack([s1, s2, s3, s4])

        train_scores = get_scores(train_req, train_hist_mean)
        val_scores = get_scores(val_req, val_hist_mean)
        test_scores = get_scores(test_req, test_hist_mean)

        # 4. Save
        np.savez(cache_path, train=train_scores, val=val_scores, test=test_scores)

        return train_scores, val_scores, test_scores
