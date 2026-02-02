import os
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from datetime import timedelta
from library import config, data_loader, retrieval


class RankerDatasetGenerator:
    """
    Generates the training and validation datasets for the Stage 2 LightGBM Ranker.
    Uses a time-split strategy on the 'val_metadata' users to avoid leakage from the
    graph structures built on 'train_metadata'.
    """

    def __init__(self, retriever):
        """
        Args:
            retriever (SparseRetriever): Initialized retriever instance containing T_seq, T_vis.
        """
        self.retriever = retriever
        self.train_users = None
        self.val_users = None

        # Load Article and Customer mappings for feature engineering
        self.articles_df = data_loader.load_articles(load_cached_data=True)
        self.customers_df = data_loader.load_customers(load_cached_data=True)

        # Pre-compute global popularity from training data for fallback
        # (retriever already has this, but we might want it as a feature)
        self.pop_map = self._compute_global_popularity_map()

    def _compute_global_popularity_map(self):
        """Computes a map of article_id -> log(purchase_count) from training data."""
        # We can approximate this using the diagonal of the transition matrix or U_weighted sum
        # But simpler to just use the retriever's U_weighted
        pop_scores = np.array(self.retriever.U_weighted.sum(axis=0)).flatten()
        # Normalize to log scale
        pop_scores = np.log1p(pop_scores)

        # Map index to score
        pop_map = {}
        for idx, score in enumerate(pop_scores):
            aid = self.retriever.art_idx_to_id[idx]
            pop_map[aid] = score
        return pop_map

    def _build_custom_user_vectors(self, transactions_df, customer_ids):
        """
        Constructs sparse user vectors (U) for a specific set of transactions.
        Used to build history vectors for validation users dynamically.

        Args:
            transactions_df (pd.DataFrame): History transactions.
            customer_ids (list): List of customer_ids to include.

        Returns:
            sp.csr_matrix: Sparse matrix (N_users, N_items)
            sp.csr_matrix: Raw binary matrix (N_users, N_items)
        """
        # Filter transactions
        df = transactions_df[
            transactions_df["customer_id"].isin(customer_ids)
            & transactions_df["article_id"].isin(self.retriever.art_id_to_idx)
        ].copy()

        # Map IDs
        # We need a local map for the rows (users) to align with the output matrix
        cust_to_local_idx = {cid: i for i, cid in enumerate(customer_ids)}

        df["user_idx"] = df["customer_id"].map(cust_to_local_idx).astype(np.int32)
        df["item_idx"] = (
            df["article_id"].map(self.retriever.art_id_to_idx).astype(np.int32)
        )

        # Time Decay
        max_date = df["t_dat"].max()
        df["days_diff"] = (max_date - df["t_dat"]).dt.days
        df["weight"] = np.exp(-df["days_diff"] / config.TIME_DECAY_DAYS)

        # Build Weighted Matrix
        row_idx = df["user_idx"].values
        col_idx = df["item_idx"].values
        weights = df["weight"].values

        n_users = len(customer_ids)
        n_items = self.retriever.n_items

        u_weighted = sp.coo_matrix(
            (weights, (row_idx, col_idx)), shape=(n_users, n_items)
        ).tocsr()

        # Build Raw Matrix (for alpha term)
        ones = np.ones(len(weights))
        u_raw = sp.coo_matrix(
            (ones, (row_idx, col_idx)), shape=(n_users, n_items)
        ).tocsr()

        return u_weighted, u_raw

    def _generate_features_and_labels(self, u_weighted, u_raw, target_df, customer_ids):
        """
        Generates candidates, computes scores, and creates the labeled dataset.
        """
        # 1. Compute Scores: S = U_w @ T_seq + lambda * U_w @ T_vis + alpha * U_raw
        print("Computing retrieval scores...")
        s_seq = u_weighted.dot(self.retriever.T_seq)
        s_vis = u_weighted.dot(self.retriever.T_vis)

        # Linear Combination
        scores_sparse = (
            s_seq + (config.LAMBDA_VISUAL * s_vis) + (config.ALPHA_HISTORY * u_raw)
        )
        scores_dense = scores_sparse.toarray()

        # 2. Extract Top-K Candidates
        print(f"Extracting Top-{config.RETRIEVAL_TOP_K} candidates...")
        k = config.RETRIEVAL_TOP_K
        n_users = len(customer_ids)

        # Handle cases where we have fewer items than K
        if scores_dense.shape[1] < k:
            k = scores_dense.shape[1]

        # Get indices of top K
        # argpartition is faster than sort
        top_k_idx = np.argpartition(scores_dense, -k, axis=1)[:, -k:]

        # Get values to sort them properly (rank feature)
        rows = np.arange(n_users)[:, None]
        top_k_scores = scores_dense[rows, top_k_idx]

        # Sort descending
        sort_order = np.argsort(top_k_scores, axis=1)[:, ::-1]
        sorted_indices = top_k_idx[rows, sort_order]
        sorted_scores = top_k_scores[rows, sort_order]

        # 3. Prepare Ground Truth
        print("Preparing ground truth labels...")
        target_df = target_df[target_df["customer_id"].isin(customer_ids)]
        # Group purchases by customer
        ground_truth = (
            target_df.groupby("customer_id")["article_id"].apply(set).to_dict()
        )

        # 4. Build Dataset Rows
        print("Constructing dataset...")
        records = []

        # Pre-fetch article features to avoid repeated lookups
        art_cols = [
            "article_id",
            "product_type_no",
            "graphical_appearance_no",
            "colour_group_code",
            "perceived_colour_value_id",
            "department_no",
            "index_group_no",
            "section_no",
            "garment_group_no",
        ]
        art_feat_df = self.articles_df[art_cols].set_index("article_id")

        # Pre-fetch customer features
        cust_cols = [
            "customer_id",
            "age",
            "club_member_status",
            "fashion_news_frequency",
        ]
        cust_feat_df = self.customers_df[cust_cols].set_index("customer_id")

        # Iterate and build
        for i, cid in enumerate(customer_ids):
            # Get user features
            try:
                u_feat = cust_feat_df.loc[cid]
                age = u_feat["age"]
                club = u_feat["club_member_status"]
                news = u_feat["fashion_news_frequency"]
            except KeyError:
                age, club, news = -1, -1, -1  # Should not happen if data is consistent

            purchased_items = ground_truth.get(cid, set())

            # If user has no purchases in target week, skip (cannot train ranker)
            if not purchased_items:
                continue

            for rank, idx in enumerate(sorted_indices[i]):
                article_id = self.retriever.art_idx_to_id[idx]
                score = sorted_scores[i, rank]

                # Label
                label = 1 if article_id in purchased_items else 0

                # Article Features
                try:
                    a_feat = art_feat_df.loc[article_id]
                    # Convert to dict
                    rec = {
                        "customer_id": cid,
                        "article_id": article_id,
                        "label": label,
                        "retrieval_score": float(score),
                        "retrieval_rank": int(rank),
                        "age": int(age),
                        "club_member_status": str(club),
                        "fashion_news_frequency": str(news),
                        "global_popularity": self.pop_map.get(article_id, 0.0),
                        # Article Metadata
                        "product_type_no": int(a_feat["product_type_no"]),
                        "graphical_appearance_no": int(
                            a_feat["graphical_appearance_no"]
                        ),
                        "colour_group_code": int(a_feat["colour_group_code"]),
                        "perceived_colour_value_id": int(
                            a_feat["perceived_colour_value_id"]
                        ),
                        "department_no": int(a_feat["department_no"]),
                        "index_group_no": int(a_feat["index_group_no"]),
                        "section_no": int(a_feat["section_no"]),
                        "garment_group_no": int(a_feat["garment_group_no"]),
                    }
                    records.append(rec)
                except KeyError:
                    continue

        return pd.DataFrame(records)

    def create_dataset(self, load_cached_data=True):
        """
        Main method to create ranker training and validation sets.
        """
        # Paths
        train_path = config.RANKER_TRAIN_PATH
        val_path = config.RANKER_VAL_PATH

        if load_cached_data and train_path.exists() and val_path.exists():
            print("Loading cached ranker datasets...")
            return pd.read_parquet(train_path), pd.read_parquet(val_path)

        print("Generating ranker datasets from scratch using Validation Metadata...")

        # 1. Load Validation Metadata (Hold-out users)
        # We use these users because their history is NOT in the training graphs,
        # allowing us to simulate a clean inference step.
        val_df = data_loader.load_transactions("val", load_cached_data=True)

        # 2. Time Split for these users
        # Target: Last 7 days
        max_date = val_df["t_dat"].max()
        split_date = max_date - timedelta(days=7)

        print(f"Splitting Val Data at {split_date}...")
        history_df = val_df[val_df["t_dat"] <= split_date].copy()
        target_df = val_df[val_df["t_dat"] > split_date].copy()

        # Identify active users in the target period
        active_users = target_df["customer_id"].unique()
        print(f"Found {len(active_users)} active users in target period.")

        # 3. Split Users into Ranker_Train (90%) and Ranker_Val (10%)
        np.random.seed(config.SEED)
        shuffled_users = np.random.permutation(active_users)
        n_train = int(len(shuffled_users) * 0.9)

        train_uids = shuffled_users[:n_train]
        val_uids = shuffled_users[n_train:]

        print(f"Ranker Train Users: {len(train_uids)}")
        print(f"Ranker Val Users: {len(val_uids)}")

        # 4. Process Splits
        def process_split(uids, split_name):
            print(f"\nProcessing {split_name} split...")
            # Build U vectors
            u_weighted, u_raw = self._build_custom_user_vectors(history_df, uids)

            # Generate Data
            df = self._generate_features_and_labels(u_weighted, u_raw, target_df, uids)

            # Optimize Types
            # Categoricals
            cat_cols = ["club_member_status", "fashion_news_frequency"]
            for c in cat_cols:
                df[c] = df[c].astype("category")

            return df

        ranker_train_df = process_split(train_uids, "Train")
        ranker_val_df = process_split(val_uids, "Validation")

        # 5. Save
        print(f"Saving Ranker Train ({len(ranker_train_df)} rows) to {train_path}...")
        ranker_train_df.to_parquet(train_path, index=False)

        print(f"Saving Ranker Val ({len(ranker_val_df)} rows) to {val_path}...")
        ranker_val_df.to_parquet(val_path, index=False)

        return ranker_train_df, ranker_val_df


def create_ranking_dataset(load_cached_data=True):
    """
    Wrapper function to initialize retriever and generate datasets.
    """
    # Initialize Retriever (loads graphs)
    retriever = retrieval.SparseRetriever(load_cached_data=load_cached_data)

    # Initialize Generator
    generator = RankerDatasetGenerator(retriever)

    # Generate
    return generator.create_dataset(load_cached_data=load_cached_data)
