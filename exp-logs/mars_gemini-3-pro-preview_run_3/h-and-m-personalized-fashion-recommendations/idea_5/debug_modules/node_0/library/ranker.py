import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import scipy.sparse as sp
from library import config, data_loader, retrieval, feature_engineering

# Ensure working directory exists
os.makedirs(config.WORKING_DIR, exist_ok=True)


class Ranker:
    """
    Manages Stage 2: Feature-Constrained Re-Ranking.
    """

    def __init__(self):
        self.model_path = config.WORKING_DIR / "lgbm_ranker.txt"
        self.retriever = retrieval.SparseRetriever(load_cached_data=True)

        # Load metadata for feature engineering
        self.articles_df = data_loader.load_articles(load_cached_data=True)
        self.customers_df = data_loader.load_customers(load_cached_data=True)

        # Pre-compute Global Popularity Map (log scale) for feature consistency
        self.pop_map = self._compute_global_popularity_map()

        # Prepare fast lookup tables
        self._prepare_feature_lookups()

    def _compute_global_popularity_map(self):
        """Computes log(popularity) for all items based on training history."""
        pop_scores = np.array(self.retriever.U_weighted.sum(axis=0)).flatten()
        pop_scores = np.log1p(pop_scores)

        pop_map = {}
        for idx, score in enumerate(pop_scores):
            aid = self.retriever.art_idx_to_id[idx]
            pop_map[aid] = score
        return pop_map

    def _prepare_feature_lookups(self):
        """Prepares metadata dataframes for efficient merging during inference."""
        # Articles
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
        self.art_features = self.articles_df[art_cols].copy()
        # Ensure integer types for LGBM
        for c in art_cols[1:]:
            self.art_features[c] = self.art_features[c].astype(int)

        # Customers
        cust_cols = [
            "customer_id",
            "age",
            "club_member_status",
            "fashion_news_frequency",
        ]
        self.cust_features = self.customers_df[cust_cols].copy()
        # Note: Categoricals (club_member_status, fashion_news_frequency) are already
        # cast to 'category' in data_loader.

    def train(self, load_cached_model=True):
        """
        Trains the LightGBM Ranker.
        """
        if load_cached_model and self.model_path.exists():
            print(f"Loading cached ranker model from {self.model_path}...")
            self.model = lgb.Booster(model_file=str(self.model_path))
            return self.model

        print("Training Ranker from scratch...")

        # 1. Load Datasets
        train_df, val_df = feature_engineering.create_ranking_dataset(
            load_cached_data=True
        )

        # 2. Sort by customer_id (Required for LGBM query grouping)
        print("Sorting datasets by query group...")
        train_df = train_df.sort_values("customer_id").reset_index(drop=True)
        val_df = val_df.sort_values("customer_id").reset_index(drop=True)

        # 3. Define Features
        feature_cols = [
            "retrieval_score",
            "retrieval_rank",
            "age",
            "club_member_status",
            "fashion_news_frequency",
            "global_popularity",
            "product_type_no",
            "graphical_appearance_no",
            "colour_group_code",
            "perceived_colour_value_id",
            "department_no",
            "index_group_no",
            "section_no",
            "garment_group_no",
        ]

        X_train = train_df[feature_cols]
        y_train = train_df["label"]
        # Group is the count of samples per query (customer)
        group_train = train_df.groupby("customer_id").size().values

        X_val = val_df[feature_cols]
        y_val = val_df["label"]
        group_val = val_df.groupby("customer_id").size().values

        # 4. Create LGBM Datasets
        print("Creating LightGBM datasets...")
        train_set = lgb.Dataset(
            X_train, y_train, group=group_train, feature_name=feature_cols
        )
        val_set = lgb.Dataset(
            X_val,
            y_val,
            group=group_val,
            feature_name=feature_cols,
            reference=train_set,
        )

        # 5. Train
        print("Starting training...")
        params = config.LGBM_PARAMS.copy()

        callbacks = [
            lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=50),
        ]

        self.model = lgb.train(
            params,
            train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # 6. Save
        print(f"Saving model to {self.model_path}...")
        self.model.save_model(str(self.model_path))

        # Cleanup
        del train_df, val_df, X_train, X_val
        gc.collect()

        return self.model

    def predict(self):
        """
        Generates predictions for the test set and saves to submission.csv.
        """
        print("Starting Inference...")

        # 1. Load Test Customers
        test_customers = data_loader.load_test_customers(load_cached_data=True)
        all_test_cids = test_customers["customer_id"].values

        # 2. Load Full History (Train + Val) for Inference Context
        # We need the most recent data to build U vectors for test users
        print("Loading full transaction history for inference context...")
        train_trans = data_loader.load_transactions("train", load_cached_data=True)
        val_trans = data_loader.load_transactions("val", load_cached_data=True)
        full_history = pd.concat([train_trans, val_trans], ignore_index=True)

        # Filter to known items only
        full_history = full_history[
            full_history["article_id"].isin(self.retriever.art_id_to_idx)
        ].copy()

        # Map Article IDs to global indices
        full_history["item_idx"] = (
            full_history["article_id"]
            .map(self.retriever.art_id_to_idx)
            .astype(np.int32)
        )

        # Pre-calculate time decay
        max_date = full_history["t_dat"].max()
        full_history["days_diff"] = (max_date - full_history["t_dat"]).dt.days
        full_history["weight"] = np.exp(
            -full_history["days_diff"] / config.TIME_DECAY_DAYS
        )

        # 3. Batch Processing
        batch_size = 5000
        predictions = []
        n_test = len(all_test_cids)

        # Pre-format global popularity string for cold start fallback
        global_pop_ids = self.retriever.global_popularity[:12]
        # Ensure zfill for string format
        global_pop_str = " ".join([str(x).zfill(10) for x in global_pop_ids])

        print(f"Processing {n_test} customers in batches of {batch_size}...")

        for i in range(0, n_test, batch_size):
            batch_cids = all_test_cids[i : i + batch_size]

            # --- A. Construct User Vectors (U) ---
            # Filter history for this batch
            batch_hist = full_history[
                full_history["customer_id"].isin(batch_cids)
            ].copy()

            # Map batch customers to local index 0..batch_size-1
            cust_map = {cid: idx for idx, cid in enumerate(batch_cids)}
            batch_hist["user_idx"] = (
                batch_hist["customer_id"].map(cust_map).astype(np.int32)
            )

            # Build Sparse Matrices
            if len(batch_hist) > 0:
                row_idx = batch_hist["user_idx"].values
                col_idx = batch_hist["item_idx"].values
                weights = batch_hist["weight"].values

                u_weighted = sp.coo_matrix(
                    (weights, (row_idx, col_idx)),
                    shape=(len(batch_cids), self.retriever.n_items),
                ).tocsr()

                u_raw = sp.coo_matrix(
                    (np.ones(len(weights)), (row_idx, col_idx)),
                    shape=(len(batch_cids), self.retriever.n_items),
                ).tocsr()
            else:
                # Complete Cold Start for batch
                u_weighted = sp.csr_matrix((len(batch_cids), self.retriever.n_items))
                u_raw = sp.csr_matrix((len(batch_cids), self.retriever.n_items))

            # --- B. Retrieval Propagation ---
            # S = U_w @ T_seq + lambda * U_w @ T_vis + alpha * U_raw
            s_seq = u_weighted.dot(self.retriever.T_seq)
            s_vis = u_weighted.dot(self.retriever.T_vis)

            scores_sparse = (
                s_seq + (config.LAMBDA_VISUAL * s_vis) + (config.ALPHA_HISTORY * u_raw)
            )
            scores_dense = scores_sparse.toarray()

            # --- C. Top-K Extraction ---
            k = config.RETRIEVAL_TOP_K
            if scores_dense.shape[1] < k:
                k = scores_dense.shape[1]

            # Get indices of top K
            top_k_idx = np.argpartition(scores_dense, -k, axis=1)[:, -k:]

            # Sort them descending
            rows = np.arange(len(batch_cids))[:, None]
            top_k_scores = scores_dense[rows, top_k_idx]
            sort_order = np.argsort(top_k_scores, axis=1)[:, ::-1]

            sorted_indices = top_k_idx[rows, sort_order]
            sorted_scores = top_k_scores[rows, sort_order]

            # Identify Cold Users (Max score <= 0)
            # We will overwrite their predictions later
            max_scores = sorted_scores[:, 0]
            cold_user_mask = max_scores <= 0

            # --- D. Feature Construction ---
            # Flatten to create DataFrame
            cust_ids_repeated = np.repeat(batch_cids, k)
            flat_indices = sorted_indices.flatten()
            flat_scores = sorted_scores.flatten()
            flat_ranks = np.tile(np.arange(k), len(batch_cids))

            article_ids = self.retriever.art_idx_to_id[flat_indices]

            batch_df = pd.DataFrame(
                {
                    "customer_id": cust_ids_repeated,
                    "article_id": article_ids,
                    "retrieval_score": flat_scores,
                    "retrieval_rank": flat_ranks,
                }
            )

            # Merge Features
            batch_df = batch_df.merge(self.cust_features, on="customer_id", how="left")
            batch_df = batch_df.merge(self.art_features, on="article_id", how="left")
            batch_df["global_popularity"] = (
                batch_df["article_id"].map(self.pop_map).fillna(0.0)
            )

            # --- E. Scoring ---
            feature_cols = [
                "retrieval_score",
                "retrieval_rank",
                "age",
                "club_member_status",
                "fashion_news_frequency",
                "global_popularity",
                "product_type_no",
                "graphical_appearance_no",
                "colour_group_code",
                "perceived_colour_value_id",
                "department_no",
                "index_group_no",
                "section_no",
                "garment_group_no",
            ]

            # Ensure categoricals
            for c in ["club_member_status", "fashion_news_frequency"]:
                batch_df[c] = batch_df[c].astype("category")

            batch_df["pred_score"] = self.model.predict(batch_df[feature_cols])

            # --- F. Selection & Formatting ---
            # Sort by predicted score
            result_df = batch_df.sort_values(
                ["customer_id", "pred_score"], ascending=[True, False]
            )

            # Take Top 12
            top_12 = result_df.groupby("customer_id").head(12)

            # Format article IDs (zfill 10)
            top_12["art_str"] = top_12["article_id"].astype(str).str.zfill(10)

            # Join to string
            submission_series = top_12.groupby("customer_id")["art_str"].apply(
                lambda x: " ".join(x)
            )

            # Handle Cold Users (Overwrite with Global Popularity)
            cold_cids = batch_cids[cold_user_mask]
            for cid in cold_cids:
                submission_series[cid] = global_pop_str

            # Ensure all batch users are in submission_series (groupby might miss if filtered, though unlikely here)
            # If a user is missing from submission_series but was in batch_cids, fill them
            # (e.g. if retrieval failed completely, which we handled via cold mask, but good to be safe)
            missing_cids = set(batch_cids) - set(submission_series.index)
            for cid in missing_cids:
                submission_series[cid] = global_pop_str

            predictions.append(submission_series)

            if (i // batch_size) % 10 == 0:
                print(f"Processed batch {i // batch_size}...")
                gc.collect()

        # 4. Save Submission
        print("Concatenating predictions...")
        final_series = pd.concat(predictions)

        submission_df = final_series.reset_index()
        submission_df.columns = ["customer_id", "prediction"]

        print(f"Saving submission to {config.OUTPUT_FILE}...")
        submission_df.to_csv(config.OUTPUT_FILE, index=False)
        print("Done.")


def train_lgbm_ranker(load_cached_model=True):
    """Wrapper function to train the ranker."""
    ranker = Ranker()
    return ranker.train(load_cached_model=load_cached_model)


def predict_ranker(load_cached_data=True):
    """Wrapper function to generate submission."""
    # Ensure model is trained/loaded
    ranker = Ranker()
    if not ranker.model_path.exists():
        print("Model not found. Training now...")
        ranker.train(load_cached_model=False)
    else:
        ranker.train(load_cached_model=True)

    ranker.predict()
