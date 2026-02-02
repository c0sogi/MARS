import pandas as pd
import numpy as np
import scipy.sparse as sp
import gc
import os
from library.config import Config
from library.embedder import LatentEmbedder


class CandidateEngine:
    """
    Orchestrates the Multi-Source Retrieval Strategy:
    1. Linear-Decay Co-occurrence (Structure)
    2. Latent Behavioral Embeddings (Semantics)
    3. Repurchase History (Habit)
    4. Recent Popularity (Trend)
    """

    def __init__(self):
        self.embedder = LatentEmbedder()
        self.popular_items = []

        # Co-occurrence artifacts
        self.user_map = {}
        self.item_map = {}
        self.reverse_item_map = {}
        self.cooc_matrix = None

    def fit(self, transactions: pd.DataFrame):
        """
        Prepares all retrieval models (Embeddings, Co-occurrence Matrix, Popularity).
        """
        print("Fitting Candidate Engine...")

        # 1. Fit Embedder (Source B)
        self.embedder.fit(transactions)

        # 2. Fit Co-occurrence (Source A)
        self._fit_cooccurrence(transactions)

        # 3. Fit Popularity (Source D)
        self._fit_popularity(transactions)

        print("Candidate Engine fitting complete.")

    def _fit_popularity(self, transactions):
        """
        Identifies top items from the most recent window.
        """
        max_date = transactions["t_dat"].max()
        start_date = max_date - pd.Timedelta(days=Config.POPULARITY_WINDOW_DAYS)
        pop_df = transactions[transactions["t_dat"] > start_date]

        # Get top K items
        self.popular_items = (
            pop_df["article_id"]
            .value_counts()
            .head(Config.TOP_K_POPULARITY)
            .index.tolist()
        )
        print(f"Identified {len(self.popular_items)} global popularity candidates.")

    def _fit_cooccurrence(self, transactions):
        """
        Builds the Item-Item Co-occurrence Matrix with Linear Time Decay.
        """
        print("Building Co-occurrence Matrix...")

        # Filter to configured window
        max_date = transactions["t_dat"].max()
        start_date = max_date - pd.Timedelta(weeks=Config.COOC_WINDOW_WEEKS)
        df = transactions[transactions["t_dat"] > start_date].copy()

        # Calculate Linear Time Decay Weight: w = 1 / (days_diff + 1)
        df["days_diff"] = (max_date - df["t_dat"]).dt.days
        df["weight"] = 1.0 / (df["days_diff"] + 1.0)

        # Create Mappings
        # We map all items/users found in this window
        self.user_map = {u: i for i, u in enumerate(df["customer_id"].unique())}
        self.item_map = {i: k for k, i in enumerate(df["article_id"].unique())}
        self.reverse_item_map = {v: k for k, v in self.item_map.items()}

        # Construct User-Item Sparse Matrix
        row = df["customer_id"].map(self.user_map).values
        col = df["article_id"].map(self.item_map).values
        data = df["weight"].values

        n_users = len(self.user_map)
        n_items = len(self.item_map)

        # Shape: Users x Items
        user_item_mat = sp.csr_matrix((data, (row, col)), shape=(n_users, n_items))

        # Compute Item-Item Matrix: S = U^T * U
        # This gives the dot product of user vectors for each pair of items
        self.cooc_matrix = user_item_mat.transpose().dot(user_item_mat)

        # Zero out diagonal (self-similarity)
        self.cooc_matrix.setdiag(0)

        print(f"Co-occurrence Matrix shape: {self.cooc_matrix.shape} (Sparse)")

        # Clean up
        del df, user_item_mat
        gc.collect()

    def generate_candidates(
        self,
        users: pd.DataFrame,
        transactions: pd.DataFrame,
        load_cached_data: bool = True,
    ):
        """
        Generates and merges candidates for the provided users.
        Returns a DataFrame with columns: [customer_id, article_id, cooc_score, embed_score, repur_count, pop_flag]
        """
        # Define cache key
        params = {
            "n_users": len(users),
            "max_date": str(transactions["t_dat"].max().date()),
            "top_k_cooc": Config.TOP_K_COOC,
            "top_k_embed": Config.TOP_K_EMBED,
            "top_k_repur": Config.TOP_K_REPURCHASE,
        }
        cache_path = Config.get_cache_path("candidates_combined.parquet", params)

        if load_cached_data and cache_path.exists():
            print(f"Loading cached candidates from {cache_path}")
            return pd.read_parquet(cache_path)

        print("Generating candidates from scratch...")
        customer_ids = users["customer_id"].unique()

        # 1. Source A: Co-occurrence
        print("Retrieving Source A (Co-occurrence)...")
        cands_cooc = self._retrieve_cooccurrence(customer_ids, transactions)

        # 2. Source B: Embeddings
        print("Retrieving Source B (Embeddings)...")
        cands_embed = self._retrieve_embeddings(customer_ids, transactions)

        # 3. Source C: Repurchase
        print("Retrieving Source C (Repurchase)...")
        cands_repur = self._retrieve_repurchase(customer_ids, transactions)

        # 4. Source D: Popularity
        # (Handled during merge to save memory)

        # --- Merging Strategy ---
        print("Merging candidates from all sources...")

        # Standardize DataFrames
        cands_cooc = cands_cooc.rename(columns={"score": "cooc_score"})
        cands_embed = cands_embed.rename(columns={"score": "embed_score"})
        cands_repur = cands_repur.rename(columns={"count": "repur_count"})

        # Add source flags/defaults
        # We will concatenate and then aggregate

        # Prepare Popularity Candidates (Source D)
        # Cross join unique users with popular items
        # Efficient implementation: Repeat popular items for each user chunk or use indexing
        print("Appending Popularity candidates...")
        u_df = pd.DataFrame({"customer_id": customer_ids})
        u_df["key"] = 1
        p_df = pd.DataFrame({"article_id": self.popular_items})
        p_df["key"] = 1
        p_df["pop_flag"] = 1

        # Note: Full cross join might be heavy if N_users is large.
        # But for 12 items * 1M users = 12M rows, it fits in memory.
        cands_pop = pd.merge(u_df, p_df, on="key").drop("key", axis=1)

        # Concatenate all sources
        # We align columns. Missing columns become NaN.
        all_cands = pd.concat(
            [cands_cooc, cands_embed, cands_repur, cands_pop], ignore_index=True
        )

        # Aggregate by (customer_id, article_id)
        # We take the max score for each feature (since a source produces 1 or 0/NaN for that feature)
        print("Aggregating and deduplicating...")
        final_df = all_cands.groupby(["customer_id", "article_id"], as_index=False).agg(
            {
                "cooc_score": "max",
                "embed_score": "max",
                "repur_count": "max",
                "pop_flag": "max",
            }
        )

        # Fill NaNs with 0
        cols_to_fill = ["cooc_score", "embed_score", "repur_count", "pop_flag"]
        final_df[cols_to_fill] = final_df[cols_to_fill].fillna(0)

        print(
            f"Generated {len(final_df)} unique candidates for {len(customer_ids)} users."
        )

        # Cache result
        final_df.to_parquet(cache_path, index=False)

        return final_df

    def _retrieve_cooccurrence(self, user_ids, transactions):
        """
        Retrieves candidates using the pre-computed Co-occurrence Matrix.
        Logic: User History Vector (Weighted) * Item-Item Matrix = Recommendation Scores
        """
        # Filter transactions to the relevant window and users
        max_date = transactions["t_dat"].max()
        start_date = max_date - pd.Timedelta(weeks=Config.COOC_WINDOW_WEEKS)

        mask_users = transactions["customer_id"].isin(user_ids)
        mask_date = transactions["t_dat"] > start_date
        sub_df = transactions[mask_users & mask_date].copy()

        if sub_df.empty:
            return pd.DataFrame(columns=["customer_id", "article_id", "score"])

        # Compute weights for history
        sub_df["days_diff"] = (max_date - sub_df["t_dat"]).dt.days
        sub_df["weight"] = 1.0 / (sub_df["days_diff"] + 1.0)

        # Map items to matrix indices
        sub_df["item_idx"] = sub_df["article_id"].map(self.item_map)
        sub_df = sub_df.dropna(subset=["item_idx"])
        sub_df["item_idx"] = sub_df["item_idx"].astype(int)

        # Map users to local indices (0 to N_requested-1)
        # This ensures the output matrix rows align with user_ids list
        local_user_map = {u: i for i, u in enumerate(user_ids)}

        # Only process users with history
        sub_df["user_idx_local"] = sub_df["customer_id"].map(local_user_map)
        sub_df = sub_df.dropna(subset=["user_idx_local"])
        sub_df["user_idx_local"] = sub_df["user_idx_local"].astype(int)

        # Create Sparse User History Matrix (Local Users x Global Items)
        rows = sub_df["user_idx_local"].values
        cols = sub_df["item_idx"].values
        data = sub_df["weight"].values

        n_local_users = len(user_ids)
        user_hist_mat = sp.csr_matrix(
            (data, (rows, cols)), shape=(n_local_users, len(self.item_map))
        )

        # Matrix Multiplication
        # (Local Users x Items) * (Items x Items) -> (Local Users x Items)
        scores_mat = user_hist_mat.dot(self.cooc_matrix)

        # Extract Top K per user
        result_records = []

        # Iterate over rows to get top K
        # Since we need to map back indices to article_ids, we do this manually
        for i in range(n_local_users):
            row = scores_mat.getrow(i)
            if row.nnz == 0:
                continue

            ind = row.indices
            val = row.data

            # Get top K
            if len(val) > Config.TOP_K_COOC:
                # argpartition is efficient O(N)
                top_k_idx = np.argpartition(val, -Config.TOP_K_COOC)[
                    -Config.TOP_K_COOC :
                ]
                best_ind = ind[top_k_idx]
                best_val = val[top_k_idx]
            else:
                best_ind = ind
                best_val = val

            # Map back
            u_id = user_ids[i]
            for idx, score in zip(best_ind, best_val):
                art_id = self.reverse_item_map[idx]
                result_records.append((u_id, art_id, score))

        return pd.DataFrame(
            result_records, columns=["customer_id", "article_id", "score"]
        )

    def _retrieve_embeddings(self, user_ids, transactions):
        """
        Retrieves candidates using Latent Behavioral Embeddings (Item2Vec).
        Logic: Cosine Similarity between User Vector (Mean of History) and Item Vectors.
        """
        # 1. Compute User Embeddings
        # Filter history for these users
        max_date = transactions["t_dat"].max()
        start_date = max_date - pd.Timedelta(weeks=Config.EMBED_WINDOW_WEEKS)

        mask_users = transactions["customer_id"].isin(user_ids)
        mask_date = transactions["t_dat"] > start_date
        sub_df = transactions[mask_users & mask_date].copy()

        if sub_df.empty or self.embedder.embeddings is None:
            return pd.DataFrame(columns=["customer_id", "article_id", "score"])

        user_embeds_map = self.embedder.get_user_embeddings(sub_df)

        # 2. Prepare Matrices for Similarity Search
        item_matrix = self.embedder.embeddings  # (V, D)

        # Normalize Item Matrix
        item_norms = np.linalg.norm(item_matrix, axis=1, keepdims=True)
        item_norms[item_norms == 0] = 1e-10
        item_matrix_norm = item_matrix / item_norms

        # Filter users who actually have embeddings
        valid_users = [u for u in user_ids if u in user_embeds_map]
        if not valid_users:
            return pd.DataFrame(columns=["customer_id", "article_id", "score"])

        user_matrix = np.array([user_embeds_map[u] for u in valid_users])  # (U, D)

        # Normalize User Matrix
        user_norms = np.linalg.norm(user_matrix, axis=1, keepdims=True)
        user_norms[user_norms == 0] = 1e-10
        user_matrix_norm = user_matrix / user_norms

        # 3. Batched Similarity Search
        # (U, D) @ (D, V) -> (U, V)
        batch_size = 1000
        results = []
        idx_to_article = self.embedder.inverse_vocab

        for i in range(0, len(valid_users), batch_size):
            u_batch = user_matrix_norm[i : i + batch_size]  # (B, D)

            # Compute Cosine Similarity
            scores = np.dot(u_batch, item_matrix_norm.T)  # (B, V)

            # Extract Top K
            k = Config.TOP_K_EMBED
            # argpartition on axis 1 to find top k indices
            top_k_indices = np.argpartition(scores, -k, axis=1)[:, -k:]

            # Extract corresponding scores
            rows = np.arange(scores.shape[0])[:, None]
            top_k_scores = scores[rows, top_k_indices]

            # Map back to IDs
            batch_users = valid_users[i : i + batch_size]
            for r in range(len(batch_users)):
                u_id = batch_users[r]
                indices = top_k_indices[r]
                vals = top_k_scores[r]

                for idx, val in zip(indices, vals):
                    if idx in idx_to_article:
                        art_id = idx_to_article[idx]
                        results.append((u_id, art_id, val))

        return pd.DataFrame(results, columns=["customer_id", "article_id", "score"])

    def _retrieve_repurchase(self, user_ids, transactions):
        """
        Retrieves items the user has purchased before, ranked by frequency.
        """
        mask = transactions["customer_id"].isin(user_ids)
        df = transactions[mask].copy()

        if df.empty:
            return pd.DataFrame(columns=["customer_id", "article_id", "count"])

        # Count frequency
        counts = (
            df.groupby(["customer_id", "article_id"]).size().reset_index(name="count")
        )

        # Sort and take top K per user
        counts = counts.sort_values(["customer_id", "count"], ascending=[True, False])
        counts = counts.groupby("customer_id").head(Config.TOP_K_REPURCHASE)

        return counts
