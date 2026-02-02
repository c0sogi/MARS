import pandas as pd
import numpy as np
import scipy.sparse as sp
import gc
import os
from tqdm import tqdm
from library.config import Config
from library.utils import Timer, reduce_mem_usage


class HybridRetrieval:
    """
    Implements Stage 1: Hybrid Candidate Retrieval.
    Generates candidate items for users based on:
    1. Linear-Decay Item-Item Co-occurrence (Collaborative Filtering)
    2. Repurchase History (Habitual Behavior)
    3. Global Popularity (Cold-Start Fallback)
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.weeks = Config.RETRIEVAL_HISTORY_WEEKS
        self.top_k_cooc = Config.TOP_K_COOC
        self.top_k_repurchase = Config.TOP_K_REPURCHASE
        self.top_k_pop = Config.TOP_K_POPULARITY

    def generate_candidates(
        self, train_df, target_customer_ids, mode="train", load_cached_data=True
    ):
        """
        Main method to generate candidates for a list of target customers.

        Args:
            train_df (pd.DataFrame): Transaction history with mapped integer IDs.
            target_customer_ids (array-like): List of customer_id_idx to generate candidates for.
            mode (str): 'train' or 'test', used for cache file naming.
            load_cached_data (bool): Whether to attempt loading from parquet cache.

        Returns:
            pd.DataFrame: A dataframe containing candidate pairs and their source scores.
                          Columns: [customer_id_idx, article_id_idx, cooc_score, repurchase_score, pop_score]
        """
        # 1. Cache Check
        cache_file = self.working_dir / f"candidates_{mode}.parquet"
        if load_cached_data and cache_file.exists():
            print(f"[HybridRetrieval] Loading cached candidates from {cache_file}")
            return pd.read_parquet(cache_file)

        print(
            f"[HybridRetrieval] Generating candidates for {mode} (Target Users: {len(target_customer_ids)})..."
        )

        # 2. Preprocess History (Filter by Time Window & Calculate Weights)
        with Timer("History Preprocessing"):
            # Filter to recent weeks to capture current trends
            max_date = train_df["t_dat"].max()
            start_date = max_date - pd.Timedelta(weeks=self.weeks)
            history_df = train_df[train_df["t_dat"] > start_date].copy()

            if history_df.empty:
                print("[HybridRetrieval] Warning: History is empty after filtering!")
                return pd.DataFrame(
                    columns=[
                        "customer_id_idx",
                        "article_id_idx",
                        "cooc_score",
                        "repurchase_score",
                        "pop_score",
                    ]
                )

            # Calculate Linear Decay Weights
            # Formula: w = 1 / (days_diff + 1)
            # days_diff = 0 for the most recent day in the dataset
            history_df["days_diff"] = (max_date - history_df["t_dat"]).dt.days
            history_df["weight"] = 1.0 / (history_df["days_diff"] + 1.0)

            # Keep only necessary columns for matrix operations
            history_df = history_df[["customer_id_idx", "article_id_idx", "weight"]]

        # 3. Generate Candidates from Sources

        # Source A: Co-occurrence
        # Only relevant for users who actually have history in the window
        users_with_history = history_df["customer_id_idx"].unique()
        target_set = set(target_customer_ids)
        # Identify which target users have history (and thus can get CF recommendations)
        active_targets = [u for u in users_with_history if u in target_set]

        candidates_cooc = pd.DataFrame()
        if len(active_targets) > 0:
            candidates_cooc = self._compute_cooccurrence(history_df, active_targets)

        # Source B: Repurchase
        candidates_rep = pd.DataFrame()
        if len(active_targets) > 0:
            candidates_rep = self._compute_repurchase(history_df, active_targets)

        # Source C: Popularity
        # Relevant for ALL target users (provides coverage for cold-start)
        candidates_pop = self._compute_popularity(history_df, target_customer_ids)

        # 4. Merge Candidates
        with Timer("Merging Candidates"):
            # We want an outer join of all candidates to form the union

            # Start with Co-occurrence and Repurchase
            if not candidates_cooc.empty and not candidates_rep.empty:
                candidates = pd.merge(
                    candidates_cooc,
                    candidates_rep,
                    on=["customer_id_idx", "article_id_idx"],
                    how="outer",
                )
            elif not candidates_cooc.empty:
                candidates = candidates_cooc
                candidates["repurchase_score"] = 0.0
            elif not candidates_rep.empty:
                candidates = candidates_rep
                candidates["cooc_score"] = 0.0
            else:
                candidates = pd.DataFrame(
                    columns=[
                        "customer_id_idx",
                        "article_id_idx",
                        "cooc_score",
                        "repurchase_score",
                    ]
                )

            # Fill NaNs from the first merge
            if "cooc_score" in candidates.columns:
                candidates["cooc_score"] = candidates["cooc_score"].fillna(0.0)
            if "repurchase_score" in candidates.columns:
                candidates["repurchase_score"] = candidates["repurchase_score"].fillna(
                    0.0
                )

            # Merge Popularity
            # Popularity candidates are generated for everyone, ensuring every user has at least TOP_K_POP items
            candidates = pd.merge(
                candidates,
                candidates_pop,
                on=["customer_id_idx", "article_id_idx"],
                how="outer",
            )

            # Fill NaNs for all score columns
            candidates["pop_score"] = candidates["pop_score"].fillna(0.0)
            candidates["cooc_score"] = candidates["cooc_score"].fillna(0.0)
            candidates["repurchase_score"] = candidates["repurchase_score"].fillna(0.0)

        # 5. Save and Return
        with Timer("Saving Candidates"):
            candidates = reduce_mem_usage(candidates)
            # Ensure directory exists
            os.makedirs(self.working_dir, exist_ok=True)
            candidates.to_parquet(cache_file, index=False)

        return candidates

    def _compute_cooccurrence(self, history_df, target_users):
        """
        Generates candidates based on Item-Item Co-occurrence matrix.
        Score = User_History_Vector * Item_Item_Similarity_Matrix
        """
        with Timer("Source: Co-occurrence"):
            # 1. Build User-Item Matrix R (Sparse)
            # Rows: Users, Cols: Items
            # Determine dimensions based on max indices found in history
            # Note: We use max() + 1 to accommodate 0-based indexing
            n_users = history_df["customer_id_idx"].max() + 1
            n_items = history_df["article_id_idx"].max() + 1

            # Construct R: Users x Items
            # Values are the linear decay weights
            R = sp.csr_matrix(
                (
                    history_df["weight"].values,
                    (
                        history_df["customer_id_idx"].values,
                        history_df["article_id_idx"].values,
                    ),
                ),
                shape=(n_users, n_items),
            )

            # 2. Compute Item-Item Similarity S = R.T @ R
            # This computes the unnormalized co-occurrence strength between items
            # S[i, j] = sum(weight_u_i * weight_u_j) for all users u
            S = R.T.dot(R)

            # 3. Predict for Target Users
            # We process users in batches to manage memory usage of the dense score matrix
            batch_size = 1000
            result_list = []

            target_users = np.array(target_users)

            for i in range(0, len(target_users), batch_size):
                batch_uids = target_users[i : i + batch_size]

                # Extract history vectors for this batch of users
                Q_batch = R[batch_uids, :]

                # Compute Scores: (Batch, Items) = (Batch, Items) @ (Items, Items)
                Scores = Q_batch.dot(S)

                # Convert to dense to perform top-k selection efficiently
                if sp.issparse(Scores):
                    Scores = Scores.toarray()

                # Select Top-K items per user
                k = min(self.top_k_cooc, Scores.shape[1])

                # argpartition moves the top k elements to the end of the array (unsorted)
                # We take the indices of these top k elements
                top_k_idx = np.argpartition(Scores, -k, axis=1)[:, -k:]

                # Retrieve the corresponding scores
                rows = np.arange(Scores.shape[0])[:, None]
                top_k_scores = Scores[rows, top_k_idx]

                # Flatten the arrays to create a dataframe
                u_ids = np.repeat(batch_uids, k)
                i_ids = top_k_idx.flatten()
                s_vals = top_k_scores.flatten()

                # Filter out zero scores (items with no similarity to user history)
                mask = s_vals > 0
                if mask.sum() > 0:
                    batch_df = pd.DataFrame(
                        {
                            "customer_id_idx": u_ids[mask],
                            "article_id_idx": i_ids[mask],
                            "cooc_score": s_vals[mask],
                        }
                    )
                    result_list.append(batch_df)

            if not result_list:
                return pd.DataFrame(
                    columns=["customer_id_idx", "article_id_idx", "cooc_score"]
                )

            return pd.concat(result_list, ignore_index=True)

    def _compute_repurchase(self, history_df, target_users):
        """
        Generates candidates based on users' own purchase history (Repurchase).
        Score = Sum of weights (frequency/recency).
        """
        with Timer("Source: Repurchase"):
            # Filter history to only target users
            target_set = set(target_users)
            relevant_history = history_df[
                history_df["customer_id_idx"].isin(target_set)
            ]

            if relevant_history.empty:
                return pd.DataFrame()

            # Aggregate weights by User and Item
            repurchase = (
                relevant_history.groupby(["customer_id_idx", "article_id_idx"])[
                    "weight"
                ]
                .sum()
                .reset_index()
            )
            repurchase.rename(columns={"weight": "repurchase_score"}, inplace=True)

            # Keep Top-K per user
            # Sort descending by score
            repurchase = repurchase.sort_values(
                ["customer_id_idx", "repurchase_score"], ascending=[True, False]
            )

            # Select top K
            repurchase = repurchase.groupby("customer_id_idx").head(
                self.top_k_repurchase
            )

            return repurchase

    def _compute_popularity(self, history_df, target_users):
        """
        Generates candidates based on Global Popularity in the window.
        Score = Sum of weights across all users.
        """
        with Timer("Source: Popularity"):
            # 1. Compute Global Popularity
            pop_scores = (
                history_df.groupby("article_id_idx")["weight"].sum().reset_index()
            )
            pop_scores.rename(columns={"weight": "pop_score"}, inplace=True)

            # 2. Get Top-K Global Items
            top_pop = pop_scores.nlargest(self.top_k_pop, "pop_score")

            # 3. Broadcast to all target users
            # Create a dataframe of all target users
            users_df = pd.DataFrame({"customer_id_idx": target_users})

            # Perform a cross join (Cartesian product)
            # Assign a temporary key for merging
            users_df["key"] = 1
            top_pop_copy = top_pop.copy()
            top_pop_copy["key"] = 1

            candidates_pop = pd.merge(users_df, top_pop_copy, on="key").drop(
                "key", axis=1
            )

            return candidates_pop
