import os
import gc
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from library.utils import reduce_mem_usage, set_seed


class RepurchaseGenerator:
    """
    Retrieves items previously purchased by the customer, ranked by recency.
    """

    def __init__(self, history_df):
        self.history_df = history_df

    def generate(self, target_customers, n_top=12):
        """
        Generates repurchase candidates for the target customers.
        """
        target_set = set(target_customers)

        # Filter transactions for target customers
        # We assume history_df is already temporally split
        df = self.history_df[self.history_df["customer_id"].isin(target_set)].copy()

        if df.empty:
            return pd.DataFrame(
                columns=["customer_id", "article_id", "repurchase_score"]
            )

        # Sort by date descending (most recent first)
        df = df.sort_values(["customer_id", "t_dat"], ascending=[True, False])

        # Deduplicate, keeping the most recent purchase of an article
        df = df.drop_duplicates(subset=["customer_id", "article_id"], keep="first")

        # Keep top K per customer
        df = df.groupby("customer_id").head(n_top)

        # Calculate a simple rank-based score (1.0 for most recent, 0.5 for 2nd, etc.)
        # Using 1 / rank
        df["rank"] = df.groupby("customer_id").cumcount() + 1
        df["repurchase_score"] = 1.0 / df["rank"]

        return df[["customer_id", "article_id", "repurchase_score"]]


class PopularityGenerator:
    """
    Retrieves globally popular items with a time-decay factor to prioritize trends.
    """

    def __init__(self, history_df):
        self.history_df = history_df

    def generate(self, n_top=12):
        """
        Generates a global list of popular items.
        Returns a DataFrame with [article_id, pop_score].
        """
        # Use only the last 4 weeks of available history for trend calculation
        min_week = self.history_df["week"].min()
        mask = self.history_df["week"] <= (min_week + 4)
        pop_df = self.history_df[mask].copy()

        if pop_df.empty:
            # Fallback if no recent history (unlikely)
            pop_df = self.history_df.copy()
            min_week = pop_df["week"].min()

        # Calculate time decay weight: 1 / (week_diff + 1)
        # week_diff is 0 for the most recent week in history
        pop_df["week_diff"] = pop_df["week"] - min_week
        pop_df["pop_weight"] = 1.0 / (pop_df["week_diff"] + 1)

        # Aggregate scores
        pop_scores = pop_df.groupby("article_id")["pop_weight"].sum().reset_index()
        pop_scores = pop_scores.sort_values("pop_weight", ascending=False).head(n_top)
        pop_scores = pop_scores.rename(columns={"pop_weight": "pop_score"})

        # Normalize score to 0-1 range for consistency
        if not pop_scores.empty:
            pop_scores["pop_score"] = (
                pop_scores["pop_score"] / pop_scores["pop_score"].max()
            )

        return pop_scores[["article_id", "pop_score"]]


class ItemCFGenerator:
    """
    Generates candidates using Item-Item Collaborative Filtering.
    Uses sparse matrices with row-wise normalization and temporal decay.
    """

    def __init__(self, history_df):
        self.history_df = history_df

    def generate(self, target_customers, n_top=12, batch_size=5000):
        """
        Generates CF candidates.
        """
        # Configuration
        # Cite solution_lesson_node_00007: Reduced history window from 10 to 5 to avoid noise/drift.
        HISTORY_WINDOW = 5  # Weeks of history to use for building the matrix
        DECAY_RATE = 0.5  # Decay rate for user history weighting
        SIMILARITY_PRUNE = 0.05  # Threshold to prune low similarity values

        # 1. Prepare Data
        # Filter history to reduce noise and memory usage
        min_week = self.history_df["week"].min()
        window_df = self.history_df[
            self.history_df["week"] <= min_week + HISTORY_WINDOW
        ].copy()

        if window_df.empty:
            return pd.DataFrame(columns=["customer_id", "article_id", "cf_score"])

        # Create mappings
        users = window_df["customer_id"].unique()
        items = window_df["article_id"].unique()

        # Ensure target customers are in the user map (even if they have no history in window)
        target_set = set(target_customers)
        all_users = np.unique(np.concatenate([users, list(target_set)]))

        user_map = {u: i for i, u in enumerate(all_users)}
        item_map = {i: idx for idx, i in enumerate(items)}
        reverse_item_map = {idx: i for i, idx in item_map.items()}

        n_users = len(all_users)
        n_items = len(items)

        # 2. Build Interaction Matrix R (Users x Items)
        # We use binary interactions for the similarity learning phase
        train_users = window_df["customer_id"].map(user_map).values
        train_items = window_df["article_id"].map(item_map).values

        # Create CSR matrix
        R = sp.coo_matrix(
            (np.ones(len(window_df), dtype=np.float32), (train_users, train_items)),
            shape=(n_users, n_items),
        ).tocsr()

        # Row-wise L2 Normalization (User Normalization)
        # This reduces the impact of power users who buy everything
        R_norm = normalize(R, norm="l2", axis=1)

        # 3. Compute Item-Item Similarity S = R_norm.T @ R_norm
        # Result is (Items x Items)
        S = R_norm.T @ R_norm

        # Prune small values to keep density manageable
        S.data[S.data < SIMILARITY_PRUNE] = 0
        S.eliminate_zeros()

        # 4. Build Query Matrix Q for Target Customers
        # Filter history for target users only
        target_history = window_df[window_df["customer_id"].isin(target_set)].copy()

        if target_history.empty:
            return pd.DataFrame(columns=["customer_id", "article_id", "cf_score"])

        # Apply Temporal Decay to User History
        target_history["week_diff"] = target_history["week"] - min_week
        target_history["weight"] = 1.0 / (
            1.0 + DECAY_RATE * target_history["week_diff"]
        )

        q_users = target_history["customer_id"].map(user_map).values
        q_items = target_history["article_id"].map(item_map).values
        q_weights = target_history["weight"].values

        # Q is (N_Users x N_Items), but we only care about rows for target_customers
        Q = sp.coo_matrix(
            (q_weights, (q_users, q_items)), shape=(n_users, n_items)
        ).tocsr()

        # Extract indices for target customers in the correct order
        target_indices = [user_map[u] for u in target_customers]
        Q_target = Q[target_indices]

        # 5. Generate Predictions in Batches
        results = []

        # Process in batches to manage memory during dense expansion
        for start in range(0, Q_target.shape[0], batch_size):
            end = min(start + batch_size, Q_target.shape[0])
            Q_batch = Q_target[start:end]

            # Compute Scores: (Batch x Items)
            # This represents the weighted sum of similarities to items in user history
            Scores = Q_batch @ S

            # Convert to dense to find top K
            Scores_dense = Scores.toarray()

            # Skip if batch is empty (all zeros)
            if Scores_dense.sum() == 0:
                continue

            # Use argpartition to efficiently find top K indices
            # We want top n_top items
            # Handle case where n_items < n_top
            k = min(n_top, n_items)
            if k == 0:
                continue

            top_k_idx = np.argpartition(Scores_dense, -k, axis=1)[:, -k:]

            # Extract values
            rows = np.arange(Scores_dense.shape[0])[:, None]
            top_k_scores = Scores_dense[rows, top_k_idx]

            # Prepare DataFrame
            batch_cust_ids = target_customers[start:end]

            # Flatten for DataFrame construction
            u_ids = np.repeat(batch_cust_ids, k)
            i_indices = top_k_idx.flatten()
            s_vals = top_k_scores.flatten()

            # Map back to article_id
            i_ids = [reverse_item_map[idx] for idx in i_indices]

            batch_df = pd.DataFrame(
                {"customer_id": u_ids, "article_id": i_ids, "cf_score": s_vals}
            )

            # Filter out zero scores (irrelevant items)
            batch_df = batch_df[batch_df["cf_score"] > 0]

            results.append(batch_df)

        if not results:
            return pd.DataFrame(columns=["customer_id", "article_id", "cf_score"])

        return pd.concat(results, ignore_index=True)


class CandidateEngine:
    def __init__(self, transactions_df, cache_dir="./working/idea_2"):
        self.transactions = transactions_df
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        set_seed(42)

    def generate_candidates(self, target_customers, target_week, load_cached_data=True):
        """
        Main method to generate candidates for a specific target week.
        Combines Repurchase, Popularity, and CF strategies.
        """
        cache_file = os.path.join(
            self.cache_dir, f"candidates_week_{target_week}.parquet"
        )

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading candidates from cache: {cache_file}")
            return pd.read_parquet(cache_file)

        print(f"Generating candidates for week {target_week}...")

        # 1. Split Data
        # History is strictly before the target week
        history_df = self.transactions[self.transactions["week"] > target_week].copy()

        if history_df.empty:
            raise ValueError(f"No history available for week > {target_week}")

        # 2. Initialize Generators
        repurchase_gen = RepurchaseGenerator(history_df)
        pop_gen = PopularityGenerator(history_df)
        cf_gen = ItemCFGenerator(history_df)

        # 3. Generate Candidates
        print("Running Repurchase Generator...")
        repurchase_cands = repurchase_gen.generate(target_customers, n_top=12)

        print("Running Popularity Generator...")
        pop_items = pop_gen.generate(n_top=12)

        print("Running ItemCF Generator...")
        # CF can be slow, so we process it carefully
        cf_cands = cf_gen.generate(target_customers, n_top=12)

        # 4. Merge Candidates
        print("Merging candidates...")

        # Start with Repurchase
        candidates = repurchase_cands.copy()

        # Merge CF (Outer Join)
        candidates = candidates.merge(
            cf_cands, on=["customer_id", "article_id"], how="outer"
        )

        # Add Popularity Candidates
        # Strategy: We want to ensure every user has the popularity candidates available as a fallback/signal.
        # We create a Cartesian product of target_customers x pop_items
        # To save memory, we construct this efficiently

        # Create a DataFrame of all target customers
        cust_df = pd.DataFrame({"customer_id": target_customers})

        # Cross join with pop items
        # pandas cross join
        pop_cands_full = cust_df.merge(pop_items, how="cross")

        # Merge into main candidates
        candidates = candidates.merge(
            pop_cands_full, on=["customer_id", "article_id"], how="outer"
        )

        # 5. Cleanup and Formatting
        # Fill NaNs in score columns with 0
        score_cols = ["repurchase_score", "cf_score", "pop_score"]
        for col in score_cols:
            if col in candidates.columns:
                candidates[col] = candidates[col].fillna(0.0)
            else:
                candidates[col] = 0.0

        # Optimize types
        candidates = reduce_mem_usage(candidates, verbose=False)

        # Save to cache
        print(f"Saving {len(candidates)} candidates to {cache_file}")
        candidates.to_parquet(cache_file, index=False)

        return candidates
