import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import gc
from pathlib import Path
from tqdm import tqdm

from library.config import (
    WORKING_DIR,
    SUBMISSION_DIR,
    HISTORY_WEEKS,
    DECAY_RATE,
    TOP_K,
    RANDOM_STATE,
    SUBMISSION_PATH,
)
from library.data_factory import load_and_filter_data, get_target_customers, IdEncoder
from library.metrics import calculate_map12

# Ensure reproducibility
np.random.seed(RANDOM_STATE)


class TimeWeightedCooccurrence:
    def __init__(self, decay_rate=DECAY_RATE, top_k=TOP_K):
        self.decay_rate = decay_rate
        self.top_k = top_k
        self.similarity_matrix = None
        self.global_popularity = []
        self.encoder = IdEncoder()

        # Paths for caching
        self.cache_dir = WORKING_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sim_matrix_path = self.cache_dir / "similarity_matrix.npz"
        self.pop_path = self.cache_dir / "global_popularity.npy"

    def fit(self, train_df, load_cached_data=True):
        """
        Fits the model by constructing the item-item similarity matrix.
        Uses caching to avoid re-computation.
        """
        # 1. Fit Encoder
        # We always ensure the encoder is fitted on the current data or loaded from its own cache
        # The data_factory handles the encoder caching, but we need to call fit.
        print("Fitting ID Encoder...")
        self.encoder.fit(
            train_df["customer_id"],
            train_df["article_id"],
            load_cached_data=load_cached_data,
        )

        # 2. Check for Model Cache
        if (
            load_cached_data
            and self.sim_matrix_path.exists()
            and self.pop_path.exists()
        ):
            print("Loading model artifacts from cache...")
            self.similarity_matrix = sp.load_npz(self.sim_matrix_path)
            self.global_popularity = np.load(self.pop_path, allow_pickle=True).tolist()
            return

        print("Computing model from scratch...")

        # 3. Calculate Global Popularity (Fallback)
        # Use last 7 days of training data for "Trending Now"
        print("Calculating global popularity...")
        train_df["t_dat"] = pd.to_datetime(train_df["t_dat"])
        max_date = train_df["t_dat"].max()
        last_week_start = max_date - pd.Timedelta(days=7)

        pop_df = train_df[train_df["t_dat"] > last_week_start]
        # Count frequency
        pop_counts = pop_df["article_id"].value_counts().head(self.top_k)
        self.global_popularity = pop_counts.index.tolist()

        # 4. Prepare Data for Matrix
        print("Preparing sparse matrix data...")
        # Map IDs to integers
        user_idxs = self.encoder.transform_customers(train_df["customer_id"])
        item_idxs = self.encoder.transform_articles(train_df["article_id"])

        # Calculate Time Weights
        # w = 1 / (diff_days + 1) ^ decay
        diff_days = (max_date - train_df["t_dat"]).dt.days.values
        weights = 1.0 / np.power(diff_days + 1.0, self.decay_rate)

        # 5. Build User-Item Matrix (M)
        # Shape: (Num Users, Num Items)
        # We use max index + 1 to define shape
        n_users = len(self.encoder.customer_to_idx)
        n_items = len(self.encoder.article_to_idx)

        # Filter out -1 (unknowns, though shouldn't happen in fit with same data)
        mask = (user_idxs != -1) & (item_idxs != -1)
        M = sp.csr_matrix(
            (weights[mask], (user_idxs[mask], item_idxs[mask])),
            shape=(n_users, n_items),
        )

        # 6. Compute Item-Item Similarity (S = M.T @ M)
        print("Computing item-item similarity matrix (M.T @ M)...")
        # This results in a (n_items, n_items) matrix
        S = M.T.dot(M)

        # Cosine Normalization
        print("Applying Cosine Normalization...")
        # Calculate sqrt of diagonal elements (L2 norms of columns of M)
        diag = S.diagonal()
        norms = np.sqrt(diag)
        # Handle zero norms to avoid division by zero
        norms[norms == 0] = 1.0

        # Create diagonal inverse matrix
        D_inv = sp.diags(1.0 / norms)

        # S_norm = D_inv @ S @ D_inv
        # This effectively computes S_ij / (norm_i * norm_j)
        S = D_inv.dot(S).dot(D_inv)

        # Zero out diagonal (we want to recommend *other* items associated, not self)
        # Although re-purchase is valid, co-occurrence usually implies A->B
        S.setdiag(0)

        self.similarity_matrix = S

        # 7. Save to Cache
        print("Saving model artifacts...")
        sp.save_npz(self.sim_matrix_path, S)
        np.save(self.pop_path, np.array(self.global_popularity))

        # Cleanup
        del M, user_idxs, item_idxs, weights
        gc.collect()

    def predict(self, target_customer_ids, history_df, batch_size=1000):
        """
        Generates predictions for target_customer_ids using their history in history_df.
        """
        print(f"Generating predictions for {len(target_customer_ids)} customers...")

        # Ensure history_df is encoded
        # We create a User-Item matrix for the target users based on history
        # Filter history to only relevant customers for speed
        target_set = set(target_customer_ids)
        relevant_history = history_df[history_df["customer_id"].isin(target_set)].copy()

        # Pre-encode history
        relevant_history["user_idx"] = self.encoder.transform_customers(
            relevant_history["customer_id"]
        )
        relevant_history["item_idx"] = self.encoder.transform_articles(
            relevant_history["article_id"]
        )

        # Calculate weights for history (same decay logic)
        max_date = history_df["t_dat"].max()
        relevant_history["diff_days"] = (max_date - relevant_history["t_dat"]).dt.days
        relevant_history["weight"] = 1.0 / np.power(
            relevant_history["diff_days"] + 1.0, self.decay_rate
        )

        # Create a mapping from customer_id to a temporary row index 0..N_target
        target_cust_list = list(target_customer_ids)
        cust_to_row = {cid: i for i, cid in enumerate(target_cust_list)}

        # Map history user_idx to this temporary row index
        # We need to map the original string IDs in history to the new row indices
        # because user_idx in encoder might be sparse relative to target_customer_ids
        relevant_history["row_idx"] = relevant_history["customer_id"].map(cust_to_row)

        # Drop history that doesn't map (shouldn't happen if filtered correctly)
        # Cite debug_lesson_1: Explicitly filter -1 for unseen items as dropna ignores integers
        relevant_history = relevant_history[
            (relevant_history["row_idx"].notna()) & (relevant_history["item_idx"] != -1)
        ]

        # Create Sparse Matrix for Target Users History (U_test)
        # Shape: (Num Target Users, Num Items)
        n_target = len(target_cust_list)
        n_items = self.similarity_matrix.shape[0]

        U_test = sp.csr_matrix(
            (
                relevant_history["weight"].values,
                (
                    relevant_history["row_idx"].astype(int).values,
                    relevant_history["item_idx"].astype(int).values,
                ),
            ),
            shape=(n_target, n_items),
        )

        # Prepare Global Popularity IDs
        global_pop_idxs = self.encoder.transform_articles(self.global_popularity)
        # Filter out -1s
        global_pop_idxs = [idx for idx in global_pop_idxs if idx != -1]

        predictions = []

        # Process in batches to save memory
        for start in range(0, n_target, batch_size):
            end = min(start + batch_size, n_target)

            # Slice user history
            U_batch = U_test[start:end]

            # Compute Scores: (Batch, Items) = (Batch, Items) @ (Items, Items)
            scores = U_batch.dot(self.similarity_matrix)

            # Convert to dense for sorting
            # Note: This might be dense, but batch size controls memory
            scores_dense = scores.toarray()

            # For each user in batch
            for i in range(end - start):
                user_scores = scores_dense[i]

                # Get indices of top k items
                # argpartition is faster than full sort
                if len(user_scores) >= self.top_k:
                    top_idxs = np.argpartition(user_scores, -self.top_k)[-self.top_k :]
                    # Sort these top k strictly
                    top_idxs = top_idxs[np.argsort(user_scores[top_idxs])[::-1]]
                else:
                    top_idxs = np.argsort(user_scores)[::-1]

                # Filter out zero scores (no similarity found)
                top_idxs = [idx for idx in top_idxs if user_scores[idx] > 0]

                # Fill with popularity if needed
                if len(top_idxs) < self.top_k:
                    needed = self.top_k - len(top_idxs)
                    # Add global pop, excluding already selected
                    # (Simple approach: just append and unique, preserving order)
                    candidates = list(top_idxs)
                    for pop_idx in global_pop_idxs:
                        if pop_idx not in candidates:
                            candidates.append(pop_idx)
                            if len(candidates) == self.top_k:
                                break
                    final_idxs = candidates
                else:
                    final_idxs = top_idxs[: self.top_k]

                # Decode
                pred_art_ids = self.encoder.inverse_transform_articles(final_idxs)
                predictions.append(" ".join(pred_art_ids))

        # Create Result DataFrame
        result_df = pd.DataFrame(
            {"customer_id": target_cust_list, "prediction": predictions}
        )

        return result_df


def run_model(load_cached_data=True):
    """
    Main execution function.
    1. Loads data
    2. Trains model
    3. Validates
    4. Generates submission
    """
    # 1. Load Data
    train_df, val_df = load_and_filter_data(load_cached_data=load_cached_data)

    # 2. Initialize and Train Model
    model = TimeWeightedCooccurrence()
    model.fit(train_df, load_cached_data=load_cached_data)

    # 3. Validation
    print("\nRunning Validation...")
    # We predict for customers in the validation set
    val_customers = val_df["customer_id"].unique()

    # Note: For validation, we use train_df as history.
    # The validation set is "future" relative to train_df.
    val_preds = model.predict(val_customers, train_df)

    # Calculate MAP@12
    map_score = calculate_map12(val_df, val_preds)
    print(f"Validation MAP@12: {map_score:.10f}")

    # 4. Submission
    print("\nGenerating Submission...")
    test_customers_df = get_target_customers()
    test_ids = test_customers_df["customer_id"].values

    # For submission, we ideally use all available data (train + val) as history
    # Concatenate train and val for history
    full_history = pd.concat([train_df, val_df], axis=0)

    # Predict
    submission_df = model.predict(test_ids, full_history)

    # Save
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
