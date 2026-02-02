import pandas as pd
import numpy as np
import os
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from library.config import Config
from library.utils import set_seed
from library.metrics import calculate_map12
from library.data_loader import load_filtered_transactions


class TrendRepurchaseCascade:
    """
    A heuristic model combining Global Trend (Time-Decayed Popularity),
    Item-Item Collaborative Filtering, and Personal History (Repurchase).

    The model operates in three stages:
    1. History: Retrieves a user's recent purchases.
    2. CF: Retrieves items similar to those in history (Time-Weighted Co-occurrence).
    3. Trend: Fills remaining slots with global popularity.

    Prediction is a cascade: History -> CF -> Trend.
    """

    def __init__(
        self,
        top_k=12,
        decay_alpha=2.5,
        cf_neighbors=20,
        repurchase_window=35,
    ):
        self.top_k = top_k
        self.decay_alpha = decay_alpha
        self.cf_neighbors = cf_neighbors
        self.repurchase_window = repurchase_window
        self.global_trend = []
        self.repurchase_items = {}
        self.seed_items = {}
        self.seed_days = {}
        self.item_similarity = {}

    def fit(self, df):
        """
        Fits the model components using the provided transaction DataFrame.
        """
        # --- 1. Global Trend Calculation ---
        days = df["days_elapsed"].values
        weights = 1.0 / (days + self.decay_alpha)

        trend_df = pd.DataFrame(
            {"article_id": df["article_id"].values, "weight": weights}
        )
        global_scores = trend_df.groupby("article_id")["weight"].sum()
        self.global_trend = (
            global_scores.sort_values(ascending=False).head(self.top_k).index.tolist()
        )

        # --- 2. Customer History Construction ---
        # Sort by customer and recency
        hist_df = df[["customer_id", "article_id", "days_elapsed"]].sort_values(
            ["customer_id", "days_elapsed"], ascending=[True, True]
        )

        # A. Full History for CF Seeds (Cite Lesson 3 & 5)
        grouped = hist_df.groupby("customer_id")
        self.seed_items = grouped["article_id"].apply(list).to_dict()
        self.seed_days = grouped["days_elapsed"].apply(list).to_dict()

        # B. Recent History for Direct Repurchase
        # Filter to window to avoid stale recommendations
        rep_df = hist_df[hist_df["days_elapsed"] <= self.repurchase_window]
        grouped_rep = rep_df.groupby("customer_id")
        self.repurchase_items = grouped_rep["article_id"].apply(list).to_dict()

        # --- 3. Item-Item Similarity (CF) ---
        print("Computing Item-Item Similarity...")
        row_ind = df["customer_id"].values
        col_ind = df["article_id"].values

        # Ensure dimensions cover all mapped IDs
        n_users = row_ind.max() + 1
        n_items = col_ind.max() + 1

        # Create Sparse User-Item Matrix (Time-Weighted)
        X = sp.csr_matrix((weights, (row_ind, col_ind)), shape=(n_users, n_items))

        # Normalize rows (users) to prevent active users from dominating co-occurrence
        X = normalize(X, norm="l2", axis=1)

        # Compute Similarity: A^T * A
        # Result is Cosine Similarity between items based on user vectors
        S = X.T.dot(X)
        S.setdiag(0)

        # Prune Matrix to top K neighbors per item to save memory/time
        print("Pruning Similarity Matrix...")
        for i in range(n_items):
            start = S.indptr[i]
            end = S.indptr[i + 1]
            if start == end:
                continue

            cols = S.indices[start:end]
            vals = S.data[start:end]

            # Keep top K
            if len(cols) > self.cf_neighbors:
                top_k_idx = np.argpartition(vals, -self.cf_neighbors)[
                    -self.cf_neighbors :
                ]
                cols = cols[top_k_idx]
                vals = vals[top_k_idx]

            # Store sorted list of (item_idx, score)
            if len(cols) > 0:
                sorted_pairs = sorted(zip(cols, vals), key=lambda x: x[1], reverse=True)
                self.item_similarity[i] = sorted_pairs

        return self

    def predict(self, customer_ids):
        """
        Generates predictions using History -> CF -> Trend cascade.
        """
        predictions = []
        global_trend = self.global_trend
        top_k = self.top_k

        # Accessors for speed
        rep_get = self.repurchase_items.get
        seed_items_get = self.seed_items.get
        seed_days_get = self.seed_days.get
        sim_get = self.item_similarity.get
        decay_alpha = self.decay_alpha

        for cid in customer_ids:
            selection = []
            seen = set()

            # Step A: Personal History (Repurchase)
            # Only use items within the repurchase window
            user_rep_items = rep_get(cid, [])
            for item in user_rep_items:
                if item not in seen:
                    selection.append(item)
                    seen.add(item)
                    if len(selection) >= top_k:
                        break

            # Step B: Collaborative Filtering (Similar Items)
            if len(selection) < top_k:
                cf_candidates = {}
                # Use FULL history (seeds) to find similar items
                # Cite solution_lesson_node_00005: Weight neighbor contribution by recency of seed item
                user_seeds = seed_items_get(cid, [])
                user_seed_days = seed_days_get(cid, [])

                for hist_item, hist_day in zip(user_seeds, user_seed_days):
                    weight = 1.0 / (hist_day + decay_alpha)
                    neighbors = sim_get(hist_item, [])
                    for neighbor, score in neighbors:
                        if neighbor not in seen:
                            cf_candidates[neighbor] = (
                                cf_candidates.get(neighbor, 0.0) + score * weight
                            )

                # Sort by accumulated score
                sorted_cf = sorted(
                    cf_candidates.items(), key=lambda x: x[1], reverse=True
                )

                for item, score in sorted_cf:
                    if len(selection) >= top_k:
                        break
                    selection.append(item)
                    seen.add(item)

            # Step C: Global Trend
            if len(selection) < top_k:
                for item in global_trend:
                    if item not in seen:
                        selection.append(item)
                        seen.add(item)
                        if len(selection) >= top_k:
                            break

            predictions.append(selection)

        return predictions


def run_validation():
    """
    Runs the validation pipeline.
    Splits data into history (Train) and last 7 days (Validation).
    Calculates MAP@12.
    """
    print("--- Starting Validation ---")
    set_seed()

    # Load data using the configured history window
    df, mapper = load_filtered_transactions(weeks=Config.HISTORY_WEEKS)

    # Split Data
    # days_elapsed < 7 corresponds to the last 7 days (Validation Target)
    # days_elapsed >= 7 corresponds to the prior history (Training Data)
    val_mask = df["days_elapsed"] < 7
    train_df = df[~val_mask].copy()
    val_df = df[val_mask].copy()

    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")

    # Fit Model
    print("Fitting model on training split...")
    model = TrendRepurchaseCascade(
        top_k=Config.TOP_K,
        decay_alpha=Config.DECAY_ALPHA,
        cf_neighbors=Config.CF_NEIGHBORS,
        repurchase_window=Config.REPURCHASE_WINDOW_DAYS,
    )
    model.fit(train_df)

    # Predict
    val_customers = val_df["customer_id"].unique()
    print(f"Predicting for {len(val_customers)} unique validation customers...")
    preds_int = model.predict(val_customers)

    # Format for Metric Calculation
    # The metric function expects space-separated strings in a 'prediction' column
    pred_strings = []
    for p_list in preds_int:
        # Join integer IDs with spaces
        p_str = " ".join(map(str, p_list))
        pred_strings.append(p_str)

    submission_df = pd.DataFrame(
        {"customer_id": val_customers, "prediction": pred_strings}
    )

    # Calculate MAP@12
    print("Calculating MAP@12...")
    score = calculate_map12(val_df, submission_df)
    print(f"Validation MAP@12: {score:.10f}")

    return score


def run_submission():
    """
    Runs the full submission pipeline.
    Retrains on all available data and predicts for the test set.
    """
    print("--- Starting Submission Generation ---")
    set_seed()

    # Load full dataset
    df, mapper = load_filtered_transactions(weeks=Config.HISTORY_WEEKS)

    # Fit Model on all data
    print("Fitting model on full dataset...")
    model = TrendRepurchaseCascade(
        top_k=Config.TOP_K,
        decay_alpha=Config.DECAY_ALPHA,
        cf_neighbors=Config.CF_NEIGHBORS,
        repurchase_window=Config.REPURCHASE_WINDOW_DAYS,
    )
    model.fit(df)

    # Load Test Customers
    print("Loading test customer list...")
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    # Transform test customers to integer indices
    # This handles the mapping. Unknown customers (if any) map to -1.
    test_ids_int = mapper.transform_customers(test_df, col="customer_id")

    # Predict
    print(f"Generating predictions for {len(test_ids_int)} customers...")
    preds_int = model.predict(test_ids_int)

    # Format Predictions
    print("Formatting predictions and mapping back to original IDs...")
    pred_strings = []
    idx_to_article = mapper.idx_to_article

    for p_list in preds_int:
        # Map mapped_int -> original_int -> formatted_string
        # Original article IDs are integers (e.g., 108775015)
        # Submission requires 10-digit strings (e.g., "0108775015")
        art_strs = []
        for idx in p_list:
            original_id = idx_to_article.get(idx, -1)
            if original_id != -1:
                art_strs.append(str(original_id).zfill(10))

        pred_strings.append(" ".join(art_strs))

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"customer_id": test_df["customer_id"], "prediction": pred_strings}
    )

    # Save
    save_path = Config.SUBMISSION_FILE
    print(f"Saving submission to {save_path}...")
    submission_df.to_csv(save_path, index=False)
    print("Submission generation complete.")
