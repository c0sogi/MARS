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

    def __init__(self, top_k=12, decay_alpha=2.5, cf_neighbors=None):
        self.top_k = top_k
        self.decay_alpha = decay_alpha
        self.cf_neighbors = cf_neighbors  # Unused in vectorized approach
        self.global_trend = []
        self.history_items = {}
        self.history_days = {}
        self.S = None
        self.n_items = 0

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
        hist_df = df[["customer_id", "article_id", "days_elapsed"]].sort_values(
            ["customer_id", "days_elapsed"], ascending=[True, True]
        )
        # Cite solution_lesson_node_00005: Store days_elapsed to use for recency weighting in inference
        grouped = hist_df.groupby("customer_id")
        self.history_items = grouped["article_id"].apply(list).to_dict()
        self.history_days = grouped["days_elapsed"].apply(list).to_dict()

        # --- 3. Item-Item Similarity (CF) ---
        print("Computing Item-Item Similarity with IDF and Vectorization...")
        row_ind = df["customer_id"].values
        col_ind = df["article_id"].values

        # Ensure dimensions cover all mapped IDs
        n_users = row_ind.max() + 1
        self.n_items = col_ind.max() + 1

        # Cite solution_lesson_node_00011: Apply IDF weighting to interaction matrix
        # Count unique users per item
        unique_pairs = df[["customer_id", "article_id"]].drop_duplicates()
        user_counts = np.bincount(
            unique_pairs["article_id"].values, minlength=self.n_items
        )
        idf = np.log(n_users / (user_counts + 1))

        # Apply IDF to time-decay weights
        weights = weights * idf[col_ind]

        # Create Sparse User-Item Matrix (Time-Weighted & IDF-Weighted)
        X = sp.csr_matrix((weights, (row_ind, col_ind)), shape=(n_users, self.n_items))

        # Cite solution_lesson_node_00004: Row-wise Normalization
        X = normalize(X, norm="l2", axis=1)

        # Compute Similarity: A^T * A
        # Result is Cosine Similarity between items based on user vectors
        self.S = X.T.dot(X)
        self.S.setdiag(0)

        # Cite solution_lesson_node_00012: Removed hard neighbor pruning to capture long-tail signals
        return self

    def predict(self, customer_ids, batch_size=1000):
        """
        Generates predictions using History -> CF -> Trend cascade.
        Cite solution_lesson_node_00012: Implements vectorized inference.
        """
        predictions = []
        global_trend = self.global_trend
        top_k = self.top_k
        items_get = self.history_items.get
        days_get = self.history_days.get
        decay_alpha = self.decay_alpha

        # Process in batches to manage memory
        for i in range(0, len(customer_ids), batch_size):
            batch_ids = customer_ids[i : i + batch_size]

            # 1. Build Sparse History Matrix for Batch (U_batch)
            rows = []
            cols = []
            data = []
            batch_history_sets = []

            for r, cid in enumerate(batch_ids):
                items = items_get(cid, [])
                days = days_get(cid, [])
                batch_history_sets.append(set(items))

                for item, day in zip(items, days):
                    rows.append(r)
                    cols.append(item)
                    # Cite solution_lesson_node_00005: Recency weighting for seed items
                    data.append(1.0 / (day + decay_alpha))

            U_batch = sp.csr_matrix(
                (data, (rows, cols)), shape=(len(batch_ids), self.n_items)
            )

            # 2. Vectorized CF Scoring
            # Scores = U_batch * S
            batch_scores = U_batch.dot(self.S)

            # 3. Selection
            scores_dense = batch_scores.toarray()

            for r in range(len(batch_ids)):
                user_preds = []
                seen = set()

                # A. Personal History (Repurchase)
                hist_items = items_get(batch_ids[r], [])
                for item in hist_items:
                    if item not in seen:
                        user_preds.append(item)
                        seen.add(item)
                        if len(user_preds) >= top_k:
                            break

                # B. Collaborative Filtering
                if len(user_preds) < top_k:
                    # Mask history items in scores to avoid re-recommending
                    for h_item in hist_items:
                        if h_item < self.n_items:
                            scores_dense[r, h_item] = -np.inf

                    # Get top candidates from CF
                    # Grab more candidates than needed to be safe
                    n_candidates = min(self.n_items, top_k + len(hist_items))

                    # Use argpartition for efficiency
                    top_indices = np.argpartition(scores_dense[r], -n_candidates)[
                        -n_candidates:
                    ]

                    # Sort top candidates by score descending
                    top_candidates = sorted(
                        zip(top_indices, scores_dense[r][top_indices]),
                        key=lambda x: x[1],
                        reverse=True,
                    )

                    for item, score in top_candidates:
                        if score <= 0:
                            break
                        if item not in seen:
                            user_preds.append(item)
                            seen.add(item)
                            if len(user_preds) >= top_k:
                                break

                # C. Global Trend
                if len(user_preds) < top_k:
                    for item in global_trend:
                        if item not in seen:
                            user_preds.append(item)
                            seen.add(item)
                            if len(user_preds) >= top_k:
                                break

                predictions.append(user_preds)

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
    model = TrendRepurchaseCascade(top_k=Config.TOP_K, decay_alpha=Config.DECAY_ALPHA)
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
