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
    A heuristic model combining Global Trend, Item-Item CF, and Repurchase.

    Structure:
    1. Repurchase (Personal History)
    2. Item-Item Collaborative Filtering (Cite solution_lesson_node_00003)
    3. Global Trend (Time-Decayed Popularity)
    """

    def __init__(self, top_k=12, decay_alpha=2.5):
        self.top_k = top_k
        self.decay_alpha = decay_alpha
        self.global_trend = []
        self.customer_history = {}
        self.customer_history_weights = {}
        self.similarity_matrix = None
        self.n_articles = 0

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
        # Store items and weights for prediction
        df["weight"] = weights
        hist_df = df[["customer_id", "article_id", "weight"]].sort_values(
            ["customer_id", "weight"], ascending=[True, False]
        )

        # Group items and weights separately for efficiency
        grouped = hist_df.groupby("customer_id")
        self.customer_history = grouped["article_id"].apply(list).to_dict()
        self.customer_history_weights = grouped["weight"].apply(list).to_dict()

        # --- 3. Item-Item CF Matrix Construction ---
        # Cite solution_lesson_node_00003
        print("Constructing Item-Item Similarity Matrix...")
        self.n_articles = int(df["article_id"].max()) + 1

        # Create sparse user-item matrix with time-decay weights
        X = sp.csr_matrix(
            (weights, (df["customer_id"], df["article_id"])),
            shape=(int(df["customer_id"].max()) + 1, self.n_articles),
        )

        # Apply IDF Weighting (Cite solution_lesson_node_00011)
        # IDF = log(N_users / (N_item + 1))
        # We calculate N_item by binarizing X and summing columns
        X_binary = X.copy()
        X_binary.data = np.ones_like(X_binary.data)
        item_counts = np.array(X_binary.sum(axis=0)).flatten()
        n_users = X.shape[0]
        idf = np.log(n_users / (item_counts + 1))

        # Apply IDF to columns of X
        D_idf = sp.diags(idf)
        X = X @ D_idf

        # Normalize rows (users) instead of columns (items)
        # Cite solution_lesson_node_00004
        X_norm = normalize(X, axis=1, norm="l2")

        # Compute Item-Item Similarity: S = X^T * X
        self.similarity_matrix = X_norm.T @ X_norm

        # Zero out diagonal to prevent self-recommendation in CF step
        self.similarity_matrix.setdiag(0)

        return self

    def predict(self, customer_ids):
        """
        Generates predictions using Cascade: Repurchase -> CF -> Trend.
        """
        predictions = []
        global_trend = self.global_trend
        top_k = self.top_k

        # Batch processing for efficient matrix multiplication
        batch_size = 1000
        n_users = len(customer_ids)

        print(f"Predicting for {n_users} users in batches...")

        for start_idx in range(0, n_users, batch_size):
            end_idx = min(start_idx + batch_size, n_users)
            batch_ids = customer_ids[start_idx:end_idx]

            # Build User-History Matrix for this batch
            # Use weighted history (Cite solution_lesson_node_00005)
            rows = []
            cols = []
            data = []
            for i, cid in enumerate(batch_ids):
                hist_items = self.customer_history.get(cid, [])
                hist_weights = self.customer_history_weights.get(cid, [])

                # Deduplicate keeping max weight (most recent)
                seen_items = {}
                for item, w in zip(hist_items, hist_weights):
                    if item not in seen_items:
                        seen_items[item] = w
                    # Since sorted by weight desc, first occurrence is max weight

                if seen_items:
                    r_idx = [i] * len(seen_items)
                    c_idx = list(seen_items.keys())
                    vals = list(seen_items.values())

                    rows.extend(r_idx)
                    cols.extend(c_idx)
                    data.extend(vals)

            # Compute CF Scores: User_History @ Item_Similarity
            batch_scores = None
            if len(rows) > 0 and self.similarity_matrix is not None:
                X_batch = sp.csr_matrix(
                    (data, (rows, cols)),
                    shape=(len(batch_ids), self.n_articles),
                )
                batch_scores = X_batch @ self.similarity_matrix

            # Generate predictions for each user in batch
            for i, cid in enumerate(batch_ids):
                user_history = self.customer_history.get(cid, [])
                selection = []
                seen = set()

                # A. Repurchase (History)
                for item in user_history:
                    if item not in seen:
                        selection.append(item)
                        seen.add(item)
                        if len(selection) >= top_k:
                            break

                # B. Collaborative Filtering
                # Cite solution_lesson_node_00003
                if len(selection) < top_k and batch_scores is not None:
                    # Get scores for this user
                    user_scores = batch_scores.getrow(i)

                    if user_scores.nnz > 0:
                        # Extract indices and values
                        candidates = user_scores.indices
                        scores = user_scores.data

                        # Sort by score descending
                        # Since it's sparse, sorting is usually fast enough
                        sorted_indices = np.argsort(scores)[::-1]

                        for idx in sorted_indices:
                            item = candidates[idx]
                            if item not in seen:
                                selection.append(item)
                                seen.add(item)
                                if len(selection) >= top_k:
                                    break

                # C. Global Trend
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
    model = TrendRepurchaseCascade(top_k=Config.TOP_K, decay_alpha=Config.DECAY_ALPHA)
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
