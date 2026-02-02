import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import set_seed
from library.metrics import calculate_map12
from library.data_loader import load_filtered_transactions


class TrendRepurchaseCascade:
    """
    A heuristic model combining Global Trend (Time-Decayed Popularity)
    and Personal History (Repurchase).

    The model operates in two stages:
    1. Trend: Calculates a global score for items based on sales velocity.
    2. History: Retrieves a user's recent purchases.

    Prediction is a cascade: History -> Fill with Trend.
    """

    def __init__(self, top_k=12, decay_alpha=2.5):
        self.top_k = top_k
        self.decay_alpha = decay_alpha
        self.global_trend = []
        self.customer_history = {}

    def fit(self, df):
        """
        Fits the model components using the provided transaction DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing columns:
                'customer_id' (int): Mapped customer index.
                'article_id' (int): Mapped article index.
                'days_elapsed' (int): Days since the transaction (0 = most recent).
        """
        # --- 1. Global Trend Calculation ---
        # Formula: Score = Sum(1 / (days_elapsed + alpha))
        # We use numpy arrays for efficient computation
        days = df["days_elapsed"].values
        # Calculate weight for each transaction
        weights = 1.0 / (days + self.decay_alpha)

        # Aggregate weights by article_id
        # Create a temporary DF for grouping
        trend_df = pd.DataFrame(
            {"article_id": df["article_id"].values, "weight": weights}
        )
        global_scores = trend_df.groupby("article_id")["weight"].sum()

        # Sort by score descending and take top K
        self.global_trend = (
            global_scores.sort_values(ascending=False).head(self.top_k).index.tolist()
        )

        # --- 2. Customer History Construction ---
        # We want a list of articles for each customer, ordered by recency.
        # days_elapsed=0 is the most recent.

        # Sort by customer then by recency (days_elapsed ascending)
        # We select only necessary columns to save memory
        hist_df = df[["customer_id", "article_id", "days_elapsed"]].sort_values(
            ["customer_id", "days_elapsed"], ascending=[True, True]
        )

        # Group by customer and collect article_ids into lists
        # This creates a dict: {customer_idx: [art_idx_recent, art_idx_older, ...]}
        self.customer_history = (
            hist_df.groupby("customer_id")["article_id"].apply(list).to_dict()
        )

        return self

    def predict(self, customer_ids):
        """
        Generates predictions for the given list of customer IDs.

        Args:
            customer_ids (iterable): List or array of mapped customer integer indices.

        Returns:
            list of lists: Each inner list contains 'top_k' article integer indices.
        """
        predictions = []
        global_trend = self.global_trend
        top_k = self.top_k
        # Localize lookup for speed
        history_get = self.customer_history.get

        for cid in customer_ids:
            # Retrieve user history (default to empty list if new user)
            user_history = history_get(cid, [])

            selection = []
            seen = set()

            # Step A: Fill with Personal History
            for item in user_history:
                if item not in seen:
                    selection.append(item)
                    seen.add(item)
                    if len(selection) >= top_k:
                        break

            # Step B: Fill with Global Trend if slots remain
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
