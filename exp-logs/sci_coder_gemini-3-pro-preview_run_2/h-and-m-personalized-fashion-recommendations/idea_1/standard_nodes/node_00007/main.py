import os
import sys
import numpy as np
import pandas as pd
import warnings
import random

# Import from provided libraries
from library.config import SUBMISSION_PATH, SUBMISSION_DIR, RANDOM_STATE
from library.data_factory import load_and_filter_data, get_target_customers
from library.model import TimeWeightedCooccurrence
from library.metrics import calculate_map12, apk

# --- Configuration & Setup ---
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_failure_analysis(val_df, val_preds, train_df):
    """
    Analyzes model performance by calculating per-user AP@12 and correlating
    it with user history features (history length, recency, average spend).
    """
    print("\nPerforming Failure Analysis...")

    # 1. Prepare Ground Truth
    # Ensure article_id is string
    val_df_copy = val_df.copy()
    val_df_copy["article_id"] = val_df_copy["article_id"].astype(str)
    ground_truth = val_df_copy.groupby("customer_id")["article_id"].unique()

    # 2. Prepare Predictions
    preds_map = val_preds.set_index("customer_id")["prediction"].to_dict()

    # 3. Calculate AP per user
    user_metrics = []

    # Identify common customers
    common_customers = ground_truth.index.intersection(val_preds["customer_id"])

    for cust_id in common_customers:
        actual = ground_truth.loc[cust_id]
        pred_str = preds_map.get(cust_id, "")
        predicted = pred_str.split()

        score = apk(actual, predicted, k=12)
        user_metrics.append({"customer_id": cust_id, "ap_score": score})

    metrics_df = pd.DataFrame(user_metrics)

    if metrics_df.empty:
        print("No overlapping customers for failure analysis.")
        return

    # 4. Extract User History Features from Training Data
    # We analyze the 'train_df' because that's what the model used to make predictions
    train_df_copy = train_df[train_df["customer_id"].isin(common_customers)].copy()
    train_df_copy["t_dat"] = pd.to_datetime(train_df_copy["t_dat"])
    max_date = train_df_copy["t_dat"].max()

    # Aggregations
    # - History Length (count)
    # - Recency (days since last purchase relative to end of train period)
    # - Avg Price (spending power)

    # Helper for recency
    train_df_copy["days_since"] = (max_date - train_df_copy["t_dat"]).dt.days

    user_features = (
        train_df_copy.groupby("customer_id")
        .agg(
            history_length=("article_id", "count"),
            recency_min=("days_since", "min"),  # Minimum days since = most recent
            avg_price=("price", "mean"),
        )
        .reset_index()
    )

    # 5. Merge and Correlate
    analysis_df = metrics_df.merge(user_features, on="customer_id", how="inner")

    if analysis_df.empty:
        print("Could not merge features for analysis.")
        return

    # Calculate correlations with AP Score
    # Note: 'recency_min' -> lower is more recent.
    # If correlation is negative, it means higher recency (older) -> lower score (worse).
    # If correlation is positive, it means higher recency (older) -> higher score.

    correlations = analysis_df[
        ["ap_score", "history_length", "recency_min", "avg_price"]
    ].corr()["ap_score"]

    print("-" * 30)
    print("Correlation with Error Magnitude (AP Score):")
    print("(Positive corr with AP means feature is associated with BETTER performance)")
    print(f"  - History Length: {correlations['history_length']:.4f}")
    print(f"  - Recency (Days ago): {correlations['recency_min']:.4f}")
    print(f"  - Average Price:    {correlations['avg_price']:.4f}")
    print("-" * 30)

    # Interpretation for the log
    print("Interpretation:")
    if correlations["history_length"] > 0:
        print("  -> Users with more history tend to have better predictions.")
    else:
        print("  -> Users with more history tend to have worse predictions.")

    if correlations["recency_min"] < 0:
        print(
            "  -> Users who bought recently (low days ago) tend to have better predictions."
        )
    else:
        print(
            "  -> Recent activity does not strongly correlate with better predictions."
        )


def main():
    # 1. Setup
    set_seed(RANDOM_STATE)

    # 2. Load Data
    # Using cached data if available for speed
    print("Loading data...")
    train_df, val_df = load_and_filter_data(load_cached_data=True)

    # 3. Train Model
    print("Initializing and training model...")
    model = TimeWeightedCooccurrence()
    # fit uses caching internally if load_cached_data=True
    model.fit(train_df, load_cached_data=True)

    # 4. Validation Inference
    print("Running validation inference...")
    val_customers = val_df["customer_id"].unique()

    # Predict using training data as history
    val_preds = model.predict(val_customers, train_df)

    # 5. Validation Metric
    map_score = calculate_map12(val_df, val_preds)
    # REQUIRED FORMAT
    print(f"Final Validation Metric: {map_score:.16f}")

    # 6. Failure Analysis
    perform_failure_analysis(val_df, val_preds, train_df)

    # 7. Generate Submission
    if map_score > 0.0096263154326182:
        print("Generating final submission...")
        test_customers_df = get_target_customers()
        test_ids = test_customers_df["customer_id"].values

        # Combine history for final inference
        full_history = pd.concat([train_df, val_df], axis=0)

        submission_df = model.predict(test_ids, full_history)

        # Save
        SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"Validation score ({map_score}) did not beat threshold. Skipping submission."
        )


if __name__ == "__main__":
    main()
