import pandas as pd
import numpy as np
import warnings
import os

# Import from provided library files
from library.config import Config
from library.utils import set_seed
from library.data_loader import load_filtered_transactions
from library.model import TrendRepurchaseCascade, run_submission
from library.metrics import calculate_map12

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    set_seed()

    # 2. Validation & Failure Analysis
    print("--- Starting Validation & Failure Analysis ---")

    # Load data (Validation Split)
    # We use the same configuration as the library defaults
    # load_cached_data=True ensures we use the cache if available
    df, mapper = load_filtered_transactions(
        weeks=Config.HISTORY_WEEKS, load_cached_data=True
    )

    # Split into Train (History) and Validation (Last 7 days)
    # days_elapsed < 7 implies the most recent 7 days (0 to 6)
    val_mask = df["days_elapsed"] < 7
    train_df = df[~val_mask].copy()
    val_df = df[val_mask].copy()

    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

    # Fit Model
    print("Fitting model on training split...")
    model = TrendRepurchaseCascade(
        top_k=Config.TOP_K,
        decay_alpha=Config.DECAY_ALPHA,
        cf_neighbors=Config.CF_NEIGHBORS,
    )
    model.fit(train_df)

    # Predict on Validation
    val_customers = val_df["customer_id"].unique()
    print(f"Predicting for {len(val_customers)} validation customers...")
    preds_int = model.predict(val_customers)

    # Format predictions for MAP calculation
    # The metric function expects a DataFrame with 'customer_id' and 'prediction' (space-separated string)
    pred_strings = []
    for p_list in preds_int:
        pred_strings.append(" ".join(map(str, p_list)))

    submission_df = pd.DataFrame(
        {"customer_id": val_customers, "prediction": pred_strings}
    )

    # Calculate and Print Official Metric
    map_score = calculate_map12(val_df, submission_df)
    print(f"Final Validation Metric: {map_score}")

    # --- Failure Analysis ---
    print("\n--- Performing Failure Analysis ---")

    # Calculate Per-User AP for correlation analysis
    # We need to re-calculate AP locally to get per-user granularity,
    # as calculate_map12 returns the aggregated mean.

    # Ground truth: customer_id -> set of article_ids
    ground_truth = val_df.groupby("customer_id")["article_id"].apply(set).to_dict()
    preds_map = submission_df.set_index("customer_id")["prediction"].to_dict()

    user_errors = []
    user_ids = []

    for cid in val_customers:
        actual_items = ground_truth.get(cid, set())
        if not actual_items:
            continue

        pred_str = preds_map.get(cid, "")
        try:
            predicted_items = [int(x) for x in pred_str.split()][: Config.TOP_K]
        except:
            predicted_items = []

        # Calculate AP
        score = 0.0
        num_hits = 0.0
        for i, p in enumerate(predicted_items):
            if p in actual_items:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        ap = score / min(len(actual_items), Config.TOP_K)

        # Error Magnitude = 1.0 - AP (Higher error means lower performance)
        user_errors.append(1.0 - ap)
        user_ids.append(cid)

    analysis_df = pd.DataFrame(
        {"customer_id": user_ids, "error_magnitude": user_errors}
    )

    # Load Customer Metadata for Features
    print("Loading customer metadata for analysis...")
    cust_df = pd.read_csv(Config.CUSTOMERS_PATH)

    # Map customer strings to integers to merge with analysis_df
    # Note: mapper is already fitted from load_filtered_transactions
    cust_df["customer_id_int"] = mapper.transform_customers(cust_df)

    # Merge Error data with Customer Features
    # Drop original string customer_id from right DF to avoid collision with integer customer_id in left DF
    merged_df = analysis_df.merge(
        cust_df.drop(columns=["customer_id"]),
        left_on="customer_id",
        right_on="customer_id_int",
        how="left",
    )

    # Add History Length (Activity Level) from Train Data
    print("Calculating user activity features...")
    activity_counts = train_df["customer_id"].value_counts().reset_index()
    activity_counts.columns = ["customer_id", "history_count"]

    merged_df = merged_df.merge(activity_counts, on="customer_id", how="left")
    merged_df["history_count"] = merged_df["history_count"].fillna(0)

    # Preprocess Features for Correlation
    # Fill NaNs
    if "age" in merged_df.columns:
        merged_df["age"] = merged_df["age"].fillna(merged_df["age"].median())

    for col in ["club_member_status", "fashion_news_frequency"]:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna("MISSING")
            # Create coded column
            merged_df[f"{col}_code"] = merged_df[col].astype("category").cat.codes

    # Calculate Correlations
    features_to_check = [
        "age",
        "club_member_status_code",
        "fashion_news_frequency_code",
        "history_count",
    ]
    valid_features = [f for f in features_to_check if f in merged_df.columns]

    print("\nCorrelation between Model Error (1 - AP) and Features:")
    if valid_features:
        correlations = (
            merged_df[["error_magnitude"] + valid_features]
            .corr()["error_magnitude"]
            .drop("error_magnitude")
        )
        print(correlations)
    else:
        print("Insufficient features for correlation analysis.")

    # 3. Submission
    baseline_score = 0.024574069561944318
    if map_score > baseline_score:
        print(
            f"\nMetric ({map_score:.6f}) improved over baseline ({baseline_score:.6f})."
        )
        print("--- Generating Submission ---")
        # run_submission handles full retraining (on all weeks) and test prediction
        run_submission()
    else:
        print(
            f"\nMetric ({map_score:.6f}) did not improve over baseline ({baseline_score:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
