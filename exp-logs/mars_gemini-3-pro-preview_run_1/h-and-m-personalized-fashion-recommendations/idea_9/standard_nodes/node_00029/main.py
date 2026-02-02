import pandas as pd
import numpy as np
import os
import sys
import gc

# Import provided library modules
from library.config import Config
from library.data_manager import DataManager
from library.model_dwsc import DWSCRecommender
from library.evaluation import calculate_map12


def set_seed(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_failure_analysis(target_df, submission_df, train_df, customers_path):
    print("\nPerforming Failure Analysis...")

    # 1. Calculate Per-User AP (Error Magnitude)
    # Replicating logic from evaluation.py to get granular scores
    truth = target_df[["customer_id", "article_id"]].copy()
    preds = submission_df[["customer_id", "prediction"]].copy()

    # Format IDs
    if pd.api.types.is_numeric_dtype(truth["article_id"]):
        truth["article_id"] = truth["article_id"].apply(lambda x: f"{x:010d}")
    else:
        truth["article_id"] = truth["article_id"].astype(str).str.zfill(10)

    truth_grouped = truth.groupby("customer_id")["article_id"].apply(set).reset_index()
    truth_grouped.rename(columns={"article_id": "actual"}, inplace=True)

    preds["prediction"] = preds["prediction"].astype(str).str.split()

    merged = truth_grouped.merge(preds, on="customer_id", how="left")

    def get_ap(row):
        actual = row["actual"]
        predicted = row["prediction"]
        if not actual or not isinstance(predicted, list):
            return 0.0

        predicted = predicted[:12]
        score = 0.0
        num_hits = 0.0
        for k, item in enumerate(predicted):
            if item in actual:
                num_hits += 1.0
                score += num_hits / (k + 1.0)
        return score / min(len(actual), 12)

    merged["ap"] = merged.apply(get_ap, axis=1)
    merged["error_magnitude"] = 1.0 - merged["ap"]

    # 2. Extract User Features
    # A. Metadata (Age)
    cust_df = pd.read_csv(customers_path, usecols=["customer_id", "age", "Active"])
    cust_df["age"] = cust_df["age"].fillna(cust_df["age"].median())
    cust_df["Active"] = cust_df["Active"].fillna(0)

    # B. Behavioral (History Length, Recency)
    # train_df has 'customer_id' and 'days_elapsed'
    user_stats = (
        train_df.groupby("customer_id")
        .agg(
            history_length=("article_id", "count"),
            min_days_elapsed=("days_elapsed", "min"),  # min days = most recent
        )
        .reset_index()
    )

    # 3. Merge
    analysis_df = merged[["customer_id", "error_magnitude"]].merge(
        cust_df, on="customer_id", how="left"
    )
    analysis_df = analysis_df.merge(user_stats, on="customer_id", how="left")

    # Fill missing behavioral stats (implies 0 history in training window)
    analysis_df["history_length"] = analysis_df["history_length"].fillna(0)
    analysis_df["min_days_elapsed"] = analysis_df["min_days_elapsed"].fillna(999)

    # 4. Correlation
    features = ["age", "Active", "history_length", "min_days_elapsed"]
    print("Correlation with Error Magnitude (1 - AP):")
    correlations = (
        analysis_df[features + ["error_magnitude"]]
        .corr()["error_magnitude"]
        .drop("error_magnitude")
    )
    print(correlations)

    return correlations


def main():
    # 1. Setup
    set_seed(Config.SEED)
    Config.setup()

    # 2. Data Manager
    dm = DataManager()

    # =========================================================================
    # VALIDATION PHASE
    # =========================================================================
    print("--- Starting Validation Phase ---")

    # Load Validation Data
    # train_val: History for users (last 10 weeks before validation week)
    # target_val: Ground truth (validation week)
    # test_users_val: Users to predict for
    train_val, target_val, test_users_val = dm.get_validation_data(
        load_cached_data=True
    )

    # Initialize Model
    model = DWSCRecommender()

    # Fit Model
    # We pass the total number of users/items known to the encoder to ensure matrix dimensions are correct
    n_users = len(dm.encoder.user_to_idx)
    n_items = len(dm.encoder.item_to_idx)
    model.fit(train_val, n_users, n_items, load_cached_data=True)

    # Predict
    # Note: predict() saves to Config.SUBMISSION_PATH, but we capture the DF return for scoring
    val_preds = model.predict(
        test_users_val, train_val, dm.encoder, load_cached_data=True
    )

    # Score
    metric = calculate_map12(target_val, val_preds)
    print(f"Final Validation Metric: {metric:.10f}")

    # Failure Analysis
    perform_failure_analysis(target_val, val_preds, train_val, Config.CUSTOMERS_PATH)

    # Cleanup to free memory before full training
    del model, train_val, target_val, val_preds
    gc.collect()

    # =========================================================================
    # SUBMISSION PHASE
    # =========================================================================
    threshold = 0.0265060791

    if metric > threshold:
        print(
            f"\nMetric ({metric:.5f}) > Threshold ({threshold}). Proceeding to Submission..."
        )

        # Load Full Submission Data
        # train_full: History for users (last 10 weeks relative to test period)
        # test_users_full: All users in sample_submission
        train_full, test_users_full = dm.get_submission_data(load_cached_data=True)

        # Re-Initialize Model (Fresh instance)
        model_full = DWSCRecommender()

        # We might have new users/items in the full set, but the encoder was fit on the union
        # inside DataManager, so n_users/n_items from dm.encoder remains valid.
        n_users = len(dm.encoder.user_to_idx)
        n_items = len(dm.encoder.item_to_idx)

        # Fit on Full Data
        # Note: We set load_cached_data=False for the matrix building steps to ensure
        # we don't accidentally load validation matrices, although the cache keys in
        # SparseEngine are generic.
        # However, the library code uses specific filenames like 'X_decay.npz'.
        # To avoid conflict, we should ideally clear cache or rely on the fact that
        # DataManager handles different splits.
        # Looking at SparseEngine, it saves to 'X_decay.npz'. If we run validation then submission,
        # submission would load validation matrices if we aren't careful.
        # Strategy: We will force re-computation for the submission fit by setting load_cached_data=False
        # for the model fit method.
        model_full.fit(train_full, n_users, n_items, load_cached_data=False)

        # Predict
        # This will overwrite the validation submission file at Config.SUBMISSION_PATH
        model_full.predict(
            test_users_full, train_full, dm.encoder, load_cached_data=False
        )

        print("Submission generation complete.")
    else:
        print(
            f"\nMetric ({metric:.5f}) <= Threshold ({threshold}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
