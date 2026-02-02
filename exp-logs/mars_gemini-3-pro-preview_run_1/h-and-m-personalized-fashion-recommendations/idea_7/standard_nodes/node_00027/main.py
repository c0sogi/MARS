import os
import numpy as np
import pandas as pd
import random
import warnings
from scipy import stats

# Import provided library modules
from library.config import Config
from library.smdc_model import SMDCRecommender
from library.metrics import calculate_map12

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_per_user_ap(ground_truth, predictions):
    """
    Calculates Average Precision @ 12 for each user.
    Returns a dictionary {customer_id: score}.
    """
    user_scores = {}

    # Pre-process predictions into a dict for fast lookup if it's a list
    # The recommender returns a list of strings corresponding to the input dataframe order
    # We need to map them back to customer_ids.
    # However, for failure analysis, we assume inputs are aligned or dicts.

    # In this script, we will handle the alignment explicitly.
    pass
    # Logic implemented inline in main for context awareness
    return user_scores


def failure_analysis(recommender, preds, gt_df):
    print("\n--- Failure Analysis ---")

    # 1. Calculate Per-User AP
    # preds is a list of strings.
    # recommender.test_df contains the customer_ids in the same order.
    test_users = recommender.test_df["customer_id"].values

    # Map customer_id -> prediction string
    pred_map = dict(zip(test_users, preds))

    # Map customer_id -> ground truth list
    # gt_df is a Series with index customer_id and values list of article_ids
    gt_map = gt_df.to_dict()

    scores = []
    ages = []
    hist_lens = []

    # Pre-compute history lengths from train_df
    user_hist_counts = recommender.train_df["customer_id"].value_counts().to_dict()

    # Pre-compute ages
    # recommender.customers_df has 'customer_id' and 'age'
    # We use a dict for fast lookup
    age_map = dict(
        zip(recommender.customers_df["customer_id"], recommender.customers_df["age"])
    )

    for uid in test_users:
        if uid not in gt_map:
            continue

        actual = gt_map[uid]
        predicted_str = pred_map.get(uid, "")
        predicted = predicted_str.split()[:12]

        # Calculate AP@12
        if not actual:
            ap = 0.0
        else:
            actual_set = set(str(x).zfill(10) for x in actual)
            score = 0.0
            num_hits = 0.0
            already_predicted = set()

            for i, p in enumerate(predicted):
                p_str = str(p).zfill(10)
                if p_str in already_predicted:
                    continue
                already_predicted.add(p_str)

                if p_str in actual_set:
                    num_hits += 1.0
                    score += num_hits / (i + 1.0)

            ap = score / min(len(actual), 12)

        scores.append(ap)

        # Get Features
        # Age (fill nan with -1 or mean, here -1 as used in processing)
        age = age_map.get(uid, -1)
        if pd.isna(age):
            age = -1
        ages.append(age)

        # History Length
        hist_lens.append(user_hist_counts.get(uid, 0))

    # Calculate Correlations
    if len(scores) > 1:
        corr_age, _ = stats.pearsonr(scores, ages)
        corr_hist, _ = stats.pearsonr(scores, hist_lens)

        print(
            f"Correlation (Error vs Age): {-corr_age:.4f} (Negative corr means higher age -> lower score)"
        )
        print(
            f"Correlation (Error vs History Length): {-corr_hist:.4f} (Negative corr means more history -> lower score)"
        )

        # Interpretation for logging
        print(f"Direct Correlation (Score vs Age): {corr_age:.4f}")
        print(f"Direct Correlation (Score vs History Length): {corr_hist:.4f}")
    else:
        print("Insufficient data for failure analysis.")


def main():
    set_seed(Config.SEED)

    print("Starting Pipeline...")

    # ==========================================
    # 1. Validation Phase
    # ==========================================
    print("\n[Phase 1] Validation")
    val_recommender = SMDCRecommender()

    # Fit on training split (T-5w to T-1w)
    # This loads data, builds U, S_hybrid, and trend vectors
    val_recommender.fit(validate=True, load_cached_data=True)

    # Generate Predictions on Validation Set (Week T)
    # val_recommender.test_df contains the transactions for the validation week
    # We need unique users to predict for
    val_test_users = val_recommender.test_df[["customer_id"]].drop_duplicates()
    print(f"Predicting for {len(val_test_users)} validation users...")

    val_preds = val_recommender.predict(val_test_users)

    # Prepare Ground Truth
    # Group by customer_id -> list of article_ids
    gt_series = val_recommender.test_df.groupby("customer_id")["article_id"].apply(list)
    gt_dict = gt_series.to_dict()

    # Prepare Predictions Dictionary
    # val_preds is a list of strings aligned with val_test_users
    pred_dict = dict(zip(val_test_users["customer_id"], val_preds))

    # Calculate Metric
    map_score = calculate_map12(gt_dict, pred_dict)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {map_score}")

    # ==========================================
    # 2. Failure Analysis
    # ==========================================
    failure_analysis(val_recommender, val_preds, gt_series)

    # Clean up validation objects to free memory
    del val_recommender
    del val_preds
    del gt_series
    del gt_dict
    import gc

    gc.collect()

    # ==========================================
    # 3. Submission Phase
    # ==========================================
    THRESHOLD = 0.0265060791

    if map_score > THRESHOLD:
        print(f"\n[Phase 2] Submission (Metric {map_score:.5f} > {THRESHOLD})")

        # Re-initialize for full training
        full_recommender = SMDCRecommender()

        # Fit on Full Dataset (T-5w to T)
        # validate=False changes the data loading logic to include the validation week in training
        full_recommender.fit(validate=False, load_cached_data=True)

        # Generate Submission
        # This uses the sample_submission.csv users defined in load_data(validate=False)
        full_recommender.generate_submission()

    else:
        print(f"\n[Phase 2] Submission Skipped (Metric {map_score:.5f} <= {THRESHOLD})")


if __name__ == "__main__":
    main()
