import pandas as pd
import numpy as np
import os
import gc
from pathlib import Path
from tqdm.auto import tqdm

from library import config
from library import utils
from library import data_loader
from library import ranker_prep
from library import ranker_model


# ==========================================
# METRIC IMPLEMENTATION
# ==========================================
def apk(actual, predicted, k=12):
    """
    Computes the average precision at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=12):
    """
    Computes the mean average precision at k.
    actual: list of lists of ground truth items
    predicted: list of lists of predicted items
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def get_ground_truth(split="val", days=7):
    """
    Extracts the ground truth (actual purchases) for the target period of the split.
    """
    df = data_loader.load_transactions(split=split, load_cached_data=True)
    if "t_dat" not in df.columns:
        df["t_dat"] = pd.to_datetime(df["t_dat"])

    max_date = df["t_dat"].max()
    cutoff_date = max_date - pd.Timedelta(days=days)

    target_df = df[df["t_dat"] >= cutoff_date].copy()

    # Group by customer and list article_ids
    # Convert to set for faster lookup in apk, but apk expects list for order?
    # apk implementation above iterates predicted. 'if p in actual' is faster if actual is set.
    # But for the outer list, we need alignment.

    # We return a dictionary for easy mapping: customer_id -> set(article_ids)
    gt_df = target_df.groupby("customer_id")["article_id"].agg(set).reset_index()
    return dict(zip(gt_df["customer_id"], gt_df["article_id"]))


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    utils.seed_everything(config.SEED)

    print("Starting End-to-End Orchestration...")

    # ---------------------------------------------------------
    # 1. Dataset Construction
    # ---------------------------------------------------------
    print("\n[Step 1/5] Constructing Ranker Datasets...")
    dataset_builder = ranker_prep.RankerDatasetBuilder()

    # Build Train Set (Candidates from Train History -> Train Target)
    train_df = dataset_builder.build_ranker_train_set(load_cached_data=True)

    # Build Val Set (Candidates from Val History -> Val Target)
    val_df = dataset_builder.build_ranker_val_set(load_cached_data=True)

    # ---------------------------------------------------------
    # 2. Model Training
    # ---------------------------------------------------------
    print("\n[Step 2/5] Training Ranker...")
    ranker = ranker_model.LGBMRankerWrapper()
    ranker.fit(train_df, val_df)

    # Free up training memory
    del train_df
    gc.collect()

    # ---------------------------------------------------------
    # 3. Validation & Metrics
    # ---------------------------------------------------------
    print("\n[Step 3/5] Validating Model...")

    # Predict scores for validation set
    val_scores = ranker.predict(val_df)
    val_df["predicted_score"] = val_scores

    # Select Top 12 Predictions per User
    print("Selecting top 12 candidates per user...")
    # Sort by user and score
    val_df_sorted = val_df.sort_values(
        ["customer_id", "predicted_score"], ascending=[True, False]
    )
    top_preds = val_df_sorted.groupby("customer_id").head(12)

    # Create dictionary: customer_id -> list of predicted article_ids
    preds_map = top_preds.groupby("customer_id")["article_id"].apply(list).to_dict()

    # Get Ground Truth
    print("Loading Ground Truth...")
    gt_map = get_ground_truth(split="val", days=7)

    # Align predictions and ground truth
    # Note: We only score users who are in the ground truth (active in target week)
    # val_df contains candidates for users active in target week (handled by dataset builder)

    common_users = set(gt_map.keys()).intersection(set(preds_map.keys()))

    actuals = []
    predictions = []

    for uid in common_users:
        actuals.append(gt_map[uid])
        predictions.append(preds_map[uid])

    # Handle users in GT but with no predictions (e.g. retriever found 0 candidates? Unlikely with fallback)
    # If they are missing from preds_map, it counts as empty prediction.
    missing_users = set(gt_map.keys()) - set(preds_map.keys())
    for uid in missing_users:
        actuals.append(gt_map[uid])
        predictions.append([])

    # Compute MAP@12
    val_metric = mapk(actuals, predictions, k=12)
    print(f"Final Validation Metric: {val_metric:.16f}")

    # ---------------------------------------------------------
    # 4. Failure Analysis
    # ---------------------------------------------------------
    print("\n[Step 4/5] Performing Failure Analysis...")

    # Calculate AP per user to define "Error"
    user_aps = []
    user_ids = []

    # Re-iterate to capture per-user AP
    # We iterate over common_users + missing_users
    all_eval_users = list(common_users) + list(missing_users)

    for uid in all_eval_users:
        act = gt_map[uid]
        pred = preds_map.get(uid, [])
        ap = apk(act, pred, k=12)
        user_aps.append(ap)
        user_ids.append(uid)

    error_df = pd.DataFrame(
        {"customer_id": user_ids, "ap": user_aps, "error": 1.0 - np.array(user_aps)}
    )

    # Merge with Customer Metadata
    customers = data_loader.load_customers(load_cached_data=True)

    # Preprocess customers for correlation (encode categoricals)
    cust_features = [
        "age",
        "FN",
        "Active",
        "club_member_status",
        "fashion_news_frequency",
    ]

    # Handle simple encoding for correlation
    customers_encoded = customers.copy()
    # Fill NaNs
    customers_encoded["age"] = customers_encoded["age"].fillna(
        customers_encoded["age"].median()
    )
    customers_encoded["FN"] = customers_encoded["FN"].fillna(0)
    customers_encoded["Active"] = customers_encoded["Active"].fillna(0)

    for col in ["club_member_status", "fashion_news_frequency"]:
        if customers_encoded[col].dtype == "object":
            customers_encoded[col] = pd.factorize(customers_encoded[col])[0]

    analysis_df = error_df.merge(
        customers_encoded[["customer_id"] + cust_features], on="customer_id", how="left"
    )

    # Compute Correlation
    print("Correlation between Error (1-AP) and User Features:")
    corr_matrix = analysis_df[["error"] + cust_features].corr()
    print(corr_matrix["error"].sort_values(ascending=False))

    # ---------------------------------------------------------
    # 5. Submission
    # ---------------------------------------------------------
    threshold = 0.026059042

    if val_metric > threshold:
        print(
            f"\n[Step 5/5] Metric ({val_metric:.6f}) > Threshold ({threshold}). Generating Submission..."
        )

        # Build Inference Set (Retriever fit on Full History)
        test_df = dataset_builder.build_inference_set(load_cached_data=True)

        # Generate Submission
        sub_path = config.SUBMISSION_DIR / "submission.csv"
        ranker.generate_submission(test_df, sub_path)
    else:
        print(
            f"\n[Step 5/5] Metric ({val_metric:.6f}) <= Threshold ({threshold}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
