import sys
import os
import gc
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.data_utils import build_user_history_vectors
from library.hybrid_recommender import HybridRecommender


# Set seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)


def map_at_k(actual, predicted, k=12):
    """
    Computes Mean Average Precision @ k
    actual: list of sets of ground truth items
    predicted: list of lists of predicted items
    """
    scores = []
    for truth, preds in zip(actual, predicted):
        if not truth:
            continue

        # Ensure preds are unique and truncated to k
        preds = preds[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(preds):
            if p in truth:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        # MAP@12 definition: 1/min(m, 12) * sum(P(k) * rel(k))
        denominator = min(len(truth), k)
        scores.append(score / denominator)

    return np.mean(scores) if scores else 0.0


def run_validation(recommender):
    print("Loading validation data...")
    val_df = pd.read_csv(
        Config.PATH_VAL,
        dtype={"article_id": "int32", "price": "float32", "sales_channel_id": "int8"},
    )
    val_df["t_dat"] = pd.to_datetime(val_df["t_dat"])

    # Time split for validation: Last 7 days is target, previous 5 weeks is input
    max_date = val_df["t_dat"].max()
    split_date = max_date - pd.Timedelta(days=7)
    start_date = split_date - pd.Timedelta(weeks=Config.TRAIN_WEEKS)

    print(
        f"Splitting validation data. Input: {start_date} to {split_date}. Target: > {split_date}"
    )

    val_target = val_df[val_df["t_dat"] > split_date].copy()
    val_input = val_df[
        (val_df["t_dat"] <= split_date) & (val_df["t_dat"] > start_date)
    ].copy()

    target_users = val_target["customer_id"].unique()
    print(f"Total validation target users: {len(target_users)}")

    # --- Final Validation on ALL Target Users ---
    print("Running final validation...")

    # Build history for ALL target users
    # We must rebuild U_hist using the validation input split to avoid leakage
    val_input_all = val_input[val_input["customer_id"].isin(target_users)]

    # Temporarily swap U_hist in recommender
    original_U_hist = recommender.U_hist

    U_hist_all = build_user_history_vectors(
        val_input_all, recommender.mapper, load_cached_data=False
    )
    recommender.U_hist = U_hist_all

    all_indices = recommender.mapper.map_users(pd.Series(target_users)).values
    gt_dict_all = val_target.groupby("customer_id")["article_id"].apply(set).to_dict()

    batch_size = 2000
    user_aps = []
    user_hist_lens = []

    for start in range(0, len(target_users), batch_size):
        end = min(start + batch_size, len(target_users))
        batch_indices = all_indices[start:end]
        batch_users = target_users[start:end]

        scores = recommender.predict_scores(batch_indices)

        top_indices = np.argpartition(scores, -12, axis=1)[:, -12:]
        rows = np.arange(len(scores))[:, None]
        top_vals = scores[rows, top_indices]
        sort_inds = np.argsort(-top_vals, axis=1)
        final_indices = top_indices[rows, sort_inds]

        for i, u in enumerate(batch_users):
            truth = gt_dict_all.get(u, set())
            if not truth:
                continue

            pred_ids = recommender.mapper.get_items_from_indices(final_indices[i])

            # Compute AP
            ap = 0.0
            hits = 0.0
            for k, p in enumerate(pred_ids):
                if p in truth:
                    hits += 1.0
                    ap += hits / (k + 1.0)
            ap /= min(len(truth), 12)
            user_aps.append(ap)

            # History Length
            hist_len = U_hist_all[batch_indices[i]].getnnz()
            user_hist_lens.append(hist_len)

    # Restore original U_hist
    recommender.U_hist = original_U_hist

    final_map = np.mean(user_aps)
    print(f"Final Validation Metric: {final_map:.10f}")

    return final_map, user_aps, user_hist_lens


def analyze_failures(user_aps, user_hist_lens):
    print("\n--- Failure Analysis ---")
    df_fail = pd.DataFrame({"AP": user_aps, "HistoryLength": user_hist_lens})

    corr = df_fail.corr().iloc[0, 1]
    print(f"Correlation between History Length and AP: {corr:.4f}")

    df_fail["HistBin"] = pd.cut(
        df_fail["HistoryLength"],
        bins=[-1, 0, 5, 10, 50, 1000],
        labels=["0", "1-5", "6-10", "11-50", "50+"],
    )
    print("Average AP by History Length:")
    print(df_fail.groupby("HistBin")["AP"].mean())


def main():
    # 1. Initialize Recommender (Loads Train Data & Builds Matrices)
    rec = HybridRecommender(load_cached_data=True)

    # 2. Validation
    val_metric, user_aps, user_hist_lens = run_validation(rec)

    # 3. Failure Analysis
    analyze_failures(user_aps, user_hist_lens)

    # 4. Submission
    THRESHOLD = 0.0265060791
    if val_metric > THRESHOLD:
        print(
            f"\nValidation metric {val_metric:.6f} > {THRESHOLD}. Generating submission..."
        )

        # Prepare Full History (Train + Val)
        print("Re-building user history with full dataset (Train + Val)...")
        train_df = pd.read_csv(
            Config.PATH_TRAIN, dtype={"article_id": "int32", "t_dat": "object"}
        )
        val_df = pd.read_csv(
            Config.PATH_VAL, dtype={"article_id": "int32", "t_dat": "object"}
        )

        full_df = pd.concat([train_df, val_df], ignore_index=True)
        full_df["t_dat"] = pd.to_datetime(full_df["t_dat"])

        # Filter last 5 weeks of the FULL dataset
        max_date = full_df["t_dat"].max()
        cutoff = max_date - pd.Timedelta(weeks=Config.TRAIN_WEEKS)
        full_df = full_df[full_df["t_dat"] > cutoff]

        # Build vector
        full_U_hist = build_user_history_vectors(
            full_df, rec.mapper, load_cached_data=False
        )

        # Update recommender
        rec.U_hist = full_U_hist

        # Generate
        rec.generate_submission()

    else:
        print(
            f"\nValidation metric {val_metric:.6f} <= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
