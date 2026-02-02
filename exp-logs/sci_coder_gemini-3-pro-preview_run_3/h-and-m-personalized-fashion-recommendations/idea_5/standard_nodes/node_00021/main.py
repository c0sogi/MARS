import os
import sys
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from datetime import timedelta
import lightgbm as lgb
import torch

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library import config, data_loader, ranker, retrieval, feature_engineering

# Set Seeds
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)


def calculate_ap_at_12(predictions, ground_truth):
    """
    Calculates Average Precision @ 12 for a single user.
    predictions: list of predicted article_ids (ordered)
    ground_truth: set of actual article_ids
    """
    if not ground_truth:
        return 0.0

    score = 0.0
    num_hits = 0

    # We only care about the first 12 predictions
    preds = predictions[:12]

    for i, p in enumerate(preds):
        if p in ground_truth:
            num_hits += 1
            score += num_hits / (i + 1.0)

    return score / min(len(ground_truth), 12)


def validate_model(ranker_instance):
    """
    Performs validation on the hold-out validation set.
    Returns:
        map_score (float): MAP@12 score
        user_metrics (pd.DataFrame): DataFrame containing AP and features per user for failure analysis
    """
    print("\n" + "=" * 40)
    print("STARTING VALIDATION")
    print("=" * 40)

    # 1. Load Validation Data
    val_df = data_loader.load_transactions("val", load_cached_data=True)

    # 2. Split into Context (History) and Ground Truth (Target)
    # Target is the last 7 days of the validation set
    max_date = val_df["t_dat"].max()
    split_date = max_date - timedelta(days=7)

    print(f"Splitting validation data at {split_date}...")
    history_df = val_df[val_df["t_dat"] <= split_date].copy()
    target_df = val_df[val_df["t_dat"] > split_date].copy()

    # Ground Truth Map: customer_id -> set(article_ids)
    ground_truth = target_df.groupby("customer_id")["article_id"].apply(set).to_dict()
    active_users = np.array(list(ground_truth.keys()))
    n_users = len(active_users)
    print(f"Validating on {n_users} active users...")

    # 3. Prepare for Inference
    # We reuse the retriever and feature lookups from the ranker instance
    retriever = ranker_instance.retriever
    model = ranker_instance.model

    # Pre-calculate time decay for history
    max_hist_date = history_df["t_dat"].max()
    history_df["days_diff"] = (max_hist_date - history_df["t_dat"]).dt.days
    history_df["weight"] = np.exp(-history_df["days_diff"] / config.TIME_DECAY_DAYS)

    # Map IDs for sparse matrix construction
    # We filter history to only include items known to the retriever
    history_df = history_df[
        history_df["article_id"].isin(retriever.art_id_to_idx)
    ].copy()
    history_df["item_idx"] = (
        history_df["article_id"].map(retriever.art_id_to_idx).astype(np.int32)
    )

    # 4. Batch Inference
    batch_size = 5000
    ap_scores = []
    user_features_list = []

    # Global popularity fallback
    global_pop_ids = retriever.global_popularity[:12].tolist()

    for i in range(0, n_users, batch_size):
        batch_cids = active_users[i : i + batch_size]

        # --- A. Construct User Vectors ---
        batch_hist = history_df[history_df["customer_id"].isin(batch_cids)].copy()

        # Local map for user rows
        cust_map = {cid: idx for idx, cid in enumerate(batch_cids)}
        batch_hist["user_idx"] = (
            batch_hist["customer_id"].map(cust_map).astype(np.int32)
        )

        if len(batch_hist) > 0:
            row_idx = batch_hist["user_idx"].values
            col_idx = batch_hist["item_idx"].values
            weights = batch_hist["weight"].values

            u_weighted = sp.coo_matrix(
                (weights, (row_idx, col_idx)),
                shape=(len(batch_cids), retriever.n_items),
            ).tocsr()

            u_raw = sp.coo_matrix(
                (np.ones(len(weights)), (row_idx, col_idx)),
                shape=(len(batch_cids), retriever.n_items),
            ).tocsr()
        else:
            u_weighted = sp.csr_matrix((len(batch_cids), retriever.n_items))
            u_raw = sp.csr_matrix((len(batch_cids), retriever.n_items))

        # --- B. Retrieval ---
        s_seq = u_weighted.dot(retriever.T_seq)
        s_vis = u_weighted.dot(retriever.T_vis)
        scores_sparse = (
            s_seq + (config.LAMBDA_VISUAL * s_vis) + (config.ALPHA_HISTORY * u_raw)
        )
        scores_dense = scores_sparse.toarray()

        # --- C. Top-K Extraction ---
        k = config.RETRIEVAL_TOP_K
        if scores_dense.shape[1] < k:
            k = scores_dense.shape[1]

        top_k_idx = np.argpartition(scores_dense, -k, axis=1)[:, -k:]
        rows = np.arange(len(batch_cids))[:, None]
        top_k_scores = scores_dense[rows, top_k_idx]
        sort_order = np.argsort(top_k_scores, axis=1)[:, ::-1]

        sorted_indices = top_k_idx[rows, sort_order]
        sorted_scores = top_k_scores[rows, sort_order]

        # Detect cold users
        max_scores = sorted_scores[:, 0]
        cold_user_mask = max_scores <= 0

        # --- D. Feature Construction ---
        cust_ids_repeated = np.repeat(batch_cids, k)
        flat_indices = sorted_indices.flatten()
        flat_scores = sorted_scores.flatten()
        flat_ranks = np.tile(np.arange(k), len(batch_cids))
        article_ids = retriever.art_idx_to_id[flat_indices]

        batch_df = pd.DataFrame(
            {
                "customer_id": cust_ids_repeated,
                "article_id": article_ids,
                "retrieval_score": flat_scores,
                "retrieval_rank": flat_ranks,
            }
        )

        # Merge Metadata
        batch_df = batch_df.merge(
            ranker_instance.cust_features, on="customer_id", how="left"
        )
        batch_df = batch_df.merge(
            ranker_instance.art_features, on="article_id", how="left"
        )
        batch_df["global_popularity"] = (
            batch_df["article_id"].map(ranker_instance.pop_map).fillna(0.0)
        )

        # --- E. Scoring ---
        feature_cols = [
            "retrieval_score",
            "retrieval_rank",
            "age",
            "club_member_status",
            "fashion_news_frequency",
            "global_popularity",
            "product_type_no",
            "graphical_appearance_no",
            "colour_group_code",
            "perceived_colour_value_id",
            "department_no",
            "index_group_no",
            "section_no",
            "garment_group_no",
        ]

        # Cast categoricals
        for c in ["club_member_status", "fashion_news_frequency"]:
            batch_df[c] = batch_df[c].astype("category")

        batch_df["pred_score"] = model.predict(batch_df[feature_cols])

        # --- F. Ranking & Metric Calculation ---
        # Sort by prediction
        batch_df.sort_values(
            ["customer_id", "pred_score"], ascending=[True, False], inplace=True
        )

        # Group by customer
        # We process user by user in this batch for metric calculation
        # To speed up, we can use groupby

        top_preds = batch_df.groupby("customer_id").head(12)
        pred_map = top_preds.groupby("customer_id")["article_id"].apply(list).to_dict()

        # Collect features for failure analysis (Age and History Length)
        # History length is number of items in batch_hist for that user
        hist_counts = batch_hist["customer_id"].value_counts().to_dict()

        for j, cid in enumerate(batch_cids):
            # Predictions
            if cold_user_mask[j]:
                preds = global_pop_ids
            else:
                preds = pred_map.get(
                    cid, global_pop_ids
                )  # Fallback if groupby missed it
                # If less than 12, append global pop
                if len(preds) < 12:
                    preds.extend(global_pop_ids[: 12 - len(preds)])

            # Ground Truth
            truth = ground_truth[cid]

            # Metric
            ap = calculate_ap_at_12(preds, truth)
            ap_scores.append(ap)

            # Features for analysis
            # Get age from batch_df (first row for this customer)
            # or lookup in cust_features
            age = ranker_instance.cust_features.loc[
                ranker_instance.cust_features["customer_id"] == cid, "age"
            ].values
            age = age[0] if len(age) > 0 else -1

            hist_len = hist_counts.get(cid, 0)

            user_features_list.append(
                {"customer_id": cid, "ap": ap, "age": age, "history_len": hist_len}
            )

        if (i // batch_size) % 5 == 0:
            print(f"Validated {min(i + batch_size, n_users)}/{n_users} users...")
            gc.collect()

    # Final Metric
    map_score = np.mean(ap_scores)
    user_metrics_df = pd.DataFrame(user_features_list)

    return map_score, user_metrics_df


def failure_analysis(user_metrics_df):
    """
    Analyzes correlation between Error (1 - AP) and features.
    """
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    user_metrics_df["error"] = 1.0 - user_metrics_df["ap"]

    # Correlation with Age
    # Filter valid ages
    valid_age = user_metrics_df[user_metrics_df["age"] > 0]
    if len(valid_age) > 0:
        corr_age = valid_age["error"].corr(valid_age["age"])
        print(f"Correlation (Error vs Age): {corr_age:.4f}")
    else:
        print("Correlation (Error vs Age): N/A (No valid ages)")

    # Correlation with History Length
    corr_hist = user_metrics_df["error"].corr(user_metrics_df["history_len"])
    print(f"Correlation (Error vs History Length): {corr_hist:.4f}")

    # Binning analysis
    print("\nError by History Length Bins:")
    user_metrics_df["hist_bin"] = pd.cut(
        user_metrics_df["history_len"],
        bins=[-1, 0, 5, 20, 100, 1000],
        labels=["0", "1-5", "6-20", "21-100", ">100"],
    )
    print(user_metrics_df.groupby("hist_bin", observed=True)["error"].mean())


def main():
    # 1. Train Model
    # This orchestrates data loading, graph building, and training
    print("Initializing and Training Ranker...")
    ranker_instance = ranker.Ranker()
    ranker_instance.train(load_cached_model=True)

    # 2. Validate
    map_score, user_metrics_df = validate_model(ranker_instance)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {map_score:.9f}")

    # 3. Failure Analysis
    failure_analysis(user_metrics_df)

    # 4. Submission
    THRESHOLD = 0.026059042
    if map_score > THRESHOLD:
        print(
            f"\nValidation score ({map_score:.6f}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        ranker_instance.predict()
    else:
        print(
            f"\nValidation score ({map_score:.6f}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
