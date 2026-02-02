import pandas as pd
import numpy as np
import os
import torch
import gc
import random
from collections import defaultdict
from library import (
    config,
    data_manager,
    visual_module,
    retrieval_engine,
    feature_engine,
    ranking_model,
)


# =============================================================================
# CONFIGURATION & SEEDING
# =============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(config.SEED)

# Limits for Fast Baseline
TRAIN_SAMPLE_USERS = 20000  # Users per window
VAL_DATE_SPLIT = "2020-09-16"
SUBMISSION_THRESHOLD = 0.026059042


# =============================================================================
# METRICS
# =============================================================================
def apk(actual, predicted, k=12):
    if not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def map_at_12(predictions_df, ground_truth_df):
    """
    Computes MAP@12.
    predictions_df: DataFrame with ['customer_id', 'prediction'] (space separated)
    ground_truth_df: DataFrame with ['customer_id', 'article_id']
    """
    # Group ground truth
    gt_dict = ground_truth_df.groupby("customer_id")["article_id"].apply(set).to_dict()

    # Parse predictions
    scores = []
    for _, row in predictions_df.iterrows():
        cust_id = row["customer_id"]
        if cust_id not in gt_dict:
            continue

        preds = [int(x) for x in str(row["prediction"]).split()]
        actual = gt_dict[cust_id]
        scores.append(apk(actual, preds, k=12))

    return np.mean(scores) if scores else 0.0


# =============================================================================
# PIPELINE ORCHESTRATION
# =============================================================================
def run():
    print("Initializing pipeline...")

    # 1. Load Data
    train_df, val_df, test_df = data_manager.load_metadata()

    # --- DYNAMIC DATE ADJUSTMENT ---
    # Detect actual max date to prevent empty windows/validation sets
    # Cite debug_lesson_6: Enforce Temporal Alignment
    max_train_date = train_df["t_dat"].max()
    max_val_date = val_df["t_dat"].max()
    real_max_date = max(max_train_date, max_val_date)

    print(f"Detected Max Date in Data: {real_max_date.date()}")

    # Update Config dynamically
    config.DATA_END_DATE = str(real_max_date.date())

    # Set Validation Split to the start of the last 7-day window
    val_start_date = real_max_date - pd.Timedelta(days=6)
    val_split_date_str = str(val_start_date.date())
    print(f"Adjusted VAL_DATE_SPLIT to: {val_split_date_str}")
    # -------------------------------

    # Initialize Modules
    retriever = retrieval_engine.DualViewRetriever()
    feat_gen = feature_engine.FeatureGenerator()
    ranker = ranking_model.Ranker()

    # 2. Build Static Visual Graph (Once)
    # This is time-independent
    visual_graph = visual_module.build_visual_graph(load_cached_data=True)

    # =========================================================================
    # STAGE 1: TRAINING DATA GENERATION (SLIDING WINDOW)
    # =========================================================================
    print("\n=== Stage 1: Generating Training Data ===")

    windows = data_manager.get_sliding_windows()
    # Use only first 2 windows for speed
    windows = windows[:2]

    full_train_data = []

    for i, window in enumerate(windows):
        print(f"\nProcessing Window {i}: History < {window['target_start'].date()}")

        # A. Temporal Split
        # We use train_df for training the ranker
        hist_df = data_manager.filter_transactions(
            train_df,
            pd.to_datetime("2000-01-01"),
            window["history_end"],
            inclusive_end=False,
        )
        target_df = data_manager.filter_transactions(
            train_df, window["target_start"], window["target_end"], inclusive_end=True
        )

        if len(target_df) == 0:
            print("No target data for this window. Skipping.")
            continue

        # B. Sample Users
        target_users = target_df["customer_id"].unique()
        if len(target_users) > TRAIN_SAMPLE_USERS:
            sampled_users = np.random.choice(
                target_users, TRAIN_SAMPLE_USERS, replace=False
            )
        else:
            sampled_users = target_users

        print(f"Sampled {len(sampled_users)} users for training.")

        # C. Build Sequential Graph (Strict Recency for this window)
        # We build graph on the history of this window
        seq_graph = retriever.build_sequential_graph(
            hist_df, cache_key=f"train_w{i}", load_cached_data=True
        )

        # D. Build User Vectors
        u_vecs = retriever.build_user_vectors(hist_df, sampled_users)

        # E. Retrieve Candidates
        candidates = retriever.retrieve(u_vecs, seq_graph, visual_graph, sampled_users)

        # F. Generate Labeled Features
        # Filter target_df to sampled users for labeling
        target_subset = target_df[target_df["customer_id"].isin(sampled_users)]
        features = feat_gen.generate_features(
            candidates,
            hist_df,
            target_subset,
            cache_key=f"train_w{i}",
            load_cached_data=True,
        )

        full_train_data.append(features)

        # Cleanup
        del hist_df, target_df, seq_graph, u_vecs, candidates, features
        gc.collect()

    # Combine
    if not full_train_data:
        raise ValueError("No training data generated!")

    train_concat = pd.concat(full_train_data, ignore_index=True)
    del full_train_data
    gc.collect()

    # =========================================================================
    # STAGE 2: RANKER TRAINING
    # =========================================================================
    print("\n=== Stage 2: Training Ranker ===")

    # Split into Train/Val for LightGBM early stopping
    # We split by user to avoid leakage
    unique_users = train_concat["customer_id"].unique()
    np.random.shuffle(unique_users)
    split_idx = int(len(unique_users) * 0.9)
    train_users = set(unique_users[:split_idx])

    lgb_train = train_concat[train_concat["customer_id"].isin(train_users)]
    lgb_val = train_concat[~train_concat["customer_id"].isin(train_users)]

    print(f"Train Rows: {len(lgb_train)}, Val Rows: {len(lgb_val)}")

    ranker.train(lgb_train, lgb_val)

    del train_concat, lgb_train, lgb_val
    gc.collect()

    # =========================================================================
    # STAGE 3: VALIDATION (HOLD-OUT SET)
    # =========================================================================
    print("\n=== Stage 3: Validation ===")

    # We validate on the 'val_df' users (hold-out group).
    # We simulate the task: Predict purchases in [2020-09-16, 2020-09-22]
    # History: All data < 2020-09-16 (from both train_df and val_df to simulate global knowledge)

    split_date = pd.to_datetime(VAL_DATE_SPLIT)

    # 1. Prepare Data
    # Global history for graph (Train + Val < Split)
    # Note: In a real scenario, we have access to all history up to the test point.
    global_hist_train = data_manager.filter_transactions(
        train_df, pd.to_datetime("2000-01-01"), split_date, inclusive_end=False
    )
    global_hist_val = data_manager.filter_transactions(
        val_df, pd.to_datetime("2000-01-01"), split_date, inclusive_end=False
    )
    global_hist = pd.concat([global_hist_train, global_hist_val])

    # Validation Target (Val users >= Split)
    val_target = data_manager.filter_transactions(
        val_df, split_date, pd.to_datetime("2099-12-31"), inclusive_end=True
    )
    val_users = val_target["customer_id"].unique()

    print(f"Validation Users: {len(val_users)}")

    # 2. Build Inference Graph
    seq_graph_val = retriever.build_sequential_graph(
        global_hist, cache_key="val_inference", load_cached_data=True
    )

    # 3. Retrieve
    # User vectors built from their specific history (global_hist_val)
    u_vecs_val = retriever.build_user_vectors(global_hist_val, val_users)
    val_candidates = retriever.retrieve(
        u_vecs_val, seq_graph_val, visual_graph, val_users
    )

    # 4. Generate Features
    val_features = feat_gen.generate_features(
        val_candidates, global_hist, cache_key="val_inference", load_cached_data=True
    )

    # 5. Predict
    val_features["score"] = ranker.predict(val_features)

    # 6. Select Top 12
    val_features = val_features.sort_values(
        ["customer_id", "score"], ascending=[True, False]
    )
    val_preds = (
        val_features.groupby("customer_id")["article_id"]
        .apply(lambda x: list(x)[:12])
        .reset_index()
    )

    # Format for MAP calculation
    val_preds["prediction"] = val_preds["article_id"].apply(
        lambda x: " ".join(map(str, x))
    )

    # 7. Compute Metric
    map_score = map_at_12(val_preds, val_target)
    print(f"Final Validation Metric: {map_score:.9f}")

    # =========================================================================
    # STAGE 4: FAILURE ANALYSIS
    # =========================================================================
    print("\n=== Stage 4: Failure Analysis ===")

    # Calculate AP per user
    gt_dict = val_target.groupby("customer_id")["article_id"].apply(set).to_dict()
    user_aps = []
    user_ids = []

    for _, row in val_preds.iterrows():
        cid = row["customer_id"]
        if cid in gt_dict:
            preds = [int(x) for x in str(row["prediction"]).split()]
            actual = gt_dict[cid]
            ap = apk(actual, preds, k=12)
            user_aps.append(ap)
            user_ids.append(cid)

    analysis_df = pd.DataFrame({"customer_id": user_ids, "ap": user_aps})

    # Merge with user features (Age, History Length)
    # History length from global_hist_val
    hist_counts = global_hist_val["customer_id"].value_counts().reset_index()
    hist_counts.columns = ["customer_id", "hist_len"]

    # Age from customers.csv
    cust_meta = pd.read_csv(config.CUSTOMERS_CSV)

    analysis_df = analysis_df.merge(hist_counts, on="customer_id", how="left").fillna(0)
    analysis_df = analysis_df.merge(
        cust_meta[["customer_id", "age"]], on="customer_id", how="left"
    )
    analysis_df["age"] = analysis_df["age"].fillna(analysis_df["age"].median())

    # Correlations
    corr_len = analysis_df["ap"].corr(analysis_df["hist_len"])
    corr_age = analysis_df["ap"].corr(analysis_df["age"])

    print("Correlation of Error (Low AP) with Features:")
    print(f"  AP vs History Length: {corr_len:.4f}")
    print(f"  AP vs Age: {corr_age:.4f}")

    # =========================================================================
    # STAGE 5: SUBMISSION
    # =========================================================================
    if map_score > SUBMISSION_THRESHOLD:
        print("\n=== Stage 5: Generating Submission ===")

        # 1. Prepare Data (Full History)
        full_hist = pd.concat([train_df, val_df])
        test_users = test_df["customer_id"].unique()

        # 2. Build Final Graph
        seq_graph_test = retriever.build_sequential_graph(
            full_hist, cache_key="test_inference", load_cached_data=True
        )

        # 3. Retrieve
        u_vecs_test = retriever.build_user_vectors(full_hist, test_users)
        test_candidates = retriever.retrieve(
            u_vecs_test, seq_graph_test, visual_graph, test_users
        )

        # 4. Generate Features
        test_features = feat_gen.generate_features(
            test_candidates,
            full_hist,
            cache_key="test_inference",
            load_cached_data=True,
        )

        # 5. Predict & Save
        ranker.generate_and_save_submission(test_features, test_df)

    else:
        print(
            f"\nSkipping submission. Score {map_score:.6f} <= Threshold {SUBMISSION_THRESHOLD}"
        )


if __name__ == "__main__":
    run()
