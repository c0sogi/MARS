import numpy as np
import pandas as pd
import os
import random
import sys
from library import config, data_factory, graph_engine, stratified_inference, evaluation


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_failure_analysis(preds_df, val_df, train_df):
    """
    Analyzes model performance against user features.
    """
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    # 1. Calculate Per-User AP
    gt_grouped = val_df.groupby("customer_id")["article_id"].apply(list).to_dict()

    # Convert predictions to dict
    preds_dict = {}
    cust_ids = preds_df["customer_id"].values
    pred_strs = preds_df["prediction"].values

    for cid, pred_str in zip(cust_ids, pred_strs):
        if pd.isna(pred_str) or pred_str == "":
            preds_dict[cid] = []
        else:
            try:
                preds_dict[cid] = [int(x) for x in str(pred_str).split()]
            except ValueError:
                preds_dict[cid] = []

    user_scores = []
    users = []

    for cid, actual_items in gt_grouped.items():
        pred_items = preds_dict.get(cid, [])
        score = evaluation.apk(actual_items, pred_items, k=12)
        user_scores.append(score)
        users.append(cid)

    score_df = pd.DataFrame({"customer_id": users, "ap": user_scores})

    # 2. Extract User Features from Train
    # Feature 1: History Length (Count)
    # Feature 2: Recency (Min days_elapsed in train, lower is more recent)
    user_stats = (
        train_df.groupby("customer_id")
        .agg(history_len=("article_id", "count"), recency=("days_elapsed", "min"))
        .reset_index()
    )

    # Merge
    analysis_df = score_df.merge(user_stats, on="customer_id", how="left")

    # Fill NaNs for users in Val but not in Train (Cold Start in Train context)
    analysis_df["history_len"] = analysis_df["history_len"].fillna(0)
    # For recency, if missing, set to max (least recent)
    max_days = train_df["days_elapsed"].max() if not train_df.empty else 0
    analysis_df["recency"] = analysis_df["recency"].fillna(max_days)

    # 3. Compute Correlations
    corr_len = analysis_df["ap"].corr(analysis_df["history_len"])
    corr_rec = analysis_df["ap"].corr(analysis_df["recency"])

    print(f"Correlation (AP vs History Length): {corr_len:.4f}")
    print(f"Correlation (AP vs Recency [Days]): {corr_rec:.4f}")
    print(
        "(Note: Negative correlation with Recency is good -> Recent users have higher AP)"
    )
    print("-" * 40)


def main():
    # 1. Setup
    set_seed(config.RANDOM_SEED)

    print("Starting Runfile Execution...")

    # 2. Validation Pipeline
    # We replicate the steps from evaluation.validate to keep objects for analysis
    print("\n[Validation] Loading Data...")
    # Load the 80% train split metadata
    df = data_factory.load_and_preprocess(config.TRAIN_META_PATH, load_cached_data=True)

    # Split into Train (History) and Val (Ground Truth)
    # Val is last 7 days of this dataset
    train_df, val_df = data_factory.get_time_split(df, val_days=7)

    if len(val_df) == 0:
        print("Error: Validation set empty.")
        return

    # Build Artifacts on Train Split (No Leakage)
    print("\n[Validation] Building Graph Artifacts...")
    # Force recompute (load_cached_data=False) because this specific split isn't cached
    # The cache usually stores the full dataset artifacts
    user_map, item_map = graph_engine.get_mappings(train_df, load_cached_data=False)

    interaction_matrix = graph_engine.build_decayed_interaction_matrix(
        train_df, user_map, item_map, load_cached_data=False
    )

    similarity_matrix = graph_engine.compute_similarity_matrix(
        interaction_matrix, load_cached_data=False
    )

    # Fit Recommender
    print("\n[Validation] Fitting Model...")
    recommender = stratified_inference.TGSCRecommender(
        user_map, item_map, similarity_matrix
    )
    recommender.fit(train_df, load_cached_data=False)

    # Predict
    print("\n[Validation] Predicting...")
    val_customers = val_df["customer_id"].unique()
    preds_df = recommender.predict(val_customers)

    # Score
    print("\n[Validation] Scoring...")
    final_score = evaluation.calculate_map12(preds_df, val_df)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_score}")

    # 3. Failure Analysis
    run_failure_analysis(preds_df, val_df, train_df)

    # 4. Submission
    # Threshold from instructions
    THRESHOLD = 0.0265060791

    if final_score > THRESHOLD:
        print(
            f"\nValidation score ({final_score}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        # This function handles loading full data (Train+Val), retraining, and predicting on Test
        stratified_inference.generate_submission()
    else:
        print(
            f"\nValidation score ({final_score}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
