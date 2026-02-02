import pandas as pd
import numpy as np
import torch
import gc
import logging
from pathlib import Path
from typing import List, Dict, Tuple

# Import library modules
from library.config import Paths, DATA_CONFIG, SEED, GCN_PARAMS, CANDIDATE_CONFIG
from library.utils import setup_logger, seed_everything, reduce_mem_usage
from library.data_loader import (
    load_raw_data,
    create_time_split,
    prepare_graph_data,
    get_recent_popular_items,
)
from library.graph_engine import train_graph_embeddings
from library.retrieval import CooccurrenceMatrix, CandidateGenerator
from library.features import FeatureEngineer
from library.ranker import Ranker

# Setup Logger
logger = setup_logger("pipeline")


def calculate_map12(predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    """
    Calculates Mean Average Precision @ 12.
    predictions: DataFrame with ['customer_id', 'prediction'] (space separated string)
    ground_truth: DataFrame with ['customer_id', 'article_id'] (actual purchases)
    """
    # Group ground truth into list of articles
    gt_grouped = ground_truth.groupby("customer_id")["article_id"].apply(list).to_dict()

    # Parse predictions
    preds_map = dict(zip(predictions["customer_id"], predictions["prediction"]))

    scores = []
    for cust_id, pred_str in preds_map.items():
        if cust_id not in gt_grouped:
            continue

        actual = gt_grouped[cust_id]
        predicted = pred_str.split()[:12]

        score = 0.0
        num_hits = 0

        for i, p in enumerate(predicted):
            if p in actual:
                num_hits += 1
                score += num_hits / (i + 1.0)

        scores.append(score / min(len(actual), 12))

    return np.mean(scores) if scores else 0.0


def analyze_failures(
    scored_df: pd.DataFrame, ground_truth: pd.DataFrame, features_df: pd.DataFrame
):
    """
    Performs failure analysis by correlating Average Precision with features.
    """
    logger.info("Performing Failure Analysis...")

    # 1. Calculate AP per user
    gt_grouped = ground_truth.groupby("customer_id")["article_id"].apply(set).to_dict()

    # Sort by score
    scored_df = scored_df.sort_values(["customer_id", "score"], ascending=[True, False])
    top_k = scored_df.groupby("customer_id").head(12)

    user_aps = []
    user_ids = []

    for cust_id, group in top_k.groupby("customer_id"):
        if cust_id not in gt_grouped:
            continue

        actual = gt_grouped[cust_id]
        predicted = group["article_id"].tolist()

        score = 0.0
        num_hits = 0
        for i, p in enumerate(predicted):
            if p in actual:
                num_hits += 1
                score += num_hits / (i + 1.0)

        ap = score / min(len(actual), 12)
        user_aps.append(ap)
        user_ids.append(cust_id)

    ap_df = pd.DataFrame({"customer_id": user_ids, "ap": user_aps})
    ap_df["error"] = 1.0 - ap_df["ap"]

    # 2. Merge with aggregated features
    # We take the mean feature value of the top 12 candidates for each user
    feat_cols = [
        "sales_velocity",
        "user_dept_ratio",
        "graph_dot_product",
        "last_item_graph_similarity",
    ]
    # Ensure cols exist
    feat_cols = [c for c in feat_cols if c in features_df.columns]

    user_feats = features_df[features_df["customer_id"].isin(user_ids)][
        ["customer_id"] + feat_cols
    ]
    user_feats_agg = user_feats.groupby("customer_id")[feat_cols].mean().reset_index()

    analysis_df = pd.merge(ap_df, user_feats_agg, on="customer_id")

    # 3. Correlation
    logger.info("Correlation between Error (1-AP) and Mean Candidate Features:")
    corrs = analysis_df.drop(columns=["customer_id"]).corr()["error"].drop("error")
    print(corrs)

    return corrs


def run_pipeline():
    seed_everything(SEED)

    # ==========================================
    # 1. Data Loading & Splitting
    # ==========================================
    logger.info("Step 1: Loading Data...")
    train_meta, val_meta, test_meta = load_raw_data()

    # Combine train and val metadata to get full history for splitting
    full_df = pd.concat([train_meta, val_meta], ignore_index=True)
    full_df = reduce_mem_usage(full_df)

    # Clean up
    del train_meta, val_meta
    gc.collect()

    # Create Time Split (Last 7 days as validation)
    train_split, val_split = create_time_split(
        full_df, val_days=DATA_CONFIG["val_days"]
    )

    # Filter val_split to only customers who have history in train_split (Warm start)
    # Cold start is handled by popularity, but for strict evaluation we focus on retrieval
    train_users = set(train_split["customer_id"].unique())
    val_split = val_split[val_split["customer_id"].isin(train_users)].copy()

    logger.info(f"Train Split: {len(train_split)}, Val Split (Warm): {len(val_split)}")

    # ==========================================
    # 2. Ranker Training Set Generation (Sliding Window)
    # ==========================================
    logger.info("Step 2: Generating Ranker Training Data (Window 1)...")

    # Create a local split within train_split to simulate the task
    train_local, val_local = create_time_split(
        train_split, val_days=DATA_CONFIG["val_days"]
    )

    # Sample users for speed (Fast Baseline)
    sample_users = val_local["customer_id"].unique()
    if len(sample_users) > 10000:
        np.random.shuffle(sample_users)
        sample_users = sample_users[:10000]

    # 2a. Train Components on train_local
    # Graph
    edge_index_local, u_map_local, i_map_local = prepare_graph_data(
        train_local, load_cached=False
    )
    # Reduce epochs for speed in this sub-step
    local_gcn_params = GCN_PARAMS.copy()
    local_gcn_params["epochs"] = 5
    u_emb_local, i_emb_local = train_graph_embeddings(
        edge_index_local, len(u_map_local), len(i_map_local), params=local_gcn_params
    )

    # Cooc
    cooc_local = CooccurrenceMatrix()
    cooc_local.fit(train_local, load_cached=False)

    # Popularity
    pop_local = get_recent_popular_items(train_local)

    # 2b. Generate Candidates
    gen_local = CandidateGenerator(
        u_emb_local, i_emb_local, u_map_local, i_map_local, cooc_local
    )
    candidates_train = gen_local.generate(sample_users, train_local, pop_local)

    # 2c. Labeling
    logger.info("Labeling Ranker Train Data...")
    ground_truth_local = val_local[val_local["customer_id"].isin(sample_users)]
    gt_set = set(
        zip(ground_truth_local["customer_id"], ground_truth_local["article_id"])
    )

    candidates_train["label"] = candidates_train.apply(
        lambda x: 1 if (x["customer_id"], x["article_id"]) in gt_set else 0, axis=1
    )

    # 2d. Features
    fe = FeatureEngineer()
    # Need articles df for affinity
    articles_df = pd.read_csv(Paths.INPUT_DIR / "articles.csv")

    features_train = fe.generate_features(
        candidates_train,
        train_local,
        articles_df,
        u_emb_local,
        i_emb_local,
        u_map_local,
        i_map_local,
        load_cached=False,
        suffix="ranker_train",
    )

    # Cleanup
    del (
        train_local,
        val_local,
        edge_index_local,
        u_emb_local,
        i_emb_local,
        cooc_local,
        gen_local,
    )
    gc.collect()

    # ==========================================
    # 3. Ranker Validation Set Generation (Window 2)
    # ==========================================
    logger.info("Step 3: Generating Ranker Validation Data (Real Val)...")

    # Train components on full train_split
    # Graph
    edge_index, u_map, i_map = prepare_graph_data(train_split, load_cached=True)
    # Full epochs
    u_emb, i_emb = train_graph_embeddings(
        edge_index, len(u_map), len(i_map), params=GCN_PARAMS
    )

    # Cooc
    cooc = CooccurrenceMatrix()
    cooc.fit(train_split, load_cached=True)

    # Popularity
    pop_items = get_recent_popular_items(train_split)

    # Generate for ALL val_split users
    val_users = val_split["customer_id"].unique()
    gen = CandidateGenerator(u_emb, i_emb, u_map, i_map, cooc)
    candidates_val = gen.generate(val_users, train_split, pop_items)

    # Labeling
    gt_set_val = set(zip(val_split["customer_id"], val_split["article_id"]))
    candidates_val["label"] = candidates_val.apply(
        lambda x: 1 if (x["customer_id"], x["article_id"]) in gt_set_val else 0, axis=1
    )

    # Features
    features_val = fe.generate_features(
        candidates_val,
        train_split,
        articles_df,
        u_emb,
        i_emb,
        u_map,
        i_map,
        load_cached=True,
        suffix="ranker_val",
    )

    # ==========================================
    # 4. Train Ranker
    # ==========================================
    logger.info("Step 4: Training Ranker...")

    feature_cols = [
        c
        for c in features_train.columns
        if c not in ["customer_id", "article_id", "label"]
    ]
    logger.info(f"Features: {feature_cols}")

    ranker = Ranker()
    ranker.train(features_train, features_val, feature_cols, load_cached_model=False)

    # ==========================================
    # 5. Validation & Failure Analysis
    # ==========================================
    logger.info("Step 5: Evaluating...")

    scored_val = ranker.predict(features_val, feature_cols)

    # Generate predictions string
    submission_val = ranker.generate_submission(
        scored_val, output_path=Paths.WORKING_DIR / "val_preds.csv"
    )

    # Calculate Metric
    map_score = calculate_map12(submission_val, val_split)
    print(f"Final Validation Metric: {map_score:.16f}")

    # Failure Analysis
    analyze_failures(scored_val, val_split, features_val)

    # ==========================================
    # 6. Submission
    # ==========================================
    threshold = 0.0306342353457529
    if map_score > threshold:
        logger.info("Step 6: Generating Submission...")

        # Ideally we retrain GCN/Cooc on full_df (train_split + val_split)
        # For this baseline, we will use the models trained on train_split
        # but we must generate candidates for the TEST users.

        test_users = test_meta["customer_id"].unique()

        # Generate candidates using the models from Step 3 (trained on train_split)
        # Note: This ignores the last 7 days of data for model training, but uses it for feature calc if we pass full_df
        # Let's pass full_df to feature calculation to capture latest trends

        logger.info(f"Generating candidates for {len(test_users)} test users...")

        # Batch processing for test users to avoid OOM
        batch_size = 50000
        all_scored_test = []

        # We reuse the generator 'gen' which holds models trained on 'train_split'
        # Ideally we would retrain on 'full_df', but time is tight.

        for i in range(0, len(test_users), batch_size):
            batch_users = test_users[i : i + batch_size]

            # Generate
            cand_batch = gen.generate(batch_users, full_df, pop_items)

            if cand_batch.empty:
                continue

            # Features (use full_df for history)
            feat_batch = fe.generate_features(
                cand_batch,
                full_df,
                articles_df,
                u_emb,
                i_emb,
                u_map,
                i_map,
                load_cached=False,
                suffix=f"test_{i}",
            )

            # Predict
            scored_batch = ranker.predict(feat_batch, feature_cols)

            # Minimize storage
            scored_batch = scored_batch[["customer_id", "article_id", "score"]]
            all_scored_test.append(scored_batch)

            gc.collect()

        if all_scored_test:
            full_test_scored = pd.concat(all_scored_test, ignore_index=True)
            ranker.generate_submission(
                full_test_scored, output_path=Paths.SUBMISSION_DIR / "submission.csv"
            )
        else:
            logger.warning("No candidates generated for test set!")

    else:
        logger.info(
            f"Score {map_score} did not meet threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    run_pipeline()
