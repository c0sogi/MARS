import pandas as pd
import numpy as np
import torch
import gc
import sys
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

# Import provided library modules
from library import config
from library import utils
from library import data_loader
from library import sequential_encoder
from library import heuristics
from library import candidate_generation
from library import feature_engineering
from library import ranking

# ==========================================
# CONFIGURATION OVERRIDES FOR FAST BASELINE
# ==========================================
# Reduce epochs and increase batch size for speed
FAST_SEQ_CONFIG = config.SEQ_CONFIG.copy()
FAST_SEQ_CONFIG["epochs"] = 4
FAST_SEQ_CONFIG["batch_size"] = 4096

# Reduce ranker complexity slightly for speed
FAST_LGBM_PARAMS = config.LGBM_PARAMS.copy()
FAST_LGBM_PARAMS["n_estimators"] = 800  # Reduced from 1500

VALIDATION_THRESHOLD = 0.0306342353457529


def run_failure_analysis(predictions_df, ground_truth_df, features_df):
    """
    Analyzes correlations between error and features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate Average Precision per user
    # predictions_df: customer_id, prediction (str)
    # ground_truth_df: customer_id, article_id (one row per item)

    # Group GT
    gt_grouped = (
        ground_truth_df.groupby("customer_id")["article_id"].apply(set).to_dict()
    )

    # Parse predictions
    user_aps = []
    customer_ids = []

    for _, row in predictions_df.iterrows():
        cid = row["customer_id"]
        if cid not in gt_grouped:
            continue

        actual = gt_grouped[cid]
        preds = row["prediction"].split()[:12]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(preds):
            if p in actual:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        ap = score / min(len(actual), 12)
        user_aps.append(ap)
        customer_ids.append(cid)

    analysis_df = pd.DataFrame({"customer_id": customer_ids, "ap": user_aps})
    analysis_df["error"] = 1.0 - analysis_df["ap"]

    # Merge with some features (e.g. age, activity)
    # We can get these from features_df (taking the first row per customer)
    # features_df contains customer_id, age, purchase_count, etc.

    if "age" in features_df.columns:
        feat_subset = features_df[
            ["customer_id", "age", "FN", "Active"]
        ].drop_duplicates("customer_id")
        analysis_df = analysis_df.merge(feat_subset, on="customer_id", how="left")

        # Correlations
        print("Correlation with Error (1 - AP):")
        for col in ["age", "FN", "Active"]:
            if col in analysis_df.columns:
                corr = analysis_df["error"].corr(analysis_df[col])
                print(f"  - {col}: {corr:.4f}")
    else:
        print("Features for analysis not available.")


def main():
    # 1. Setup
    utils.seed_everything(config.RANDOM_STATE)
    print("Initializing Pipeline...")

    # 2. Data Loading
    # Load full history (Train + Val parquets combined)
    df_history, df_articles, df_customers = data_loader.load_raw_data()

    # Split into Train (History) and Validation (Target Period)
    # Idea 5: Use last 7 days for validation
    train_df, val_df = data_loader.get_time_split(df_history, val_days=config.VAL_DAYS)

    # 3. Stage 1: Train Retrieval Models (on train_df)
    print("\n=== Stage 1: Retrieval Model Training ===")

    # A. Sequential Model
    # Preprocess
    seq_data = data_loader.preprocess_sequences(
        train_df,
        min_history=config.SEQ_MIN_HISTORY,
        max_seq_len=FAST_SEQ_CONFIG["max_seq_len"],
        load_cached_data=True,
    )

    # Train
    seq_model = sequential_encoder.train_sequential_model(
        seq_data, params=FAST_SEQ_CONFIG, load_cached_data=True
    )

    # Extract Embeddings
    user_embs, item_embs = sequential_encoder.extract_embeddings(seq_model, seq_data)

    # B. Co-occurrence Matrix
    cooc_model = heuristics.CooccurrenceMatrix()
    cooc_model.fit(train_df, weeks=config.COOC_HISTORY_WEEKS, load_cached_data=True)

    # 4. Stage 2: Ranker Preparation
    print("\n=== Stage 2: Ranker Preparation (Validation Split) ===")

    # Identify unique customers in Validation set
    val_customers = val_df["customer_id"].unique()
    print(f"Total Validation Customers: {len(val_customers)}")

    # Split Validation Customers into Ranker-Train (80%) and Ranker-Val (20%)
    # This allows us to train the ranker on 'unseen' data relative to the retrieval training
    r_train_cust, r_val_cust = train_test_split(
        val_customers, test_size=0.2, random_state=config.RANDOM_STATE
    )

    orchestrator = candidate_generation.CandidateOrchestrator()
    feature_builder = feature_engineering.FeatureBuilder()

    # --- Process Ranker-Train Set ---
    print(f"Generating candidates for Ranker-Train ({len(r_train_cust)} users)...")
    cands_r_train = orchestrator.generate_candidates(
        train_df,
        r_train_cust,
        cache_path=config.WORKING_DIR / "candidates_ranker_train.parquet",
        load_cached_data=True,
    )

    feats_r_train = feature_builder.compute_features(
        cands_r_train,
        train_df,
        df_articles,
        df_customers,
        seq_data=seq_data,
        user_embeddings=user_embs,
        item_embeddings=item_embs,
        cooc_matrix=cooc_model,
        load_cached_data=True,
        cache_name="features_ranker_train.parquet",
    )

    # --- Process Ranker-Val Set ---
    print(f"Generating candidates for Ranker-Val ({len(r_val_cust)} users)...")
    cands_r_val = orchestrator.generate_candidates(
        train_df,
        r_val_cust,
        cache_path=config.WORKING_DIR / "candidates_ranker_val.parquet",
        load_cached_data=True,
    )

    feats_r_val = feature_builder.compute_features(
        cands_r_val,
        train_df,
        df_articles,
        df_customers,
        seq_data=seq_data,
        user_embeddings=user_embs,
        item_embeddings=item_embs,
        cooc_matrix=cooc_model,
        load_cached_data=True,
        cache_name="features_ranker_val.parquet",
    )

    # Prepare Ground Truth
    gt_r_train = val_df[val_df["customer_id"].isin(r_train_cust)]
    gt_r_val = val_df[val_df["customer_id"].isin(r_val_cust)]

    # 5. Stage 3: Ranker Training & Evaluation
    print("\n=== Stage 3: Ranker Training ===")

    # Define features to use (exclude IDs and non-numeric)
    feature_cols = [
        c
        for c in feats_r_train.columns
        if c not in ["customer_id", "article_id", "prediction"]
    ]

    ranker = ranking.LGBMRankerWrapper()
    ranker.train(
        feats_r_train,
        feats_r_val,
        gt_r_train,
        gt_r_val,
        feature_cols,
        params=FAST_LGBM_PARAMS,
        load_cached_model=False,
    )

    # Validation Inference
    print("Running Validation Inference...")
    val_preds = ranker.predict(feats_r_val, feature_cols)

    # Metric Calculation
    print("Calculating MAP@12...")
    map12 = utils.calculate_map12(gt_r_val, val_preds)

    print(f"Final Validation Metric: {map12}")

    # Failure Analysis
    run_failure_analysis(val_preds, gt_r_val, feats_r_val)

    # Clean up memory
    del cands_r_train, feats_r_train, cands_r_val, feats_r_val, gt_r_train, gt_r_val
    gc.collect()

    # 6. Submission
    if map12 > VALIDATION_THRESHOLD:
        print("\n=== Generating Submission ===")

        # Load Test Customers
        test_meta = pd.read_parquet(config.TEST_DATA_PATH)
        test_customers = test_meta["customer_id"].unique()
        print(f"Test Customers: {len(test_customers)}")

        # --- Update Retrieval Models on Full History (Train + Val) ---
        print("Updating Retrieval Models with Full History...")

        # 1. Update Cooc Matrix
        cooc_model.fit(
            df_history, weeks=config.COOC_HISTORY_WEEKS, load_cached_data=False
        )

        # 2. Update User Embeddings (using trained SASRec on new history)
        # We need to re-process sequences including the validation data
        full_seq_data = data_loader.preprocess_sequences(
            df_history,
            min_history=config.SEQ_MIN_HISTORY,
            max_seq_len=FAST_SEQ_CONFIG["max_seq_len"],
            load_cached_data=False,  # Recompute for full history
        )
        # Extract using the *already trained* model
        # Note: Vocabulary must match. If new items appeared in Val (unlikely to be many),
        # they are mapped. preprocess_sequences handles mapping.
        # Ideally we should retrain SASRec, but for speed we infer.
        # However, vocab size might change. If vocab size changes, the model embedding layer won't match.
        # Given the constraints and typical data, we assume vocab is largely static or we handle strictly.
        # To be safe and fast, we will use the *original* seq_data structures if possible,
        # but preprocess_sequences builds a new map.
        # Strategy: Retrain SASRec quickly on full data? Or just use Train-based model?
        # Using Train-based model on Test data is safer for dimension matching if we don't retrain.
        # But we want the latest history.
        # Let's try to extract. If dimensions mismatch, we might need to handle it.
        # Actually, `extract_embeddings` uses the model's embedding layer.
        # If `full_seq_data` has indices > model vocab, it will crash.
        # For this baseline, we will stick to the models trained on `train_df` but apply them
        # to the candidates generated from `df_history` (heuristics).
        # For SASRec, we will skip updating embeddings to avoid vocab mismatch crashes in this fast script,
        # and rely on the embeddings from Step 3 (which covers 80% of users + history up to T-7).
        # Wait, test users need predictions based on history up to T.
        # If we don't update sequences, we miss the last week.
        # Correct approach for baseline: Use `df_history` for Heuristics (Cooc, Repurchase, Trend).
        # For SASRec features, use the embeddings we have. If a user has new history, we miss it in SASRec
        # but catch it in Cooc/Repurchase. This is a valid trade-off for speed/stability.

        # --- Generate Test Candidates ---
        # We use df_history for Cooc/Repurchase/Trend to get freshest candidates
        cands_test = orchestrator.generate_candidates(
            df_history,
            test_customers,
            cache_path=config.WORKING_DIR / "candidates_test.parquet",
            load_cached_data=True,
        )

        # --- Compute Features ---
        # We pass df_history so trend/repurchase features are up to date
        feats_test = feature_builder.compute_features(
            cands_test,
            df_history,
            df_articles,
            df_customers,
            seq_data=seq_data,
            user_embeddings=user_embs,
            item_embeddings=item_embs,
            cooc_matrix=cooc_model,
            load_cached_data=True,
            cache_name="features_test.parquet",
        )

        # --- Predict ---
        ranker.predict(feats_test, feature_cols)

        print("Submission generation complete.")
    else:
        print(
            f"Validation metric {map12} did not meet threshold {VALIDATION_THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
