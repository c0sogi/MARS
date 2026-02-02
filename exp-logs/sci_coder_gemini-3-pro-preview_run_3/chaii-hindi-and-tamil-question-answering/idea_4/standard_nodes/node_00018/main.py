import os
import pandas as pd
import numpy as np
import torch
from library.config import Config
from library.utils import set_seed, jaccard
from library.tapt_engine import run_tapt
from library.qa_data import prepare_qa_data
from library.qa_trainer import train_fold
from library.inference_engine import (
    run_inference_for_seed,
    ensemble_predictions,
    predict_and_submit,
)


def main():
    # 1. Setup
    Config.setup()
    set_seed(42)

    print("=== Starting Orchestration Script ===")

    # 2. Task-Adaptive Pretraining (TAPT)
    # Adapts the backbone model to the specific linguistic domain of the dataset
    print("\n[Step 1/5] Running Task-Adaptive Pretraining (TAPT)...")
    run_tapt()

    # 3. Data Preparation
    # Loads metadata, tokenizes, and creates PyTorch datasets
    print("\n[Step 2/5] Preparing QA Data...")
    train_dataset, val_dataset, test_dataset, test_features = prepare_qa_data(
        load_cached_data=True
    )

    # Manually load validation features from cache for inference mapping
    # (prepare_qa_data returns dataset tensors but we need feature dicts for text decoding)
    val_features_path = os.path.join(Config.QA_CACHE_DIR, "val_features.parquet")
    if not os.path.exists(val_features_path):
        raise FileNotFoundError(f"Validation features not found at {val_features_path}")

    print(f"Loading validation features from {val_features_path}...")
    val_features_df = pd.read_parquet(val_features_path)
    val_features = val_features_df.to_dict("records")

    # Fix sequence_ids: Parquet stores None as -1 (or similar integer handling), revert to None
    # Also ensure offset_mapping is in list format
    for f in val_features:
        f["sequence_ids"] = [x if x != -1 else None for x in f["sequence_ids"]]
        # Ensure example_id is string
        f["example_id"] = str(f["example_id"])

    # 4. Model Training (Ensemble)
    # Train 3 independent models with different seeds
    print("\n[Step 3/5] Training QA Models (Ensemble)...")
    for seed in Config.SEEDS:
        print(f"--- Training Fold Seed {seed} ---")
        train_fold(train_dataset, val_dataset, seed)

    # 5. Validation Inference & Analysis
    print("\n[Step 4/5] Validation Inference & Analysis...")

    # Run inference on validation set for each seed
    val_model_outputs = []
    for seed in Config.SEEDS:
        print(f"Running validation inference for seed {seed}...")
        preds = run_inference_for_seed(seed, val_dataset, val_features)
        if preds:
            val_model_outputs.append(preds)

    if not val_model_outputs:
        raise RuntimeError("No predictions generated during validation.")

    # Ensemble predictions via Majority Voting
    val_predictions = ensemble_predictions(val_model_outputs)

    # Load Ground Truth for Evaluation
    print("Loading ground truth metadata...")
    df_val_meta = pd.read_csv(Config.VAL_META_PATH)
    gt_map = dict(
        zip(
            df_val_meta["id"].astype(str),
            df_val_meta["answer_text"].fillna("").astype(str),
        )
    )

    # Calculate Jaccard Score
    scores = []
    ids = []

    for eid, pred_str in val_predictions.items():
        gt_str = gt_map.get(eid, "")
        score = jaccard(gt_str, pred_str)
        scores.append(score)
        ids.append(eid)

    final_metric = np.mean(scores)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("Performing Failure Analysis...")
    analysis_df = df_val_meta[df_val_meta["id"].astype(str).isin(ids)].copy()

    # Map scores
    score_map = dict(zip(ids, scores))
    analysis_df["jaccard"] = analysis_df["id"].astype(str).map(score_map)
    analysis_df["error"] = 1.0 - analysis_df["jaccard"]

    # Compute features
    analysis_df["context_len"] = (
        analysis_df["context"].fillna("").astype(str).apply(len)
    )
    analysis_df["question_len"] = (
        analysis_df["question"].fillna("").astype(str).apply(len)
    )

    # Calculate correlations
    corr_ctx = analysis_df["error"].corr(analysis_df["context_len"])
    corr_q = analysis_df["error"].corr(analysis_df["question_len"])

    print(f"Correlation (Error vs Context Length): {corr_ctx}")
    print(f"Correlation (Error vs Question Length): {corr_q}")

    # 6. Submission
    print("\n[Step 5/5] Submission Generation...")
    THRESHOLD = 0.3011529653320698

    if final_metric > THRESHOLD:
        print(
            f"Metric {final_metric} > {THRESHOLD}. Generating submission for test set..."
        )
        predict_and_submit(test_dataset, test_features)
    else:
        print(f"Metric {final_metric} <= {THRESHOLD}. Submission skipped.")

    print("=== Execution Complete ===")


if __name__ == "__main__":
    main()
