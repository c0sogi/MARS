import os
import pandas as pd
import numpy as np
import torch
from collections import Counter
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed, jaccard
from library.tapt_engine import run_tapt
from library.trainer import run_training
from library.inference import run_inference, predict_single_model
from library.data_loader import prepare_features, QADataset, collate_fn


def main():
    # 1. Configuration and Setup
    config = Config()
    set_seed(config.seed)

    print("==================================================")
    print("Starting End-to-End Pipeline")
    print("==================================================")

    # 2. Task-Adaptive Pretraining (TAPT)
    # Adapts the base model to the specific linguistic domain (Hindi/Tamil)
    print("\n[Step 1/4] Running Task-Adaptive Pretraining (TAPT)...")
    try:
        # load_cached_data=True allows skipping if already computed
        run_tapt(config, load_cached_data=True)
    except Exception as e:
        print(f"Warning: TAPT step encountered an issue: {e}")
        print("Proceeding with base model weights...")

    # 3. Question Answering Training
    # Trains the ensemble of models (Seeds 42, 43, 44)
    print("\n[Step 2/4] Running QA Training...")
    run_training(config)

    # 4. Validation and Failure Analysis
    print("\n[Step 3/4] Validating and Analyzing Performance...")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Prepare Validation Features
    val_features = prepare_features(
        config, tokenizer, split="val", load_cached_data=True
    )

    # Create DataLoader for Validation
    # We use is_test=True to skip internal label processing, as we want to run inference logic
    val_dataset = QADataset(val_features, is_test=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Load Ground Truth Metadata for accurate metric calculation
    val_meta_path = os.path.join(config.metadata_dir, "val.csv")
    val_meta = pd.read_csv(val_meta_path)
    # Map example_id to answer_text
    gt_map = dict(zip(val_meta["id"], val_meta["answer_text"].fillna("").astype(str)))

    # Generate predictions for each seed
    seed_predictions = []
    for seed in config.seeds:
        print(f"Generating validation predictions for seed {seed}...")
        preds = predict_single_model(config, seed, val_loader, tokenizer)
        seed_predictions.append(preds)

    # Ensemble Predictions (Majority Vote)
    print("Ensembling validation predictions...")
    unique_ids = val_features["example_id"].unique()

    final_val_preds = {}
    jaccard_scores = []
    analysis_data = []

    for eid in unique_ids:
        # Collect votes from all models for this example
        votes = [model_preds.get(eid, "") for model_preds in seed_predictions]

        # Apply Majority Voting
        if not votes:
            final_pred = ""
        else:
            # Counter.most_common(1) returns [(value, count)]
            final_pred = Counter(votes).most_common(1)[0][0]

        final_val_preds[eid] = final_pred

        # Calculate Jaccard Score
        gt_text = gt_map.get(eid, "")
        score = jaccard(gt_text, final_pred)
        jaccard_scores.append(score)

        # Collect data for failure analysis
        meta_row = val_meta[val_meta["id"] == eid]
        if not meta_row.empty:
            ctx_len = len(str(meta_row.iloc[0]["context"]))
            q_len = len(str(meta_row.iloc[0]["question"]))

            analysis_data.append(
                {
                    "jaccard": score,
                    "error": 1.0 - score,
                    "context_len": ctx_len,
                    "question_len": q_len,
                }
            )

    # Compute and Print Final Metric
    final_metric = np.mean(jaccard_scores)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    if analysis_data:
        df_analysis = pd.DataFrame(analysis_data)
        corr_ctx = df_analysis["error"].corr(df_analysis["context_len"])
        corr_q = df_analysis["error"].corr(df_analysis["question_len"])

        print(f"Correlation (Error vs Context Length): {corr_ctx}")
        print(f"Correlation (Error vs Question Length): {corr_q}")
    else:
        print("No analysis data available.")

    # 5. Submission
    print("\n[Step 4/4] Checking Submission Criteria...")
    threshold = 0.3011529653320698

    if final_metric > threshold:
        print(
            f"Validation metric {final_metric} exceeds threshold {threshold}. Proceeding to submission."
        )
        run_inference(config)
    else:
        print(
            f"Validation metric {final_metric} does not exceed threshold {threshold}. Submission skipped."
        )

    print("\nPipeline Complete.")


if __name__ == "__main__":
    main()
