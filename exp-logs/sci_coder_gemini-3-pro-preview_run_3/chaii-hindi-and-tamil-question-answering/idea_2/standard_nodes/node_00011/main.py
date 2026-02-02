import os
import pandas as pd
import torch
import numpy as np
from library.config import Config
from library.trainer import run_training_pipeline
from library.inference import run_inference_pipeline, predict_with_model, majority_vote
from library.data_loader import get_dataloaders
from library.model import get_model
from library.utils import jaccard, set_seed
from transformers import AutoTokenizer


def main():
    # Ensure reproducibility
    set_seed(42)

    # 1. Training Phase
    # We use the full dataset (approx 800 samples) as it is small enough for fast training.
    # We use the default 10 epochs defined in Config.
    print("Starting training pipeline...")
    run_training_pipeline(debug=False)

    # 2. Validation Phase
    print("Starting validation...")

    # Load validation data loader and raw dataframe for ground truth
    _, val_loader, _ = get_dataloaders(debug=False, load_cached_data=True)
    val_df = pd.read_csv(Config.VAL_FILE)

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)
    device = Config.DEVICE

    # Collect predictions from all ensemble models
    all_seed_preds = []

    for seed in Config.SEEDS:
        model_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pt")
        if not os.path.exists(model_path):
            print(f"Model for seed {seed} not found, skipping.")
            continue

        # Load model
        model = get_model()
        model.load_state_dict(torch.load(model_path, map_location=device))

        # Inference
        preds = predict_with_model(model, val_loader, tokenizer, device)
        all_seed_preds.append(preds)

        # Cleanup to save memory
        del model
        torch.cuda.empty_cache()

    # Aggregate predictions using Majority Voting
    final_preds_map = majority_vote(all_seed_preds)

    # Calculate Jaccard Scores
    scores = []
    ids = []

    # Iterate through the validation dataframe to ensure alignment with Ground Truth
    for idx, row in val_df.iterrows():
        ex_id = row["id"]
        gt = str(row["answer_text"]) if not pd.isna(row["answer_text"]) else ""
        pred = final_preds_map.get(ex_id, "")

        score = jaccard(gt, pred)
        scores.append(score)
        ids.append(ex_id)

    final_metric = sum(scores) / len(scores) if scores else 0.0

    # Print required metric with full precision
    print(f"Final Validation Metric: {final_metric}")

    # 3. Failure Analysis
    print("Performing failure analysis...")

    # Create analysis dataframe
    analysis_df = pd.DataFrame(
        {
            "id": ids,
            "jaccard": scores,
            "error": [1.0 - s for s in scores],
            # Feature extraction for correlation analysis
            "context_len": val_df["context"].astype(str).apply(len),
            "question_len": val_df["question"].astype(str).apply(len),
            "answer_start": val_df["answer_start"].fillna(0),
        }
    )

    # Calculate correlations
    correlations = analysis_df[
        ["error", "context_len", "question_len", "answer_start"]
    ].corr()["error"]
    print("Correlations with Error Magnitude:")
    print(correlations)

    # 4. Submission
    THRESHOLD = 0.3011529653320698
    if final_metric > THRESHOLD:
        print(f"Metric {final_metric} > {THRESHOLD}. Generating submission...")
        run_inference_pipeline()
    else:
        print(f"Metric {final_metric} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
