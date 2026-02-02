import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import library components
from library.configuration import Config
from library.utils import seed_everything, load_data, jaccard
from library.tapt_manager import run_tapt_training
from library.qa_trainer import train_model
from library.inference_manager import (
    predict_for_model,
    majority_vote_ensemble,
    generate_submission,
)
from library.dataset_factory import prepare_qa_data, qa_collate_fn, get_tokenizer


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # --------------------------------------------------------------------------
    # Override epochs for fast baseline execution as requested
    # Reducing epochs ensures the pipeline finishes well within the time limit
    # while still demonstrating the learning capability on this small dataset.
    Config.EPOCHS = 3
    Config.TAPT_EPOCHS = 3

    # Ensure reproducibility
    seed_everything(Config.SEEDS[0])

    print(f"Starting pipeline on {Config.DEVICE}")
    print(f"QA Epochs: {Config.EPOCHS}, TAPT Epochs: {Config.TAPT_EPOCHS}")

    # --------------------------------------------------------------------------
    # 2. Task-Adaptive Pretraining (TAPT)
    # --------------------------------------------------------------------------
    print("\n=== Stage 1: Task-Adaptive Pretraining ===")
    # This adapts the base model to the domain text
    tapt_model_path = run_tapt_training(load_cached_data=True)
    print(f"TAPT model saved at: {tapt_model_path}")

    # --------------------------------------------------------------------------
    # 3. QA Fine-Tuning (Ensemble)
    # --------------------------------------------------------------------------
    print("\n=== Stage 2: QA Fine-Tuning (Ensemble) ===")
    trained_model_paths = []

    # Train a model for each seed
    for seed in Config.SEEDS:
        print(f"\nTraining model with Seed {seed}...")
        model_path = train_model(
            seed, pretrained_path=tapt_model_path, load_cached_data=True
        )
        trained_model_paths.append(model_path)

    print(f"\nTrained {len(trained_model_paths)} models.")

    # --------------------------------------------------------------------------
    # 4. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\n=== Stage 3: Validation & Failure Analysis ===")

    # Load Validation Data
    tokenizer = get_tokenizer()
    val_df = load_data("val")

    # Prepare DataLoader for Validation
    val_dataset = prepare_qa_data(tokenizer, split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=qa_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Run Inference with Ensemble
    print("Running inference on validation set...")
    val_predictions_list = []

    for path in trained_model_paths:
        # predict_for_model is designed for test set but works with any dataloader
        # that yields the expected batch structure (which val_loader does)
        preds = predict_for_model(path, val_loader, tokenizer, Config.DEVICE)
        val_predictions_list.append(preds)

    # Aggregate Predictions (Majority Vote)
    final_val_preds_map = majority_vote_ensemble(val_predictions_list)

    # Calculate Metrics
    y_true = []
    y_pred = []
    ids = []

    # Map IDs to ground truth
    gt_map = {str(row["id"]): str(row["answer_text"]) for _, row in val_df.iterrows()}

    for eid, gt_text in gt_map.items():
        pred_text = final_val_preds_map.get(eid, "")
        y_true.append(gt_text)
        y_pred.append(pred_text)
        ids.append(eid)

    # Compute Jaccard Scores
    scores = [jaccard(t, p) for t, p in zip(y_true, y_pred)]
    final_metric = np.mean(scores)

    # Print Metric (Full Precision)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    analysis_df = pd.DataFrame(
        {"id": ids, "jaccard": scores, "error_magnitude": [1.0 - s for s in scores]}
    )

    # Merge with input features
    # We use the original val_df loaded at the start
    # Ensure IDs are strings for merging
    val_df["id"] = val_df["id"].astype(str)
    analysis_df = analysis_df.merge(
        val_df[["id", "context", "question", "language"]], on="id", how="left"
    )

    # Calculate feature lengths
    analysis_df["context_char_len"] = analysis_df["context"].str.len()
    analysis_df["question_char_len"] = analysis_df["question"].str.len()

    # Calculate correlations
    # We look at how error magnitude correlates with input lengths
    correlations = analysis_df[
        ["error_magnitude", "context_char_len", "question_char_len"]
    ].corr()["error_magnitude"]

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # --------------------------------------------------------------------------
    # 5. Submission Generation
    # --------------------------------------------------------------------------
    print("\n=== Stage 4: Submission Generation ===")

    THRESHOLD = 0.3011529653320698

    if final_metric > THRESHOLD:
        print(
            f"Validation metric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(trained_model_paths, load_cached_data=True)
    else:
        print(
            f"Validation metric ({final_metric}) does NOT exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
