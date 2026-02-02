import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from collections import defaultdict

# Import from provided library
from library.config import Config
from library.utils import seed_everything, jaccard
from library.modeling import CustomXLMRoberta
from library.dataset import QADataset, process_train_data, process_test_data
from library.engine import get_optimizer, get_scheduler, train_fn
from library.predict import predict as generate_submission


def main():
    # 1. Setup & Configuration
    config = Config()

    # Override for Fast Baseline
    config.epochs = 2
    config.ensemble_seeds = [42]
    config.train_batch_size = 4
    config.eval_batch_size = 16

    # Ensure working directories exist
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.cache_dir, exist_ok=True)

    seed_everything(config.seed)
    device = config.device

    print(
        f"Running with config: Epochs={config.epochs}, Seed={config.ensemble_seeds[0]}"
    )

    # 2. Data Preparation
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Load Metadata
    train_df = pd.read_csv(config.train_path)
    val_df = pd.read_csv(config.val_path)

    print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")

    # Process Training Data (with labels and negative sampling)
    print("Processing Training Data...")
    train_features = process_train_data(train_df, tokenizer, config)
    train_ds = QADataset(train_features, mode="train")
    train_loader = DataLoader(
        train_ds,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    # Process Validation Data (as Test data for inference simulation)
    print("Processing Validation Data...")
    val_features = process_test_data(val_df, tokenizer, config)
    # Fix for QADataset compatibility: convert offset_mapping list to numpy array
    val_features["offset_mapping"] = val_features["offset_mapping"].apply(np.array)

    val_ds = QADataset(val_features, mode="test")
    val_loader = DataLoader(
        val_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    # 3. Training Loop
    model = CustomXLMRoberta(config)
    model.to(device)

    optimizer = get_optimizer(model, config)
    num_train_steps = len(train_loader) * config.epochs
    scheduler = get_scheduler(optimizer, num_train_steps, config)

    print("Starting Training...")
    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")
        train_fn(train_loader, model, optimizer, device, scheduler, config)

    # Save Model (required for generate_submission)
    model_save_path = os.path.join(
        config.output_dir, f"model_seed_{config.ensemble_seeds[0]}.pth"
    )
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # 4. Validation Inference & Scoring
    print("Running Validation Inference...")
    model.eval()
    results = defaultdict(list)

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            s, e, r = model(input_ids, attention_mask)

            # Move to CPU
            s = s.cpu().numpy()
            e = e.cpu().numpy()
            r = r.cpu().numpy()
            b_input_ids = batch["input_ids"].cpu().numpy()
            offsets = batch["offset_mapping"].numpy()
            ex_ids = batch["example_id"]

            for i, ex_id in enumerate(ex_ids):
                results[ex_id].append(
                    {
                        "start_logits": s[i],
                        "end_logits": e[i],
                        "relevance_logit": r[i],
                        "offset_mapping": offsets[i],
                        "input_ids": b_input_ids[i],
                    }
                )

    # Reconstruct Answers
    print("Reconstructing Validation Answers...")
    val_predictions = {}

    for _, row in val_df.iterrows():
        ex_id = row["id"]
        context_text = str(row["context"])

        if ex_id not in results:
            val_predictions[ex_id] = ""
            continue

        windows = results[ex_id]
        best_overall_score = -float("inf")
        best_prediction_string = ""

        for win in windows:
            start_logits = win["start_logits"]
            end_logits = win["end_logits"]
            rel_logit = win["relevance_logit"]
            offsets = win["offset_mapping"]
            input_ids = win["input_ids"]

            # Identify Context Boundaries (Same logic as predict.py)
            sep_indices = np.where(input_ids == 2)[0]
            if len(sep_indices) >= 2:
                context_start_idx = sep_indices[1] + 1
                context_end_idx = (
                    sep_indices[2] if len(sep_indices) > 2 else len(input_ids) - 1
                )
            else:
                context_start_idx = 0
                context_end_idx = len(input_ids) - 1

            top_start_indices = np.argsort(start_logits)[-config.n_best_size :]
            top_end_indices = np.argsort(end_logits)[-config.n_best_size :]

            for s_idx in top_start_indices:
                if s_idx < context_start_idx or s_idx >= context_end_idx:
                    continue
                for e_idx in top_end_indices:
                    if e_idx < context_start_idx or e_idx >= context_end_idx:
                        continue
                    if s_idx > e_idx:
                        continue
                    if e_idx - s_idx + 1 > config.max_answer_length:
                        continue

                    score = start_logits[s_idx] + end_logits[e_idx] + rel_logit

                    if score > best_overall_score:
                        best_overall_score = score
                        start_char = offsets[s_idx][0]
                        end_char = offsets[e_idx][1]

                        if 0 <= start_char < len(context_text) and end_char <= len(
                            context_text
                        ):
                            best_prediction_string = context_text[start_char:end_char]

        val_predictions[ex_id] = best_prediction_string

    # Calculate Metric
    scores = []
    for _, row in val_df.iterrows():
        gt = str(row["answer_text"])
        pred = val_predictions.get(row["id"], "")
        scores.append(jaccard(gt, pred))

    final_metric = np.mean(scores)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("Performing Failure Analysis...")
    analysis_df = val_df.copy()
    analysis_df["jaccard"] = scores
    analysis_df["error"] = 1.0 - analysis_df["jaccard"]

    # Calculate features
    analysis_df["context_len"] = analysis_df["context"].apply(len)
    analysis_df["question_len"] = analysis_df["question"].apply(len)

    # Compute correlations
    correlations = analysis_df[["error", "context_len", "question_len"]].corr()["error"]
    print("Correlation between Error and Input Features:")
    print(correlations)

    # 6. Submission
    threshold = 0.60025
    if final_metric > threshold:
        print(
            f"Metric ({final_metric:.5f}) > Threshold ({threshold}). Generating submission..."
        )
        # Config is already set up with correct seed and output dir
        generate_submission(config)
    else:
        print(
            f"Metric ({final_metric:.5f}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
