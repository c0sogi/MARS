import os
import sys
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, load_metadata, ensure_dir
from library.symbolic_model import NgramLookup
from library.training_engine import Trainer
from library.inference_engine import HybridPredictor
from library.data_processing import TextNormalizationDataset, collate_fn, CharTokenizer


def main():
    # 1. Configuration and Setup
    config = Config()
    # Override epochs for a fast baseline execution as per requirements
    config.num_epochs = 5
    seed_everything(config.seed)

    print("=== Starting End-to-End Pipeline ===")

    # 2. Train/Load Symbolic Model (The "Head")
    print("\n[Step 1/5] Initializing Symbolic Solver...")
    ngram_lookup = NgramLookup(config)
    # Fits on full training data (fast)
    ngram_lookup.fit(load_cached_data=True)

    # 3. Train/Load Neural Model (The "Tail")
    print("\n[Step 2/5] Training Neural Solver...")
    trainer = Trainer(config)
    # Limit to 300k samples to ensure completion within 2 hours
    neural_normalizer = trainer.run(load_cached_data=True, max_samples=300000)

    # 4. Validation and Metrics
    print("\n[Step 3/5] Performing Validation...")
    # Load validation metadata
    val_df = load_metadata(config.val_file)

    # Sort by sentence_id and token_id to ensure correct context reconstruction
    if "token_id" in val_df.columns:
        val_df = val_df.sort_values(["sentence_id", "token_id"])

    # Group by sentence_id to reconstruct context
    print("Grouping validation data...")
    grouped = val_df.groupby("sentence_id")[
        ["token_id", "before", "after", "class"]
    ].agg(list)

    # Prepare for inference
    val_ids = []
    val_preds = {}  # id -> predicted_text
    val_targets = {}  # id -> true_text
    val_meta = []  # List of dicts for failure analysis

    neural_candidates = []  # List of dicts for batch neural inference
    digit_pattern = re.compile(r"\d")

    print("Running Hybrid Inference on Validation Set...")
    # Iterate through reconstructed sentences
    for sentence_id, row in grouped.iterrows():
        token_ids = row["token_id"]
        tokens = row["before"]
        targets = row["after"]
        classes = row["class"]
        seq_len = len(tokens)

        for i in range(seq_len):
            t_id = token_ids[i]
            curr_token = str(tokens[i])
            target_token = str(targets[i])
            token_class = str(classes[i])

            uid = f"{sentence_id}_{t_id}"
            val_ids.append(uid)
            val_targets[uid] = target_token

            # Store metadata for failure analysis
            val_meta.append(
                {
                    "id": uid,
                    "class": token_class,
                    "len_before": len(curr_token),
                    "sent_len": seq_len,
                }
            )

            # Context
            prev_token = str(tokens[i - 1]) if i > 0 else "<s>"
            next_token = str(tokens[i + 1]) if i < seq_len - 1 else "</s>"

            # Strategy 1: Symbolic Lookup
            sym_pred = ngram_lookup.get_normalization(
                curr_token, prev_token, next_token
            )

            if sym_pred is not None:
                val_preds[uid] = sym_pred
            else:
                # Strategy 2: Check Complexity
                if digit_pattern.search(curr_token):
                    # Route to Neural Model
                    # Construct Context-Aware Input: "... prev <tgt> curr </tgt> next ..."
                    tagged_tokens = tokens.copy()
                    tagged_tokens[i] = (
                        f"{config.tgt_start_token} {curr_token} {config.tgt_end_token}"
                    )
                    input_text = " ".join(map(str, tagged_tokens))

                    neural_candidates.append(
                        {
                            "id": uid,
                            "input_text": input_text,
                            "target_text": "",  # Dummy
                        }
                    )
                else:
                    # Identity Fallback for "easy" unknown tokens
                    val_preds[uid] = curr_token

    # Run Neural Inference in Batch
    if neural_candidates:
        print(f"Processing {len(neural_candidates)} neural candidates...")
        df_neural = pd.DataFrame(neural_candidates)

        # Reuse the tokenizer from the trained model
        tokenizer = neural_normalizer.tokenizer

        # Create Dataset in 'test' mode (ignores targets)
        neural_dataset = TextNormalizationDataset(
            df_neural, tokenizer, config, mode="test"
        )

        neural_loader = DataLoader(
            neural_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # Predict
        neural_predictions = neural_normalizer.predict(neural_loader)
        val_preds.update(neural_predictions)

    # Calculate Metric
    print("Calculating Accuracy...")
    correct_count = 0
    total_count = len(val_ids)

    # Update metadata with correctness
    for meta in val_meta:
        uid = meta["id"]
        pred = val_preds.get(uid, "")
        tgt = val_targets.get(uid, "")
        is_correct = pred == tgt
        meta["is_correct"] = int(is_correct)
        if is_correct:
            correct_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0.0
    print(f"Final Validation Metric: {accuracy}")

    # 5. Failure Analysis
    print("\n[Step 4/5] Failure Analysis...")
    df_analysis = pd.DataFrame(val_meta)

    # Convert categorical class to codes for correlation
    df_analysis["class_code"] = df_analysis["class"].astype("category").cat.codes

    # Calculate correlations
    # We are interested in what correlates with 'is_correct'
    correlations = df_analysis[
        ["is_correct", "class_code", "len_before", "sent_len"]
    ].corr()["is_correct"]
    print("Correlation with Error (is_correct):")
    print(correlations.drop("is_correct"))

    # 6. Submission
    print("\n[Step 5/5] Checking Submission Criteria...")
    threshold = 0.9798075665557208

    if accuracy > threshold:
        print(
            f"Accuracy {accuracy} exceeds threshold {threshold}. Generating submission..."
        )
        # Initialize HybridPredictor (loads best model from disk)
        predictor = HybridPredictor(config)
        predictor.predict(load_cached_data=True)
    else:
        print(
            f"Accuracy {accuracy} does not exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
