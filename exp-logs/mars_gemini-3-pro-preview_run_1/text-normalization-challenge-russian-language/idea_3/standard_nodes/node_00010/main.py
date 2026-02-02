import sys
import os
import pandas as pd
import numpy as np
import torch
import re
from collections import Counter

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, compute_accuracy
from library.trainer import ModelTrainer
from library.inference_engine import HybridRouter
from library.data_processing import load_and_group_data


def run_validation_inference(router, val_df):
    """
    Runs the hybrid inference logic on the validation set to get predictions.
    Replicates the logic from HybridRouter.generate_submission but for validation evaluation.
    """
    results = []  # (index, prediction)
    neural_queue = []  # (index, context_data)
    targets = []

    # Pre-compile regex
    digit_pattern = re.compile(r"\d")

    print("Performing inference on validation set...")

    # Flatten validation data for token-level processing
    # We need to iterate tokens to apply the router logic
    # val_df is grouped by sentence.

    global_token_idx = 0

    # We will store predictions in a list aligned with the flattened tokens
    # To do this efficiently, we iterate and store 'None' for neural, then fill in.

    all_preds = []
    all_targets = []
    all_tokens = []  # For failure analysis
    all_classes = []  # For failure analysis

    # Rename 'class' column to 'class_' if it exists to avoid itertuples issues
    if "class" in val_df.columns:
        val_df = val_df.rename(columns={"class": "class_"})

    # Iterate over sentences
    for row in val_df.itertuples(index=False):
        tokens = row.before
        row_targets = row.after
        row_classes = row.class_
        n_tokens = len(tokens)

        for i in range(n_tokens):
            curr_w = tokens[i]
            target_w = row_targets[i]
            token_cls = row_classes[i]

            all_targets.append(target_w)
            all_tokens.append(curr_w)
            all_classes.append(token_cls)

            # Context
            prev_w = tokens[i - 1] if i > 0 else "<start>"
            next_w = tokens[i + 1] if i < n_tokens - 1 else "<end>"

            # --- Step 1: Trigram ---
            trigram_key = (prev_w, curr_w, next_w)
            if (
                router.symbolic.stats
                and trigram_key in router.symbolic.stats["trigram"]
            ):
                all_preds.append(router.symbolic.stats["trigram"][trigram_key])
                global_token_idx += 1
                continue

            # --- Step 2: Bigram ---
            if router.symbolic.stats:
                bigram_key = (prev_w, curr_w)
                if bigram_key in router.symbolic.stats["bigram"]:
                    all_preds.append(router.symbolic.stats["bigram"][bigram_key])
                    global_token_idx += 1
                    continue

                # --- Step 3: Unigram ---
                unigram_key = curr_w
                if unigram_key in router.symbolic.stats["unigram"]:
                    all_preds.append(router.symbolic.stats["unigram"][unigram_key])
                    global_token_idx += 1
                    continue

            # --- Step 4: Neural (Digits) ---
            if digit_pattern.search(curr_w):
                # Prepare context
                left_ctx = []
                for k in range(1, Config.CONTEXT_WINDOW + 1):
                    if i - k >= 0:
                        left_ctx.insert(0, tokens[i - k])
                    else:
                        break

                right_ctx = []
                for k in range(1, Config.CONTEXT_WINDOW + 1):
                    if i + k < n_tokens:
                        right_ctx.append(tokens[i + k])
                    else:
                        break

                neural_data = {
                    "left": " ".join(left_ctx),
                    "center": curr_w,
                    "right": " ".join(right_ctx),
                }

                all_preds.append(None)  # Placeholder
                neural_queue.append((global_token_idx, neural_data))
                global_token_idx += 1
                continue

            # --- Step 5: Identity ---
            all_preds.append(curr_w)
            global_token_idx += 1

    # Process Neural Queue
    if neural_queue:
        print(f"Processing {len(neural_queue)} neural candidates...")
        batch_size = Config.BATCH_SIZE

        # We can reuse the internal method if we access it, or replicate it.
        # Replicating to ensure standalone robustness.

        for i in range(0, len(neural_queue), batch_size):
            batch_items = neural_queue[i : i + batch_size]
            batch_indices = [item[0] for item in batch_items]
            batch_data = [item[1] for item in batch_items]

            input_tensors = []
            sep_id = router.tokenizer.sep_token_id

            for item in batch_data:
                left_ids = router.tokenizer.encode(item["left"])
                center_ids = router.tokenizer.encode(item["center"])
                right_ids = router.tokenizer.encode(item["right"])
                ids = left_ids + [sep_id] + center_ids + [sep_id] + right_ids
                input_tensors.append(torch.tensor(ids, dtype=torch.long))

            padded_input = torch.nn.utils.rnn.pad_sequence(
                input_tensors,
                batch_first=True,
                padding_value=router.tokenizer.pad_token_id,
            ).to(router.device)

            with torch.no_grad():
                output_ids = router.model.predict(padded_input)

            for j, out_seq in enumerate(output_ids):
                pred_text = router.tokenizer.decode(out_seq.cpu().tolist())
                idx = batch_indices[j]
                all_preds[idx] = pred_text

    return all_preds, all_targets, all_tokens, all_classes


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Train
    print("\n=== Starting Training Phase ===")
    trainer = ModelTrainer()
    # Force retrain to ensure we use the 1 epoch setting
    trainer.run(force_retrain=True)

    # 3. Validation
    print("\n=== Starting Validation Phase ===")
    # Load validation data
    val_df = load_and_group_data("val", load_cached_data=True)

    # Initialize Router (loads the model we just trained)
    router = HybridRouter()

    # Run Inference
    preds, targets, tokens, classes = run_validation_inference(router, val_df)

    # Compute Metric
    metric = compute_accuracy(preds, targets)
    print(f"Final Validation Metric: {metric}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Construct DataFrame for analysis
    df_analysis = pd.DataFrame(
        {"token": tokens, "pred": preds, "target": targets, "class": classes}
    )

    # Determine errors
    df_analysis["is_error"] = (df_analysis["pred"] != df_analysis["target"]).astype(int)
    df_analysis["token_len"] = df_analysis["token"].astype(str).str.len()

    total_errors = df_analysis["is_error"].sum()
    print(f"Total Errors: {total_errors} / {len(df_analysis)}")

    # Correlation with Token Length
    # Point-biserial correlation is effectively correlation between binary and continuous
    corr_len = df_analysis["is_error"].corr(df_analysis["token_len"])
    print(f"Correlation (Error vs Token Length): {corr_len:.4f}")

    # Error Rate by Class
    print("\nError Rate by Class (Top 10 by volume):")
    class_stats = (
        df_analysis.groupby("class")
        .agg(count=("is_error", "count"), error_rate=("is_error", "mean"))
        .sort_values("count", ascending=False)
        .head(10)
    )

    print(class_stats)

    # 5. Submission
    print("\n=== Submission Check ===")
    THRESHOLD = 0.9784130832472395

    if metric > THRESHOLD:
        print(f"Metric {metric} > {THRESHOLD}. Generating submission...")
        submission_path = "./submission/submission.csv"
        router.generate_submission(submission_path)
        print(f"Submission saved to {submission_path}")
    else:
        print(f"Metric {metric} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
