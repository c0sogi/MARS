import os
import pandas as pd
import numpy as np
import torch
import shutil
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from library.config import Config
from library.utils import set_seed, jaccard, get_optimizer_params
from library.model import XLMRobertaForQA
from library.data import prepare_train_features, prepare_test_features, QADataset
from library.engine import train_fn, predict_fn


def main():
    # 1. Setup
    config = Config()
    set_seed(config.seed)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    device = config.device

    print(f"Using device: {device}")

    # ==================================================================================
    # PART 1: Validation Run (Train on Train, Validate on Val)
    # ==================================================================================
    print("\n==== Starting Validation Run ====")

    # Create a temporary working directory for validation to isolate cache
    val_working_dir = os.path.join(config.working_dir, "validation_run")
    os.makedirs(val_working_dir, exist_ok=True)

    # Create a dummy empty validation file.
    # prepare_train_features merges train + val. We want only train.
    # By providing an empty val file (header only), the merge results in just train data.
    dummy_val_path = os.path.join(val_working_dir, "dummy_val.csv")
    train_header = pd.read_csv(config.train_path, nrows=0)
    train_header.to_csv(dummy_val_path, index=False)

    # Modify config for validation run
    val_config = Config()
    val_config.working_dir = val_working_dir
    val_config.val_path = dummy_val_path  # Trick: train + empty = train
    val_config.epochs = 7  # Sufficient for validation check on small data

    # Prepare Training Data (Only Train)
    print("Preparing validation training data...")
    train_dataset = prepare_train_features(
        val_config, tokenizer, load_cached_data=False
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=val_config.batch_size,
        shuffle=True,
        num_workers=val_config.num_workers,
    )

    # Initialize Model for Validation
    model = XLMRobertaForQA(config.model_name).to(device)

    # Optimizer & Scheduler
    optimizer_params = get_optimizer_params(model, val_config)
    optimizer = torch.optim.AdamW(optimizer_params)
    num_train_steps = len(train_loader) * val_config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_train_steps),
        num_training_steps=num_train_steps,
    )

    # Train Loop
    print(f"Training validation model for {val_config.epochs} epochs...")
    for epoch in range(val_config.epochs):
        loss = train_fn(train_loader, model, optimizer, device, val_config, scheduler)
        # print(f"Validation Epoch {epoch+1} Loss: {loss:.4f}")

    # Validation Inference
    print("Performing validation inference...")
    # We need to process real val.csv as if it were test data to get features
    # Point temp config test_path to real val_path
    temp_test_config = Config()
    temp_test_config.test_path = config.val_path
    temp_test_config.debug = False

    val_eval_dataset, val_eval_features = prepare_test_features(
        temp_test_config, tokenizer
    )
    val_eval_loader = torch.utils.data.DataLoader(
        val_eval_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
    )

    start_preds, end_preds, relevance_preds = predict_fn(val_eval_loader, model, device)

    # Reconstruct Answers and Compute Metric
    predictions = {}
    best_scores = {}

    # Load val df to get texts and ground truth
    val_df = pd.read_csv(config.val_path)
    val_id_to_context = dict(zip(val_df["id"], val_df["context"]))

    for idx, feature in enumerate(val_eval_features):
        example_id = feature["example_id"]
        context = feature["context"]
        offsets = feature["offset_mapping"]

        s_logit = start_preds[idx]
        e_logit = end_preds[idx]
        r_logit = relevance_preds[idx]

        # Decoding Strategy: Gated Score = (Start + End) + Relevance
        top_starts = np.argsort(s_logit)[-20:]
        top_ends = np.argsort(e_logit)[-20:]

        best_span_score = -1e9
        best_s, best_e = 0, 0
        max_len = 30

        for s in top_starts:
            for e in top_ends:
                if e >= s and (e - s) < max_len:
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    score = s_logit[s] + e_logit[e]
                    if score > best_span_score:
                        best_span_score = score
                        best_s = s
                        best_e = e

        window_score = best_span_score + r_logit

        if example_id not in best_scores or window_score > best_scores[example_id]:
            best_scores[example_id] = window_score

            if offsets[best_s] is None or offsets[best_e] is None:
                pred_text = ""
            else:
                start_char = offsets[best_s][0]
                end_char = offsets[best_e][1]
                pred_text = context[start_char:end_char]

            predictions[example_id] = pred_text

    # Compute Metric
    total_jaccard = 0
    count = 0
    gt_dict = dict(zip(val_df["id"], val_df["answer_text"]))

    error_magnitudes = []
    feature_lengths = []

    for eid, gt in gt_dict.items():
        pred = predictions.get(eid, "")
        score = jaccard(gt, pred)
        total_jaccard += score
        count += 1

        # Failure Analysis Data
        error_magnitudes.append(1.0 - score)
        feature_lengths.append(len(val_id_to_context[eid]))

    final_metric = total_jaccard / count if count > 0 else 0.0
    print(f"Final Validation Metric: {final_metric:.10f}")

    # Failure Analysis
    if len(error_magnitudes) > 1:
        correlation = np.corrcoef(error_magnitudes, feature_lengths)[0, 1]
        print(
            f"Failure Analysis: Correlation between Error Magnitude and Context Length: {correlation:.4f}"
        )
    else:
        print("Failure Analysis: Not enough samples for correlation.")

    # Clean up validation resources
    del model, optimizer, scheduler, train_loader, val_eval_loader
    torch.cuda.empty_cache()

    # ==================================================================================
    # PART 2: Submission Run (Full Data, 5 Seeds)
    # ==================================================================================
    THRESHOLD = 0.60025

    if final_metric > THRESHOLD:
        print(
            f"\nMetric {final_metric:.5f} > {THRESHOLD}. Starting Full Training with 5 Seeds..."
        )

        # Prepare Full Dataset (Train + Val)
        # Using original config which points to real train and val paths -> prepare_train_features merges them
        full_train_dataset = prepare_train_features(
            config, tokenizer, load_cached_data=False
        )

        # Prepare Test Data
        test_dataset, test_features = prepare_test_features(config, tokenizer)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=config.num_workers,
        )

        # Accumulators for Ensemble
        avg_start_logits = None
        avg_end_logits = None
        avg_relevance_logits = None

        for seed_idx, seed in enumerate(config.seeds):
            print(f"Training Seed {seed} ({seed_idx+1}/{config.n_seeds})...")
            set_seed(seed)

            # Initialize Model
            model = XLMRobertaForQA(config.model_name).to(device)

            # Optimizer & Scheduler
            optimizer_params = get_optimizer_params(model, config)
            optimizer = torch.optim.AdamW(optimizer_params)

            train_loader = torch.utils.data.DataLoader(
                full_train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
            )

            num_train_steps = len(train_loader) * config.epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * num_train_steps),
                num_training_steps=num_train_steps,
            )

            # Train
            for epoch in range(config.epochs):
                loss = train_fn(
                    train_loader, model, optimizer, device, config, scheduler
                )

            # Predict
            s_logits, e_logits, r_logits = predict_fn(test_loader, model, device)

            if avg_start_logits is None:
                avg_start_logits = s_logits
                avg_end_logits = e_logits
                avg_relevance_logits = r_logits
            else:
                avg_start_logits += s_logits
                avg_end_logits += e_logits
                avg_relevance_logits += r_logits

            # Cleanup
            del model, optimizer, scheduler, train_loader
            torch.cuda.empty_cache()

        # Average Logits
        avg_start_logits /= config.n_seeds
        avg_end_logits /= config.n_seeds
        avg_relevance_logits /= config.n_seeds

        print("Generating submission...")

        test_best_scores = {}
        test_final_preds = {}

        test_df = pd.read_csv(config.test_path)
        all_test_ids = test_df["id"].tolist()

        for idx, feature in enumerate(test_features):
            example_id = feature["example_id"]
            context = feature["context"]
            offsets = feature["offset_mapping"]

            s_logit = avg_start_logits[idx]
            e_logit = avg_end_logits[idx]
            r_logit = avg_relevance_logits[idx]

            top_starts = np.argsort(s_logit)[-20:]
            top_ends = np.argsort(e_logit)[-20:]

            best_span_score = -1e9
            best_s, best_e = 0, 0
            max_len = 30

            for s in top_starts:
                for e in top_ends:
                    if e >= s and (e - s) < max_len:
                        if offsets[s] is None or offsets[e] is None:
                            continue
                        score = s_logit[s] + e_logit[e]
                        if score > best_span_score:
                            best_span_score = score
                            best_s = s
                            best_e = e

            window_score = best_span_score + r_logit

            if (
                example_id not in test_best_scores
                or window_score > test_best_scores[example_id]
            ):
                test_best_scores[example_id] = window_score

                if offsets[best_s] is None or offsets[best_e] is None:
                    pred_text = ""
                else:
                    start_char = offsets[best_s][0]
                    end_char = offsets[best_e][1]
                    pred_text = context[start_char:end_char]

                # Escape double quotes for CSV
                pred_text = pred_text.replace('"', '""')
                test_final_preds[example_id] = f'"{pred_text}"'

        # Write Submission
        with open(config.submission_path, "w", encoding="utf-8") as f:
            f.write("id,PredictionString\n")
            for eid in all_test_ids:
                pred = test_final_preds.get(eid, '""')
                f.write(f"{eid},{pred}\n")

        print(f"Submission saved to {config.submission_path}")
    else:
        print(
            f"Validation metric {final_metric:.5f} did not pass threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
