import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
)
from library.config import Config
from library.utils import set_seed, jaccard
from library.data_processing import get_qa_data, qa_collate_fn, ID_TO_LABEL


def extract_spans(input_ids, logits, tokenizer):
    """
    Extracts candidate spans from a single window's logits.
    Returns a list of tuples: (score, prediction_string)
    """
    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=-1)
    confidences, predictions = torch.max(probs, dim=-1)

    # Convert to lists for iteration
    pred_ids = predictions.cpu().numpy()
    conf_scores = confidences.cpu().numpy()
    token_ids = input_ids.cpu().numpy()

    candidates = []

    # Find spans: B-ANS (1) followed by optional I-ANS (2)
    i = 0
    while i < len(pred_ids):
        if pred_ids[i] == 1:  # B-ANS
            start = i
            current = i + 1
            # Continue while I-ANS
            while current < len(pred_ids) and pred_ids[current] == 2:
                current += 1
            end = current

            # Decode span
            span_tokens = token_ids[start:end]
            pred_str = tokenizer.decode(span_tokens, skip_special_tokens=True).strip()

            # Calculate score (mean confidence of the span)
            span_score = np.mean(conf_scores[start:end])

            candidates.append((span_score, pred_str))

            i = end
        else:
            i += 1

    return candidates


def validate(model, val_loader, gt_df, tokenizer, device):
    """
    Runs validation on the validation set.
    Aggregates window-level predictions to document-level using max confidence.
    Computes Jaccard score.
    """
    model.eval()

    # Store candidates per example_id
    # structure: {example_id: [(score, text), ...]}
    doc_predictions = {}

    # Initialize entries for all IDs in val set to ensure coverage
    all_val_ids = gt_df["id"].unique()
    for vid in all_val_ids:
        doc_predictions[vid] = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            example_ids = batch["example_id"]  # List of strings

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            for i in range(len(example_ids)):
                eid = example_ids[i]
                # Extract all valid spans from this window
                spans = extract_spans(input_ids[i], logits[i], tokenizer)

                if eid in doc_predictions:
                    doc_predictions[eid].extend(spans)
                else:
                    # Should ideally be initialized, but handling edge cases
                    doc_predictions[eid] = spans

    # Aggregate and Score
    total_jaccard = 0.0
    count = 0

    # Map ID to Ground Truth
    id_to_gt = gt_df.set_index("id")["answer_text"].to_dict()

    for eid, candidates in doc_predictions.items():
        if eid not in id_to_gt:
            continue

        ground_truth = str(id_to_gt[eid])

        # Select best candidate
        best_pred = ""
        if candidates:
            # Sort by score descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_pred = candidates[0][1]

        # Compute Jaccard
        score = jaccard(ground_truth, best_pred)
        total_jaccard += score
        count += 1

    return total_jaccard / count if count > 0 else 0.0


def run_qa_training(tapt_model_dir=None):
    """
    Main function to run the QA training pipeline.

    Args:
        tapt_model_dir (str, optional): Path to the TAPT fine-tuned model.
                                        Defaults to Config.TAPT_OUTPUT_DIR.
    """
    # Determine model path
    if tapt_model_dir is None:
        tapt_model_dir = Config.TAPT_OUTPUT_DIR

    # Fallback to base model if TAPT output doesn't exist
    if not os.path.exists(tapt_model_dir):
        print(
            f"TAPT model not found at {tapt_model_dir}. Using base checkpoint: {Config.MODEL_CHECKPOINT}"
        )
        model_path = Config.MODEL_CHECKPOINT
    else:
        print(f"Initializing QA model from TAPT weights: {tapt_model_dir}")
        model_path = tapt_model_dir

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load Data
    print("Loading QA Datasets...")
    train_ds, val_ds, _ = get_qa_data(tokenizer, load_cached_data=True)

    # Load Validation Ground Truth for Metric Calculation
    val_meta_df = pd.read_csv(Config.VAL_META_PATH)

    # Prepare DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=qa_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=qa_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    device = torch.device(Config.DEVICE)

    # Loop over seeds for ensemble training
    for seed in Config.SEED_LIST:
        print(f"\n{'='*20}\nStarting Training for Seed {seed}\n{'='*20}")
        set_seed(seed)

        # Initialize Model (Token Classification: O, B-ANS, I-ANS -> 3 labels)
        model = AutoModelForTokenClassification.from_pretrained(
            model_path, num_labels=3
        )
        model.to(device)

        # Optimizer and Scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        num_training_steps = len(train_loader) * Config.EPOCHS
        num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Loss Function (Standard Cross Entropy, ignoring padding index -100 if necessary,
        # but here we handle masks manually or rely on HF default which ignores -100)
        # HF models compute loss internally if labels are provided.

        best_jaccard = 0.0
        best_model_path = os.path.join(Config.QA_MODELS_DIR, f"model_seed_{seed}.pt")

        for epoch in range(1, Config.EPOCHS + 1):
            # Training
            model.train()
            total_loss = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()

                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validation
            val_jaccard = validate(model, val_loader, val_meta_df, tokenizer, device)

            print(
                f"Epoch {epoch}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Jaccard: {val_jaccard:.6f}"
            )

            # Save Best Model
            if val_jaccard > best_jaccard:
                best_jaccard = val_jaccard
                # Save state dict
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> New Best Model Saved! (Jaccard: {best_jaccard:.6f})")

        # Clean up to save memory
        del model
        del optimizer
        del scheduler
        torch.cuda.empty_cache()

    print("\nQA Training Pipeline Completed.")
