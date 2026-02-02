import os
import csv
import torch
import pandas as pd
from collections import defaultdict, Counter
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from library.config import Config
from library.utils import set_seed
from library.data_loader import QADataset, collate_fn, prepare_features
from library.model_factory import get_model


def extract_candidates(input_ids, logits, tokenizer):
    """
    Extracts candidate spans from token classification logits.
    Returns a list of tuples: (score, text)
    """
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)
    # Get predicted classes (0: O, 1: B-ANS, 2: I-ANS)
    preds = torch.argmax(probs, dim=-1)

    candidates = []

    # Find all B-ANS (label 1) indices
    starts = (preds == 1).nonzero(as_tuple=True)[0]

    for start_idx in starts:
        start_idx = start_idx.item()
        end_idx = start_idx

        # Extend span while label is I-ANS (label 2)
        while end_idx + 1 < len(preds) and preds[end_idx + 1] == 2:
            end_idx += 1

        # Calculate confidence score (mean probability of the predicted labels in the span)
        span_probs = []
        for i in range(start_idx, end_idx + 1):
            pred_class = preds[i].item()
            span_probs.append(probs[i, pred_class].item())

        score = sum(span_probs) / len(span_probs)

        # Decode text
        span_ids = input_ids[start_idx : end_idx + 1]
        text = tokenizer.decode(span_ids, skip_special_tokens=True).strip()

        if text:  # Filter empty strings
            candidates.append((score, text))

    return candidates


def predict_single_model(config, seed, test_loader, tokenizer):
    """
    Runs inference for a single model seed.
    Returns a dictionary mapping example_id to the best prediction string.
    """
    set_seed(seed)

    # Initialize model
    model = get_model(config)

    # Load QA fine-tuned weights
    model_path = os.path.join(config.model_dir, f"model_seed_{seed}.pt")
    if not os.path.exists(model_path):
        print(
            f"Warning: Model checkpoint not found at {model_path}. Using initialized weights."
        )
    else:
        print(f"Loading QA weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=config.device))

    model.eval()

    # Store all candidates per example_id
    # example_id -> list of (score, text)
    all_candidates = defaultdict(list)

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            example_ids = batch["example_ids"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (batch, seq_len, 3)

            # Process each sample in the batch
            for i in range(len(example_ids)):
                eid = example_ids[i]
                seq_logits = logits[i]
                seq_input_ids = input_ids[i]

                candidates = extract_candidates(seq_input_ids, seq_logits, tokenizer)
                all_candidates[eid].extend(candidates)

    # Select best candidate per example_id
    final_predictions = {}

    # We iterate over all IDs found in the candidates
    # Note: If an ID produced no candidates across all windows, it will be missing here
    # but handled in the main loop or filled with empty string below.
    for eid, candidates in all_candidates.items():
        if not candidates:
            final_predictions[eid] = ""
        else:
            # Sort by score descending
            best_candidate = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
            final_predictions[eid] = best_candidate[1]

    return final_predictions


def run_inference(config: Config):
    """
    Main inference function.
    1. Prepares test data.
    2. Runs predictions for all seeds.
    3. Ensembles using Majority Voting.
    4. Saves submission.csv.
    """
    print("Starting Inference Pipeline...")

    # 1. Prepare Data
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Ensure test features are prepared
    test_df = prepare_features(config, tokenizer, split="test", load_cached_data=True)

    test_dataset = QADataset(test_df, is_test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # 2. Collect Predictions from all seeds
    seed_predictions = []  # List of dicts

    for seed in config.seeds:
        print(f"Running inference for seed {seed}...")
        preds = predict_single_model(config, seed, test_loader, tokenizer)
        seed_predictions.append(preds)

    # 3. Ensemble (Majority Voting)
    print("Ensembling predictions...")

    # Get all unique IDs from the test dataset
    # We use the original dataframe to ensure we have the correct order and all IDs
    unique_ids = test_df["example_id"].unique()

    final_submission = []

    for eid in unique_ids:
        votes = []
        for pred_dict in seed_predictions:
            # If a model didn't output anything for this ID (e.g. no candidates found), use empty string
            if eid in pred_dict:
                votes.append(pred_dict[eid])
            else:
                votes.append("")

        if not votes:
            final_pred = ""
        else:
            # Majority vote
            # Counter.most_common returns [(item, count), ...]
            final_pred = Counter(votes).most_common(1)[0][0]

        final_submission.append({"id": eid, "PredictionString": final_pred})

    # 4. Save Submission
    submission_df = pd.DataFrame(final_submission)

    # Ensure correct column order
    submission_df = submission_df[["id", "PredictionString"]]

    output_path = os.path.join(config.submission_dir, "submission.csv")
    print(f"Saving submission to {output_path}")

    # Use QUOTE_NONNUMERIC to ensure string fields are quoted, matching the sample format requirements
    submission_df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

    print("Inference completed.")
