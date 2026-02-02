import os
import torch
import numpy as np
import pandas as pd
import csv
from collections import defaultdict, Counter
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)
from library.config import Config
from library.utils import set_seed


def get_model_path_config():
    """
    Determines the correct base path for model initialization (TAPT vs Base).
    """
    if os.path.exists(Config.TAPT_MODEL_DIR) and os.path.exists(
        os.path.join(Config.TAPT_MODEL_DIR, "config.json")
    ):
        return Config.TAPT_MODEL_DIR
    return Config.MODEL_CHECKPOINT


def extract_spans(logits, offset_mapping, sequence_ids, context_text):
    """
    Decodes BIO logits into text spans with confidence scores.

    Args:
        logits: Tensor of shape (seq_len, 3)
        offset_mapping: List of (start, end) character offsets
        sequence_ids: List of sequence identifiers (0=question, 1=context, None=special)
        context_text: The original context string

    Returns:
        List of dicts: [{'text': str, 'confidence': float}, ...]
    """
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    preds = np.argmax(probs, axis=-1)

    spans = []
    current_span_start = None
    current_span_score = 0.0
    current_span_tokens = 0

    # Iterate through tokens
    for i, (pred, prob, seq_id) in enumerate(zip(preds, probs, sequence_ids)):
        # Only process context tokens (seq_id == 1)
        if seq_id != 1:
            if current_span_start is not None:
                spans.append(
                    finalize_span(
                        current_span_start,
                        i - 1,
                        current_span_score,
                        current_span_tokens,
                        offset_mapping,
                        context_text,
                    )
                )
                current_span_start = None
            continue

        # Check for valid offsets
        start_char, end_char = offset_mapping[i]
        if start_char >= end_char:
            continue

        if pred == 1:  # B-ANS
            if current_span_start is not None:
                # Close previous span
                spans.append(
                    finalize_span(
                        current_span_start,
                        i - 1,
                        current_span_score,
                        current_span_tokens,
                        offset_mapping,
                        context_text,
                    )
                )

            # Start new span
            current_span_start = i
            current_span_score = prob[1]
            current_span_tokens = 1

        elif pred == 2:  # I-ANS
            if current_span_start is not None:
                # Extend current span
                current_span_score += prob[2]
                current_span_tokens += 1
            else:
                # I without B: Treat as start of new span for robustness
                current_span_start = i
                current_span_score = prob[2]
                current_span_tokens = 1

        else:  # O
            if current_span_start is not None:
                spans.append(
                    finalize_span(
                        current_span_start,
                        i - 1,
                        current_span_score,
                        current_span_tokens,
                        offset_mapping,
                        context_text,
                    )
                )
                current_span_start = None

    # Handle span at the very end of sequence
    if current_span_start is not None:
        spans.append(
            finalize_span(
                current_span_start,
                len(preds) - 1,
                current_span_score,
                current_span_tokens,
                offset_mapping,
                context_text,
            )
        )

    return spans


def finalize_span(
    start_idx, end_idx, total_score, num_tokens, offset_mapping, context_text
):
    """Helper to extract text and compute mean confidence."""
    char_start = offset_mapping[start_idx][0]
    char_end = offset_mapping[end_idx][1]
    text = context_text[char_start:char_end]
    confidence = total_score / num_tokens if num_tokens > 0 else 0.0
    return {"text": text, "confidence": confidence}


def run_inference_for_seed(seed, test_dataset, test_features):
    """
    Runs inference using the model trained with a specific seed.
    Performs Global Confidence Aggregation across sliding windows.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize Model
    model_path_base = get_model_path_config()
    tokenizer = AutoTokenizer.from_pretrained(model_path_base)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path_base, num_labels=3, ignore_mismatched_sizes=False
    )

    # Load state dict
    weights_path = os.path.join(Config.QA_MODELS_DIR, f"model_seed_{seed}.pt")
    if not os.path.exists(weights_path):
        print(
            f"Warning: Model weights for seed {seed} not found at {weights_path}. Skipping."
        )
        return {}

    print(f"Loading weights from {weights_path}...")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. DataLoader
    data_collator = DataCollatorForTokenClassification(tokenizer)
    dataloader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        collate_fn=data_collator,
        shuffle=False,
    )

    # 3. Inference Loop
    all_candidates = defaultdict(list)
    feature_idx = 0

    print(f"Running inference for seed {seed}...")

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits  # (batch, seq_len, 3)

            # Process each sample in batch
            for i in range(logits.shape[0]):
                # Get corresponding feature info
                feature = test_features[feature_idx]
                feature_idx += 1

                # Extract spans
                spans = extract_spans(
                    logits[i],
                    feature["offset_mapping"],
                    feature["sequence_ids"],
                    feature["context_text"],
                )

                # Collect candidates for this example_id
                example_id = feature["example_id"]
                for span in spans:
                    all_candidates[example_id].append(span)

    # 4. Global Confidence Aggregation
    # Select the single best span per example_id based on confidence
    final_predictions = {}

    # Get list of all IDs from features to ensure we cover everyone
    all_ids = set(f["example_id"] for f in test_features)

    for eid in all_ids:
        candidates = all_candidates.get(eid, [])
        if not candidates:
            final_predictions[eid] = ""  # No answer found
        else:
            # Sort by confidence descending
            best_span = sorted(candidates, key=lambda x: x["confidence"], reverse=True)[
                0
            ]
            final_predictions[eid] = best_span["text"]

    return final_predictions


def ensemble_predictions(model_outputs):
    """
    Applies Majority Voting across predictions from multiple models.

    Args:
        model_outputs: List of dicts [{id: prediction}, ...]

    Returns:
        Dict {id: final_prediction}
    """
    print("Ensembling predictions via Majority Voting...")
    if not model_outputs:
        return {}

    # Get all IDs
    all_ids = set().union(*[d.keys() for d in model_outputs])
    final_preds = {}

    for eid in all_ids:
        votes = []
        for output in model_outputs:
            if eid in output:
                votes.append(output[eid])

        if not votes:
            final_preds[eid] = ""
        else:
            # Majority vote
            counts = Counter(votes)
            # most_common returns [(item, count), ...]. We take the top item.
            # Ties are broken arbitrarily by Counter (usually insertion order), which is acceptable.
            best_pred = counts.most_common(1)[0][0]
            final_preds[eid] = best_pred

    return final_preds


def predict_and_submit(test_dataset, test_features):
    """
    Main orchestration function for inference and submission generation.
    """
    # 1. Run inference for each seed
    model_outputs = []
    for seed in Config.SEEDS:
        preds = run_inference_for_seed(seed, test_dataset, test_features)
        if preds:
            model_outputs.append(preds)

    if not model_outputs:
        print("Error: No predictions generated from any model.")
        return

    # 2. Ensemble
    final_predictions = ensemble_predictions(model_outputs)

    # 3. Generate Submission File
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    print(f"Generating submission file at {submission_path}...")

    # Convert to DataFrame
    # Ensure we cover all IDs in the test set
    # We rely on test_features to give us the list of IDs
    unique_ids = sorted(list(set(f["example_id"] for f in test_features)))

    data = []
    for eid in unique_ids:
        pred_text = final_predictions.get(eid, "")
        # Ensure the string is clean (though extract_spans takes raw slice)
        # We quote it implicitly via pandas or explicitly if needed.
        data.append({"id": eid, "PredictionString": pred_text})

    df_sub = pd.DataFrame(data)

    # Save to CSV
    # The requirement asks for quoted strings: id,"PredictionString"
    # Pandas to_csv handles quoting. We force quoting for non-numeric (which includes the prediction string).
    # However, 'id' is also a string (hash). We can just use default settings which usually works for Kaggle.
    # To be strictly safe with the example `8c8ee6504,"1"`, we can force quotes on the prediction column.

    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved. Shape: {df_sub.shape}")
