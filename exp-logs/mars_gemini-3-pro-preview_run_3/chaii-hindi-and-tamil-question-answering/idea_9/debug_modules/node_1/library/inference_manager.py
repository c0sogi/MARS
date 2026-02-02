import os
import torch
import pandas as pd
import numpy as np
import collections
from torch.utils.data import DataLoader

from library.configuration import Config
from library.utils import load_data
from library.dataset_factory import prepare_qa_data, qa_collate_fn, get_tokenizer
from library.qa_model import XLMRobertaForQA
from library.qa_trainer import extract_answer_spans


def predict_for_model(model_path, dataloader, tokenizer, device):
    """
    Runs inference for a single model checkpoint.
    Aggregates predictions using Global Confidence Aggregation.

    Args:
        model_path (str): Path to the .pt model checkpoint.
        dataloader (DataLoader): DataLoader for the test set.
        tokenizer: The tokenizer used for decoding.
        device (str): Device to run inference on.

    Returns:
        dict: A dictionary mapping example_id to the predicted answer string.
    """
    # Initialize model architecture
    model = XLMRobertaForQA()

    # Load weights
    # map_location ensures we can load on CPU if needed
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # Store all candidates for each example_id across all sliding windows
    # key: example_id, value: list of (score, text)
    all_candidates = collections.defaultdict(list)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0]

            batch_size = input_ids.size(0)
            for b in range(batch_size):
                eid = batch["example_id"][b]
                seq_ids = batch["sequence_ids"][b]

                # Extract spans using the logic from qa_trainer
                # This returns a list of (score, text) tuples for the current window
                spans = extract_answer_spans(
                    input_ids[b], logits[b], seq_ids, tokenizer
                )
                all_candidates[eid].extend(spans)

    # Global Aggregation: Select best span per document
    predictions = {}

    for eid, candidates in all_candidates.items():
        if not candidates:
            predictions[eid] = ""
        else:
            # Sort by score descending and pick the top one
            best_candidate = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
            predictions[eid] = best_candidate[1]

    return predictions


def majority_vote_ensemble(prediction_maps):
    """
    Combines predictions from multiple models using Majority Voting.

    Args:
        prediction_maps (list of dict): List of {example_id: prediction_string} maps.

    Returns:
        dict: {example_id: final_prediction_string}
    """
    if not prediction_maps:
        return {}

    # Gather all unique IDs from all maps
    all_ids = set()
    for p_map in prediction_maps:
        all_ids.update(p_map.keys())

    final_preds = {}

    for eid in all_ids:
        votes = []
        for p_map in prediction_maps:
            # Default to empty string if model missed this ID
            votes.append(p_map.get(eid, ""))

        # Count votes
        counter = collections.Counter(votes)

        # Get the most common prediction
        # most_common(1) returns [(value, count)]
        # If there is a tie, Counter returns the one encountered first.
        top_prediction = counter.most_common(1)[0][0]

        final_preds[eid] = top_prediction

    return final_preds


def generate_submission(model_paths, load_cached_data=True):
    """
    Generates the submission file by running inference with multiple models and ensembling.

    Args:
        model_paths (list of str): Paths to the trained model checkpoints (.pt files).
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        str: Path to the generated submission file.
    """
    device = Config.DEVICE
    tokenizer = get_tokenizer()

    print("Preparing Test Data for Inference...")
    # Prepare test dataset
    # This handles caching internally based on load_cached_data
    test_dataset = prepare_qa_data(
        tokenizer, split="test", load_cached_data=load_cached_data
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=qa_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load raw test metadata to ensure we have the complete list of IDs
    # and to preserve the order for the submission file
    df_test = load_data("test")
    all_test_ids = df_test["id"].astype(str).tolist()

    model_predictions = []

    # Run inference for each model
    for path in model_paths:
        if not os.path.exists(path):
            print(f"Warning: Model checkpoint not found at {path}. Skipping.")
            continue

        print(f"Running inference with model: {path}")
        preds = predict_for_model(path, test_loader, tokenizer, device)
        model_predictions.append(preds)

    if not model_predictions:
        raise RuntimeError(
            "No valid model predictions were generated. Cannot create submission."
        )

    # Ensemble
    print(f"Ensembling predictions from {len(model_predictions)} models...")
    final_prediction_map = majority_vote_ensemble(model_predictions)

    # Construct Submission DataFrame
    submission_rows = []
    for eid in all_test_ids:
        # Get prediction, default to empty string if missing
        pred_str = final_prediction_map.get(eid, "")

        # Ensure it's a string
        if not isinstance(pred_str, str):
            pred_str = str(pred_str)

        submission_rows.append({"id": eid, "PredictionString": pred_str})

    df_submission = pd.DataFrame(submission_rows)

    # Check shape
    print(f"Generated submission with {len(df_submission)} rows.")

    # Save to CSV
    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    df_submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")

    return Config.SUBMISSION_FILE
