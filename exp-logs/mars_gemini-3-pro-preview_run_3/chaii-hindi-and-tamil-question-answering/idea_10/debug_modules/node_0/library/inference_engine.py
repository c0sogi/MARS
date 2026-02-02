import os
import torch
import pandas as pd
import numpy as np
import csv
from collections import Counter
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

from library.config import Config
from library.utils import set_seed, clean_text
from library.data_processing import get_qa_data, qa_collate_fn
from library.qa_engine import extract_spans


def predict_single_model(model, test_loader, tokenizer, device):
    """
    Generates predictions using a single model instance.
    Implements Global Confidence Aggregation: collects all spans for a document
    across multiple sliding windows and selects the one with the highest score.

    Args:
        model (torch.nn.Module): The loaded model.
        test_loader (DataLoader): DataLoader for the test set.
        tokenizer (PreTrainedTokenizer): Tokenizer for decoding spans.
        device (torch.device): Device to run inference on.

    Returns:
        dict: A dictionary mapping example_id (str) to the predicted answer string.
    """
    model.eval()

    # Dictionary to store all candidate spans for each document ID
    # Structure: {example_id: [(score, prediction_string), ...]}
    doc_candidates = {}

    # Initialize entries for all IDs in the loader to ensure every ID has a result
    # We iterate once to get all IDs (a bit inefficient but ensures safety) or handle dynamically.
    # Dynamic handling is better for memory.

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            example_ids = batch["example_id"]  # List of strings

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            for i in range(len(example_ids)):
                eid = example_ids[i]

                # Extract candidate spans from this window
                # extract_spans returns list of (score, text)
                spans = extract_spans(input_ids[i], logits[i], tokenizer)

                if eid not in doc_candidates:
                    doc_candidates[eid] = []

                doc_candidates[eid].extend(spans)

    # Aggregation: Select best span per document
    final_predictions = {}

    # We need to ensure we cover all IDs in the test set.
    # The loader covers all windows, so all IDs are visited.
    for eid, candidates in doc_candidates.items():
        if not candidates:
            # No span found (all O tags), predict empty string
            final_predictions[eid] = ""
        else:
            # Sort by score descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            # Pick the top one
            best_text = candidates[0][1]
            final_predictions[eid] = clean_text(best_text)

    return final_predictions


def ensemble_predictions(prediction_dicts):
    """
    Combines predictions from multiple models using Majority Voting.

    Args:
        prediction_dicts (list of dict): List of dictionaries, where each dict
                                         contains predictions from one model
                                         {id: prediction_string}.

    Returns:
        dict: Final ensemble predictions {id: prediction_string}.
    """
    if not prediction_dicts:
        return {}

    # Get all unique IDs
    all_ids = set().union(*[d.keys() for d in prediction_dicts])

    final_output = {}

    for eid in all_ids:
        votes = []
        for p_dict in prediction_dicts:
            if eid in p_dict:
                votes.append(p_dict[eid])

        if not votes:
            final_output[eid] = ""
            continue

        # Majority Vote
        counter = Counter(votes)
        most_common = counter.most_common()

        # most_common is list of (element, count) sorted by count desc
        # If tie, most_common order depends on insertion order or implementation.
        # To break ties consistently, we can check if the top two have same count.

        winner = most_common[0][0]

        # Tie-breaking logic (optional):
        # If there is a strict tie (e.g., 3 models, 3 different answers),
        # we default to the answer from the first model (Seed 42 usually) as a heuristic.
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            # Tie detected. Fallback to first model's prediction.
            winner = prediction_dicts[0].get(eid, "")

        final_output[eid] = winner

    return final_output


def generate_submission():
    """
    Main function to generate the submission file.
    Loads models, runs inference, ensembles results, and saves to CSV.
    """
    print("Starting Submission Generation...")

    # 1. Setup Environment
    device = torch.device(Config.DEVICE)

    # 2. Determine Base Model Path (TAPT or Original)
    # We need the correct config/tokenizer used during training
    if os.path.exists(Config.TAPT_OUTPUT_DIR):
        print(f"Using TAPT model configuration from: {Config.TAPT_OUTPUT_DIR}")
        base_model_path = Config.TAPT_OUTPUT_DIR
    else:
        print(f"TAPT model not found. Using base checkpoint: {Config.MODEL_CHECKPOINT}")
        base_model_path = Config.MODEL_CHECKPOINT

    # 3. Load Tokenizer and Data
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # We only need the test set here
    print("Loading Test Data...")
    _, _, test_ds = get_qa_data(tokenizer, load_cached_data=True)

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Important: keep order deterministic (though we map by ID anyway)
        collate_fn=qa_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Run Inference for Each Seed
    all_model_predictions = []

    for seed in Config.SEED_LIST:
        model_weights_path = os.path.join(Config.QA_MODELS_DIR, f"model_seed_{seed}.pt")

        if not os.path.exists(model_weights_path):
            print(
                f"Warning: Model weights for seed {seed} not found at {model_weights_path}. Skipping."
            )
            continue

        print(f"Running inference for model seed {seed}...")

        # Initialize Architecture
        # Must match training initialization (num_labels=3)
        model = AutoModelForTokenClassification.from_pretrained(
            base_model_path, num_labels=3
        )

        # Load Weights
        state_dict = torch.load(model_weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        # Predict
        preds = predict_single_model(model, test_loader, tokenizer, device)
        all_model_predictions.append(preds)

        # Cleanup
        del model
        del state_dict
        torch.cuda.empty_cache()

    if not all_model_predictions:
        raise RuntimeError(
            "No models were found or successfully loaded. Cannot generate submission."
        )

    # 5. Ensemble
    print("Ensembling predictions...")
    final_preds_map = ensemble_predictions(all_model_predictions)

    # 6. Create Submission DataFrame
    # Ensure we output for all IDs in the test metadata
    test_meta_df = pd.read_csv(Config.TEST_META_PATH)
    submission_ids = test_meta_df["id"].unique()

    submission_data = []
    for eid in submission_ids:
        # Default to empty string if somehow missing
        pred_str = final_preds_map.get(eid, "")
        submission_data.append({"id": eid, "PredictionString": pred_str})

    submission_df = pd.DataFrame(submission_data)

    # 7. Save
    # Ensuring directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    # Using quoting=csv.QUOTE_NONNUMERIC to quote strings as per sample format style if needed,
    # but standard CSV is usually sufficient. The prompt example shows explicit quotes.
    # We will rely on pandas default but ensure strings are clean.
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print("Submission generation complete.")
