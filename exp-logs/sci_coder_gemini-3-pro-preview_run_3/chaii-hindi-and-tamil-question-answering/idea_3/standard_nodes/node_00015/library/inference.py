import os
import torch
import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict, Optional

from library.config import Config
from library.utils import clean_text
from library.data_manager import get_test_dataloader
from library.model_factory import get_qa_model, get_tokenizer


def decode_span(
    tags: np.ndarray, offset_mapping: np.ndarray, context: str
) -> Optional[str]:
    """
    Decodes a sequence of tags into a text span using the greedy first-match strategy.

    Args:
        tags: Numpy array of shape (seq_len,) containing predicted class indices.
        offset_mapping: Numpy array of shape (seq_len, 2) containing (start, end) char offsets.
        context: The original context string.

    Returns:
        The extracted answer string, or None if no valid answer is found.
    """
    # Look for the first B-ANS (Label 1)
    start_indices = np.where(tags == 1)[0]

    if len(start_indices) == 0:
        return None

    # Greedy approach: take the first B-ANS found in the sequence
    start_idx = start_indices[0]

    # Find the end of the span.
    # The span continues as long as we see I-ANS (Label 2)
    end_idx = start_idx
    for i in range(start_idx + 1, len(tags)):
        if tags[i] == 2:
            end_idx = i
        else:
            break

    # Map token indices to character offsets
    # offset_mapping[i] is [char_start, char_end]
    char_start = offset_mapping[start_idx][0]
    char_end = offset_mapping[end_idx][1]

    # Validation checks
    # 1. Check for special tokens (often mapped to 0,0)
    if char_start == 0 and char_end == 0:
        return None

    # 2. Check for valid length
    if char_end <= char_start:
        return None

    # Extract text from the original context
    answer = context[char_start:char_end]
    return answer


def predict_fold(
    fold_idx: int,
    dataloader: torch.utils.data.DataLoader,
    model_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Performs inference for a single fold model on the test set.

    Args:
        fold_idx: The fold index (used for logging/defaults).
        dataloader: The test dataloader.
        model_path: Path to the specific model checkpoint.

    Returns:
        A dictionary mapping example_id to predicted answer string.
    """
    device = Config.DEVICE
    if model_path is None:
        model_path = os.path.join(
            Config.QA_MODEL_OUTPUT_DIR, f"model_fold_{fold_idx}.pt"
        )

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Warning: Model checkpoint not found at {model_path}. Skipping.")
        return {}

    print(f"Loading model from {model_path}...")
    model = get_qa_model(model_path=None)  # Initialize architecture
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    fold_predictions = {}

    # Iterate over batches
    # Note: Test dataloader is not shuffled, so windows for a document are contiguous.
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (batch_size, seq_len, num_labels)

            # Get predictions (Argmax)
            # We use simple argmax as per the strategy
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            # Metadata for decoding
            example_ids = batch["example_id"]
            contexts = batch["context"]
            offset_mappings = batch["offset_mapping"].numpy()

            batch_size = len(example_ids)

            for i in range(batch_size):
                eid = example_ids[i]

                # Greedy First-Match Strategy:
                # If we already found an answer for this document in a previous window (or this batch), skip.
                if eid in fold_predictions:
                    continue

                # Decode
                pred_tags = preds[i]
                offsets = offset_mappings[i]
                ctx = contexts[i]

                answer = decode_span(pred_tags, offsets, ctx)

                if answer is not None:
                    # Clean and store
                    fold_predictions[eid] = clean_text(answer)

    return fold_predictions


def perform_majority_voting(
    all_predictions: List[Dict[str, str]], all_ids: List[str]
) -> Dict[str, str]:
    """
    Aggregates predictions from multiple folds using Majority Voting.

    Args:
        all_predictions: List of dictionaries (one per fold) mapping id -> answer.
        all_ids: List of all unique IDs in the test set.

    Returns:
        Dictionary mapping id -> final answer string.
    """
    final_predictions = {}

    for eid in all_ids:
        votes = []
        for fold_preds in all_predictions:
            if eid in fold_preds:
                votes.append(fold_preds[eid])

        if not votes:
            # No model predicted an answer, default to empty string
            final_predictions[eid] = ""
        else:
            # Majority vote
            # Counter.most_common returns [(element, count), ...]
            # We take the element with highest count.
            most_common = Counter(votes).most_common(1)[0][0]
            final_predictions[eid] = most_common

    return final_predictions


def generate_submission():
    """
    Main inference routine.
    Loads data, runs inference for all folds, performs ensemble, and saves submission.
    """
    print("Starting Submission Generation...")

    # 1. Setup
    tokenizer = get_tokenizer()
    # load_cached_data=True ensures we use the cache if available
    test_loader = get_test_dataloader(tokenizer, load_cached_data=True)

    # Get list of all test IDs to ensure completeness
    df_test = pd.read_csv(Config.TEST_META_PATH)
    all_test_ids = df_test["id"].unique().tolist()

    # 2. Run Inference for each fold
    all_fold_predictions = []

    # Iterate through defined number of folds
    for fold in range(Config.N_FOLDS):
        print(f"\n--- Inference Fold {fold + 1}/{Config.N_FOLDS} ---")
        preds = predict_fold(fold, test_loader)
        if preds:
            all_fold_predictions.append(preds)

    if not all_fold_predictions:
        print("Error: No predictions generated from any fold. Check model paths.")
        # Create empty submission to avoid total failure
        final_predictions_map = {eid: "" for eid in all_test_ids}
    else:
        # 3. Ensemble (Majority Voting)
        print("\nPerforming Majority Voting Ensemble...")
        final_predictions_map = perform_majority_voting(
            all_fold_predictions, all_test_ids
        )

    # 4. Create Submission DataFrame
    print("Creating submission file...")
    submission_df = pd.DataFrame(
        {
            "id": list(final_predictions_map.keys()),
            "PredictionString": list(final_predictions_map.values()),
        }
    )

    # Ensure column order
    submission_df = submission_df[["id", "PredictionString"]]

    # 5. Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total predictions: {len(submission_df)}")
