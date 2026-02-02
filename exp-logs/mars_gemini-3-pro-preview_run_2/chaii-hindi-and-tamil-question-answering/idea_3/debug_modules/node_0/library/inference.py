import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

from library.config import Config
from library.model import get_model, get_tokenizer
from library.dataset import get_data, QADataset
from library.engine import get_predictions


def generate_submission():
    """
    Main inference function to generate the submission file.
    Loads the test data, runs the ensemble of trained models,
    aggregates predictions, and saves the result to submission.csv.
    """
    print("Starting submission generation...")

    # 1. Setup
    device = Config.DEVICE
    tokenizer = get_tokenizer()

    # 2. Load Metadata (for raw context text)
    # We need this to extract the actual answer string from character offsets
    if not os.path.exists(Config.TEST_CSV):
        raise FileNotFoundError(f"Test metadata file not found at {Config.TEST_CSV}")

    test_df = pd.read_csv(Config.TEST_CSV)
    # Create a mapping from example_id to context text
    id_to_context = dict(zip(test_df["id"], test_df["context"]))

    # 3. Load Processed Features
    # get_data handles caching automatically
    # We use load_cached_data=True to use existing cache if available, or process if not
    test_features = get_data(tokenizer, split="test", load_cached_data=True)

    # 4. Prepare DataLoader
    test_dataset = QADataset(test_features)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 5. Ensemble Prediction
    # We will accumulate logits from all available folds
    avg_start_logits = None
    avg_end_logits = None
    models_used = 0

    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"best_model_fold_{fold}.pth")

        if not os.path.exists(model_path):
            print(
                f"Warning: Model checkpoint for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        print(f"Predicting with model fold {fold}...")

        # Load model architecture and weights
        model = get_model()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Get predictions
        start_logits, end_logits = get_predictions(model, test_loader, device)

        # Accumulate
        if avg_start_logits is None:
            avg_start_logits = start_logits
            avg_end_logits = end_logits
        else:
            avg_start_logits += start_logits
            avg_end_logits += end_logits

        models_used += 1

        # Clean up to free memory
        del model, state_dict
        torch.cuda.empty_cache()
        gc.collect()

    # Handle case where no models are found (e.g., debugging or dry run)
    if models_used == 0:
        print("Error: No trained models found. Generating dummy predictions.")
        # Initialize with zeros
        num_samples = len(test_features)
        seq_len = Config.MAX_LENGTH
        avg_start_logits = np.zeros((num_samples, seq_len))
        avg_end_logits = np.zeros((num_samples, seq_len))
        models_used = 1  # Prevent division by zero

    # Average the logits
    avg_start_logits /= models_used
    avg_end_logits /= models_used

    # 6. Decode Predictions
    print("Decoding predictions into answer strings...")

    # Dictionary to store the best prediction for each example_id
    # Structure: example_id -> (score, answer_text)
    final_predictions = {}

    # Extract columns for faster iteration
    example_ids = test_features["example_id"].values
    offset_mappings = test_features["offset_mapping"].values
    sequence_ids_list = test_features["sequence_ids"].values

    # Iterate over all feature windows
    for i in range(len(test_features)):
        example_id = example_ids[i]
        start_logits = avg_start_logits[i]
        end_logits = avg_end_logits[i]
        offsets = offset_mappings[i]
        seq_ids = sequence_ids_list[i]

        # Retrieve context text
        context_text = id_to_context.get(example_id, "")

        # Identify valid context tokens (sequence_id == 1)
        # seq_ids is a list/array where 1 indicates context
        # We need to handle potential None values
        context_indices = [idx for idx, s in enumerate(seq_ids) if s == 1]

        if not context_indices:
            continue

        min_ctx_idx = context_indices[0]
        max_ctx_idx = context_indices[-1]

        # Get top-K start and end indices
        # We use argsort and take the last N_BEST_SIZE indices
        start_candidates = np.argsort(start_logits)[-Config.N_BEST_SIZE :]
        end_candidates = np.argsort(end_logits)[-Config.N_BEST_SIZE :]

        best_feature_score = -float("inf")
        best_feature_answer = ""

        # Iterate through candidates
        for start_idx in start_candidates:
            # Check if start is within context
            if start_idx < min_ctx_idx or start_idx > max_ctx_idx:
                continue

            for end_idx in end_candidates:
                # Check if end is within context
                if end_idx < min_ctx_idx or end_idx > max_ctx_idx:
                    continue

                # Basic validity checks
                if end_idx < start_idx:
                    continue

                if end_idx - start_idx + 1 > Config.MAX_ANSWER_LENGTH:
                    continue

                # Calculate score
                score = start_logits[start_idx] + end_logits[end_idx]

                if score > best_feature_score:
                    best_feature_score = score

                    # Extract text using offsets
                    try:
                        # offsets[i] is [start_char, end_char]
                        char_start = offsets[start_idx][0]
                        char_end = offsets[end_idx][1]
                        best_feature_answer = context_text[char_start:char_end]
                    except Exception:
                        continue

        # Update the global best for this example_id
        # We aggregate across multiple sliding windows for the same example
        if (
            example_id not in final_predictions
            or best_feature_score > final_predictions[example_id][0]
        ):
            final_predictions[example_id] = (best_feature_score, best_feature_answer)

    # 7. Create Submission DataFrame
    print("Formatting submission file...")
    submission_data = []

    # Ensure we output predictions for all IDs in the test set
    all_test_ids = test_df["id"].unique()

    for eid in all_test_ids:
        if eid in final_predictions:
            pred_string = final_predictions[eid][1]
        else:
            # Fallback for IDs with no valid predictions
            pred_string = ""

        submission_data.append({"id": eid, "PredictionString": pred_string})

    submission_df = pd.DataFrame(submission_data)

    # Save
    submission_df.to_csv(Config.SUBMISSION_CSV, index=False)
    print(f"Submission saved successfully to {Config.SUBMISSION_CSV}")
    print(f"Total predictions: {len(submission_df)}")
