import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.model import CustomXLMRoberta
from library.data import prepare_test_features
from library.utils import clean_text, format_prediction_string


def extract_answer_from_window(
    start_logits,
    end_logits,
    relevance_logit,
    sequence_ids,
    offset_mapping,
    context_text,
):
    """
    Finds the best span within a single window using Gated Scoring.
    Score = (Start_Logit + End_Logit) + Relevance_Logit

    Args:
        start_logits (np.array): Logits for start positions (seq_len,).
        end_logits (np.array): Logits for end positions (seq_len,).
        relevance_logit (float): Logit for answer presence.
        sequence_ids (list): List indicating sequence type (0=question, 1=context, -1=special).
        offset_mapping (list): List of [start_char, end_char] for each token.
        context_text (str): The original context text.

    Returns:
        tuple: (final_score, predicted_text)
    """
    # Constants
    MAX_ANSWER_LEN = 50  # Limit max answer length to avoid pathological long spans

    # Create masks for context tokens
    # sequence_ids: 0 for question, 1 for context, -1 for special tokens
    seq_ids = np.array(sequence_ids)
    context_mask = seq_ids == 1

    # If no context tokens (shouldn't happen with valid windows), return low score
    if not np.any(context_mask):
        return -1e9, ""

    # Mask non-context logits
    # We clone to avoid modifying the original array
    s_logits = start_logits.copy()
    e_logits = end_logits.copy()

    s_logits[~context_mask] = -1e9
    e_logits[~context_mask] = -1e9

    # Get valid indices for context
    valid_indices = np.where(context_mask)[0]
    start_idx = valid_indices[0]
    end_idx = valid_indices[-1]

    # Optimization: Slice logits to context area
    ctx_s_logits = s_logits[start_idx : end_idx + 1]
    ctx_e_logits = e_logits[start_idx : end_idx + 1]

    # Create score matrix: score[i, j] = start[i] + end[j]
    # i, j are relative to start_idx
    scores = ctx_s_logits[:, None] + ctx_e_logits[None, :]

    # Get indices for valid spans (start <= end)
    # np.triu_indices returns indices for the upper triangle
    r, c = np.triu_indices(scores.shape[0], k=0)

    # Filter by length constraint
    valid_len_mask = (c - r) < MAX_ANSWER_LEN
    r = r[valid_len_mask]
    c = c[valid_len_mask]

    if len(r) == 0:
        return -1e9, ""

    # Get max score among valid spans
    valid_scores = scores[r, c]
    best_idx = np.argmax(valid_scores)

    best_r = r[best_idx]
    best_c = c[best_idx]

    span_score = valid_scores[best_idx]

    # Map back to global token indices
    token_start = start_idx + best_r
    token_end = start_idx + best_c

    # Calculate Gated Score
    final_score = span_score + relevance_logit

    # Extract text using offset mapping
    try:
        char_start = offset_mapping[token_start][0]
        char_end = offset_mapping[token_end][1]
        predicted_text = context_text[char_start:char_end]
    except Exception:
        predicted_text = ""

    return final_score, predicted_text


def predict_and_ensemble():
    """
    Main function to run the inference pipeline.
    1. Loads test features.
    2. Runs inference with 5 seed models.
    3. Aggregates logits.
    4. Post-processes to find best answers.
    5. Saves submission.
    """
    device = Config.DEVICE

    # 1. Load Data
    # prepare_test_features handles caching and returns Dataset and Metadata DataFrame
    test_dataset, df_features = prepare_test_features(load_cached_data=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    num_samples = len(test_dataset)
    seq_len = Config.MAX_LEN

    # Accumulators for Ensemble Averaging
    total_start_logits = np.zeros((num_samples, seq_len), dtype=np.float32)
    total_end_logits = np.zeros((num_samples, seq_len), dtype=np.float32)
    total_relevance_logits = np.zeros((num_samples,), dtype=np.float32)

    # 2. Inference Loop over Seeds
    print(
        f"Running inference on {num_samples} windows with {len(Config.SEEDS)} models..."
    )

    models_found = 0
    for seed in Config.SEEDS:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"model_seed_{seed}.bin")
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found for seed {seed}. Skipping.")
            continue

        models_found += 1
        # Load Model
        model = CustomXLMRoberta(Config.MODEL_NAME)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()

        # Batch Prediction
        start_preds = []
        end_preds = []
        rel_preds = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                s_logits, e_logits, r_logits = model(input_ids, attention_mask)

                start_preds.append(s_logits.cpu().numpy())
                end_preds.append(e_logits.cpu().numpy())
                rel_preds.append(r_logits.cpu().numpy())

        # Concatenate and Add to totals
        total_start_logits += np.concatenate(start_preds, axis=0)
        total_end_logits += np.concatenate(end_preds, axis=0)
        total_relevance_logits += np.concatenate(rel_preds, axis=0)

        # Cleanup to save memory
        del model
        torch.cuda.empty_cache()

    if models_found == 0:
        print("Error: No models found. Cannot generate predictions.")
        return

    # Average Logits
    avg_start_logits = total_start_logits / models_found
    avg_end_logits = total_end_logits / models_found
    avg_relevance_logits = total_relevance_logits / models_found

    # 3. Post-Processing & Reconstruction
    print("Post-processing predictions...")

    results = {}  # example_id -> (best_score, best_text)

    # Iterate over the dataframe to access metadata aligned with logits indices
    # df_features has the same order as test_dataset
    for idx, row in df_features.iterrows():
        example_id = row["example_id"]
        sequence_ids = row["sequence_ids"]
        offset_mapping = row["offset_mapping"]
        context_text = row["context"]

        s_logits = avg_start_logits[idx]
        e_logits = avg_end_logits[idx]
        r_logit = avg_relevance_logits[idx]

        score, text = extract_answer_from_window(
            s_logits,
            e_logits,
            r_logit,
            sequence_ids,
            offset_mapping,
            context_text,
        )

        # Update best answer for this example_id
        if example_id not in results:
            results[example_id] = (score, text)
        else:
            if score > results[example_id][0]:
                results[example_id] = (score, text)

    # 4. Generate Submission
    print("Generating submission file...")

    # Load sample submission to ensure we have all IDs and correct order
    try:
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)
    except FileNotFoundError:
        print(f"Sample submission not found at {Config.SAMPLE_SUBMISSION}")
        return

    final_preds = []
    for eid in sample_sub["id"]:
        if eid in results:
            pred_text = results[eid][1]
        else:
            pred_text = ""

        # Format string (cleaning and quoting handled by format function or pandas)
        formatted_text = format_prediction_string(pred_text)
        final_preds.append(formatted_text)

    submission_df = pd.DataFrame(
        {"id": sample_sub["id"], "PredictionString": final_preds}
    )

    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
