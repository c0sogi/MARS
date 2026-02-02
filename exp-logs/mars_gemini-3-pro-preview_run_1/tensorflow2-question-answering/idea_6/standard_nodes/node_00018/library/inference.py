import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import defaultdict
from library.config import Config
from library.model import IMCN
from library.dataset import get_dataloaders
from library.embeddings import get_embedding_matrix


def get_best_span(start_probs, end_probs):
    """
    Finds the best valid span (start <= end) maximizing the joint probability.
    Args:
        start_probs (np.array): Array of start probabilities for a candidate.
        end_probs (np.array): Array of end probabilities for a candidate.
    Returns:
        tuple: (start_index, end_index, score) relative to candidate.
    """
    # Create score matrix: score[i, j] = start[i] * end[j]
    scores = np.outer(start_probs, end_probs)

    # Mask invalid spans where end index < start index
    # np.triu returns upper triangle, setting lower to 0
    scores = np.triu(scores)

    # Find indices of the maximum score
    flat_idx = np.argmax(scores)
    best_start, best_end = np.unravel_index(flat_idx, scores.shape)
    best_score = scores[best_start, best_end]

    return best_start, best_end, best_score


def format_prediction(candidates):
    """
    Selects the best candidate for a question and formats the prediction strings.

    Args:
        candidates (list): List of dicts containing probs and metadata for one example_id.

    Returns:
        tuple: (pred_long_string, pred_short_string)
    """
    if not candidates:
        return "", ""

    # 1. Select best Long Answer candidate based on LA probability
    best_cand = max(candidates, key=lambda x: x["la_prob"])

    # 2. Apply Long Answer Threshold
    if best_cand["la_prob"] < Config.TAU_LONG:
        return "", ""

    # Construct Long Answer String (Global Indices)
    # The candidate span is defined by global_start and global_end provided by the dataset
    c_start = best_cand["global_start"]
    c_end = best_cand["global_end"]
    pred_long = f"{c_start}:{c_end}"

    # 3. Determine Short Answer
    # Find best span within the selected candidate
    s_idx, e_idx, s_score = get_best_span(
        best_cand["start_probs"], best_cand["end_probs"]
    )

    # Apply Short Answer Threshold
    if s_score < Config.TAU_SHORT:
        pred_short = ""
    else:
        # Convert relative candidate indices to global document indices
        # s_idx is relative to the start of the candidate
        final_start = c_start + s_idx

        # e_idx is inclusive in our logic (from argmax), but NQ format typically
        # expects start:end where end is exclusive (like python slicing).
        # We add 1 to make it exclusive.
        final_end = c_start + e_idx + 1

        pred_short = f"{final_start}:{final_end}"

    return pred_long, pred_short


def predict(load_cached_data=True, batch_size=Config.BATCH_SIZE):
    """
    Runs the inference pipeline: loads data/model, predicts, and saves submission.
    """
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # 1. Load Data
    # We use the test loader. word2idx is needed for embedding loading.
    print("Loading test data...")
    _, _, test_loader, word2idx = get_dataloaders(
        load_cached_data=load_cached_data, batch_size=batch_size
    )

    # 2. Load Model
    print("Loading model...")
    # Embedding matrix is required to initialize the model structure
    embedding_matrix = get_embedding_matrix(word2idx, load_cached_data=load_cached_data)
    model = IMCN(embedding_matrix)

    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading weights from {Config.MODEL_SAVE_PATH}")
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model weights not found at {Config.MODEL_SAVE_PATH}. Using random initialization."
        )

    model.to(device)
    model.eval()

    # 3. Inference Loop
    # Store results grouped by example_id
    results = defaultdict(list)

    print("Predicting on test set...")
    with torch.no_grad():
        for batch in test_loader:
            q_indices = batch["q_indices"].to(device)
            c_indices = batch["c_indices"].to(device)

            # Forward Pass
            la_logits, start_logits, end_logits = model(q_indices, c_indices)

            # Convert logits to probabilities
            la_probs = torch.sigmoid(la_logits).squeeze(-1).cpu().numpy()
            start_probs = F.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = F.softmax(end_logits, dim=1).cpu().numpy()

            # Metadata for mapping back to global IDs
            example_ids = batch["example_ids"]
            global_starts = batch["global_starts"]
            global_ends = batch["global_ends"]

            # Aggregate
            for i, ex_id in enumerate(example_ids):
                results[ex_id].append(
                    {
                        "la_prob": la_probs[i],
                        "start_probs": start_probs[i],
                        "end_probs": end_probs[i],
                        "global_start": global_starts[i],
                        "global_end": global_ends[i],
                    }
                )

    # 4. Format and Save Submission
    print("Generating submission file...")

    # Load sample submission to ensure correct row order and IDs
    sample_sub_path = os.path.join(Config.INPUT_DIR, Config.SAMPLE_SUBMISSION_FILE)
    if not os.path.exists(sample_sub_path):
        # Fallback if specific sample file name differs slightly in environment
        sample_sub_path = os.path.join(Config.INPUT_DIR, "sampleSubmission.csv")

    sample_sub = pd.read_csv(sample_sub_path)

    # Pre-calculate predictions for all IDs in results
    prediction_map = {}
    for ex_id, candidates in results.items():
        long_str, short_str = format_prediction(candidates)
        prediction_map[ex_id] = (long_str, short_str)

    output_rows = []

    # Iterate through sample submission to fill values
    for _, row in sample_sub.iterrows():
        full_id = row["example_id"]

        # Parse ID: e.g., "-12345_long" -> core_id="-12345", type="long"
        if "_long" in full_id:
            core_id = full_id.replace("_long", "")
            pred_type = "long"
        elif "_short" in full_id:
            core_id = full_id.replace("_short", "")
            pred_type = "short"
        else:
            # Fallback
            core_id = full_id
            pred_type = "unknown"

        pred_str = ""
        if core_id in prediction_map:
            long_p, short_p = prediction_map[core_id]
            if pred_type == "long":
                pred_str = long_p
            elif pred_type == "short":
                pred_str = short_p

        output_rows.append({"example_id": full_id, "PredictionString": pred_str})

    submission_df = pd.DataFrame(output_rows)

    Config.ensure_directories()
    save_path = os.path.join(Config.SUBMISSION_DIR, Config.SUBMISSION_FILE)
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved successfully to {save_path}")
