import os
import torch
import numpy as np
import pandas as pd
import collections
from transformers import AutoTokenizer
from library.config import Config
from library.model import MultiTaskXLMR
from library.data import create_loaders
from library.utils import get_logger

logger = get_logger("Inference")


def predict_ensemble(test_loader, device):
    """
    Runs inference using an ensemble of models loaded from Config.SEEDS.
    Aggregates logits by averaging across all loaded models.

    Args:
        test_loader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on.

    Returns:
        tuple: (start_logits, end_logits, rel_logits) as numpy arrays.
    """
    models = []
    # Load all available seed models
    for seed in Config.SEEDS:
        model_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")
        if not os.path.exists(model_path):
            logger.warning(
                f"Model for seed {seed} not found at {model_path}. Skipping."
            )
            continue

        try:
            model = MultiTaskXLMR(Config.MODEL_CHECKPOINT)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            models.append(model)
            logger.info(f"Loaded model seed {seed}")
        except Exception as e:
            logger.error(f"Failed to load model seed {seed}: {e}")

    if not models:
        logger.error("No models loaded for ensemble!")
        return None, None, None

    all_start_logits = []
    all_end_logits = []
    all_rel_logits = []

    logger.info(f"Running inference with {len(models)} models...")

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Accumulators for the ensemble average
            sum_start = None
            sum_end = None
            sum_rel = None

            for model in models:
                s, e, r = model(input_ids, attention_mask)

                if sum_start is None:
                    sum_start = s
                    sum_end = e
                    sum_rel = r
                else:
                    sum_start += s
                    sum_end += e
                    sum_rel += r

            # Average the logits
            n = len(models)
            avg_start = sum_start / n
            avg_end = sum_end / n
            avg_rel = sum_rel / n

            # Move to CPU and store
            all_start_logits.append(avg_start.cpu().numpy())
            all_end_logits.append(avg_end.cpu().numpy())
            all_rel_logits.append(avg_rel.cpu().numpy())

    # Concatenate all batches into single arrays
    final_start = np.concatenate(all_start_logits, axis=0)
    final_end = np.concatenate(all_end_logits, axis=0)
    final_rel = np.concatenate(all_rel_logits, axis=0)

    return final_start, final_end, final_rel


def post_process_predictions(test_features_df, start_logits, end_logits, rel_logits):
    """
    Reconstructs answers from logits using the Gated Scoring mechanism.
    Selects the best span across all sliding windows for each example_id.

    Args:
        test_features_df (pd.DataFrame): DataFrame containing feature metadata (offsets, example_ids).
        start_logits (np.ndarray): Predicted start logits.
        end_logits (np.ndarray): Predicted end logits.
        rel_logits (np.ndarray): Predicted relevance logits.

    Returns:
        pd.DataFrame: Submission dataframe with 'id' and 'PredictionString'.
    """
    # Load original context mapping from metadata
    if not os.path.exists(Config.TEST_META_PATH):
        logger.error(f"Test metadata not found at {Config.TEST_META_PATH}")
        return pd.DataFrame()

    test_meta = pd.read_csv(Config.TEST_META_PATH)
    id_to_context = dict(zip(test_meta["id"], test_meta["context"]))

    # Dictionary to store the best candidate per example_id
    # Structure: {example_id: (best_score, answer_text)}
    best_candidates = collections.defaultdict(lambda: (-float("inf"), ""))

    # Iterate through each window prediction
    # Note: test_features_df rows align 1-to-1 with the logits arrays
    for idx, row in test_features_df.iterrows():
        example_id = row["example_id"]
        offsets = row["offset_mapping"]

        # Get logits for this specific window
        s_logits = start_logits[idx]  # Shape: (Seq_Len,)
        e_logits = end_logits[idx]  # Shape: (Seq_Len,)
        r_logit = rel_logits[idx][0]  # Shape: Scalar

        seq_len = len(s_logits)

        # --- Gated Scoring ---
        # Score(i, j) = Start_Logit[i] + End_Logit[j] + Relevance_Logit
        # We use broadcasting to create a (Seq_Len, Seq_Len) score matrix
        score_mat = np.expand_dims(s_logits, 1) + np.expand_dims(e_logits, 0)
        score_mat += r_logit

        # --- Masking Invalid Spans ---
        # 1. Start <= End (Upper triangle)
        mask = np.triu(np.ones((seq_len, seq_len)), k=0)

        # 2. Max Answer Length Constraint (Heuristic: e.g., 40 tokens)
        max_ans_len = 40
        mask_len = np.triu(np.ones((seq_len, seq_len)), k=max_ans_len)
        mask = mask - mask_len

        # Apply mask (set invalid positions to negative infinity)
        score_mat = np.where(mask == 1, score_mat, -float("inf"))

        # --- Find Best Span in Window ---
        flat_argmax = np.argmax(score_mat)
        best_start = flat_argmax // seq_len
        best_end = flat_argmax % seq_len
        best_score = score_mat[best_start, best_end]

        # --- Update Global Best for this ID ---
        if best_score > best_candidates[example_id][0]:
            try:
                # Extract text using offset mapping
                # offsets is a list/array of [start_char, end_char]
                start_char = offsets[best_start][0]
                end_char = offsets[best_end][1]

                # Retrieve original context text
                context = id_to_context.get(example_id, "")

                # Validate bounds
                if context and end_char <= len(context):
                    # Extract substring (preserves punctuation and spacing)
                    ans_text = context[start_char:end_char]
                    best_candidates[example_id] = (best_score, ans_text)
            except Exception:
                # Handle cases where offsets might be malformed or special tokens
                pass

    # --- Format Submission ---
    submission_rows = []
    # Ensure we generate a row for every ID in the test set
    all_ids = test_meta["id"].unique()

    for eid in all_ids:
        if eid in best_candidates:
            pred_str = best_candidates[eid][1]
        else:
            pred_str = ""  # Default to empty string if no valid span found

        submission_rows.append({"id": eid, "PredictionString": pred_str})

    return pd.DataFrame(submission_rows)


def run_inference():
    """
    Main orchestration function.
    Loads data, runs ensemble inference, and saves the submission file.
    """
    logger.info("Initializing Inference...")

    # 1. Setup
    device = Config.DEVICE
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)

    # 2. Data Loading
    # create_loaders handles caching. We only need the test components.
    logger.info("Loading test data...")
    _, test_loader, test_features_df = create_loaders(tokenizer, load_cached_data=True)

    # 3. Prediction
    start_logits, end_logits, rel_logits = predict_ensemble(test_loader, device)

    if start_logits is None:
        logger.error("Inference failed: No logits returned.")
        return

    # 4. Post-processing
    logger.info("Post-processing predictions...")
    sub_df = post_process_predictions(
        test_features_df, start_logits, end_logits, rel_logits
    )

    # 5. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Debug Output
    if Config.DEBUG:
        print("Submission Head:")
        print(sub_df.head())
