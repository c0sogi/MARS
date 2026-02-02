import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import defaultdict
import os

from library.config import Config, set_seed
from library.utils import setup_logger, load_checkpoint, format_submission_file
from library.data_processing import get_dataloaders
from library.model import DAAN

logger = setup_logger("inference")


def get_best_span(start_probs, end_probs):
    """
    Finds the optimal short answer span (start, end) that maximizes the joint probability.
    Constraints: start <= end.

    Args:
        start_probs (np.array): Probability distribution over start tokens. Shape: [SeqLen]
        end_probs (np.array): Probability distribution over end tokens. Shape: [SeqLen]

    Returns:
        best_start (int): Index of the start token.
        best_end (int): Index of the end token.
        max_score (float): The joint probability score (start_prob * end_prob).
    """
    # Create a matrix of joint probabilities: [SeqLen, SeqLen]
    # score[i, j] = start_probs[i] * end_probs[j]
    score_mat = np.outer(start_probs, end_probs)

    # Mask out invalid spans where start > end (lower triangle)
    # triu returns upper triangle, others zeroed.
    # We use np.triu to keep valid spans (i <= j)
    score_mat = np.triu(score_mat)

    # Find indices of the maximum score
    flat_idx = np.argmax(score_mat)
    best_start, best_end = np.unravel_index(flat_idx, score_mat.shape)
    max_score = score_mat[best_start, best_end]

    return best_start, best_end, max_score


def predict_on_test(model, test_loader, device):
    """
    Runs the model on the test set and collects raw predictions.

    Args:
        model (nn.Module): Loaded model.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Computation device.

    Returns:
        pd.DataFrame: DataFrame containing aggregated prediction data for post-processing.
    """
    model.eval()
    results = []

    logger.info(f"Starting inference on {len(test_loader)} batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            q_input = batch["q_input"].to(device)
            c_input = batch["c_input"].to(device)

            # Forward pass
            la_logits, start_logits, end_logits = model(q_input, c_input)

            # Process Long Answer Probabilities
            la_probs = torch.sigmoid(la_logits).squeeze(-1).cpu().numpy()

            # Process Short Answer Probabilities
            start_probs = F.softmax(start_logits, dim=-1).cpu().numpy()
            end_probs = F.softmax(end_logits, dim=-1).cpu().numpy()

            # Extract metadata
            example_ids = batch["example_id"]
            global_starts = batch["global_start"].numpy()
            global_ends = batch["global_end"].numpy()

            # Collect data
            for i in range(len(example_ids)):
                results.append(
                    {
                        "example_id": example_ids[i],
                        "la_prob": la_probs[i],
                        "start_probs": start_probs[i],
                        "end_probs": end_probs[i],
                        "global_start": global_starts[i],
                        "global_end": global_ends[i],
                    }
                )

            if batch_idx % 2000 == 0 and batch_idx > 0:
                logger.info(f"Processed batch {batch_idx}/{len(test_loader)}")

    return pd.DataFrame(results)


def select_answers(predictions_df, tau_long, tau_short):
    """
    Applies thresholds and selection logic to generate final prediction strings.

    Args:
        predictions_df (pd.DataFrame): Raw predictions.
        tau_long (float): Threshold for Long Answer.
        tau_short (float): Threshold for Short Answer confidence.

    Returns:
        dict: Mapping of submission IDs to prediction strings.
    """
    final_predictions = {}

    # Group by example_id to handle multiple candidates per question
    grouped = predictions_df.groupby("example_id")

    logger.info(f"Processing predictions for {len(grouped)} unique examples...")

    for example_id, group in grouped:
        # 1. Long Answer Selection
        # Find the candidate with the highest Long Answer probability
        best_cand_idx = group["la_prob"].idxmax()
        best_cand = group.loc[best_cand_idx]

        la_prob = best_cand["la_prob"]

        long_pred_str = ""
        short_pred_str = ""

        if la_prob >= tau_long:
            # Construct Long Answer String (global token indices)
            # Format: "start:end" (end is exclusive in NQ evaluation context usually,
            # but here global_end is the index of the token after the candidate in flatten logic)
            # Based on flatten_data: c_end = cand["end_token"] (exclusive).
            long_pred_str = f"{best_cand['global_start']}:{best_cand['global_end']}"

            # 2. Short Answer Selection (Only if Long Answer is valid)
            # Extract best span within this candidate
            s_probs = best_cand["start_probs"]
            e_probs = best_cand["end_probs"]

            start_local, end_local, span_score = get_best_span(s_probs, e_probs)

            # Check Short Answer Threshold
            # Note: span_score is a probability (product of softmax outputs), range [0, 1]
            if span_score >= tau_short:
                # Convert local candidate indices to global document indices
                # global_start is the document index of the first token of the candidate
                sa_global_start = best_cand["global_start"] + start_local

                # end_local is inclusive index within candidate (0-based)
                # For submission string "start:end", end is usually exclusive.
                sa_global_end = best_cand["global_start"] + end_local + 1

                short_pred_str = f"{sa_global_start}:{sa_global_end}"

        # Add to results
        final_predictions[f"{example_id}_long"] = long_pred_str
        final_predictions[f"{example_id}_short"] = short_pred_str

    return final_predictions


def run_inference(load_cached_data=True):
    """
    Main execution function for the inference module.

    Args:
        load_cached_data (bool): Whether to use cached data preprocessing.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running inference on device: {device}")

    # 1. Load Data
    # We only need the test_loader and embedding_matrix (for model init)
    # train_loader and val_loader are ignored here
    logger.info("Loading DataLoaders...")
    _, _, test_loader, embedding_matrix = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    logger.info("Initializing DAAN Model structure...")
    model = DAAN(embedding_matrix)
    model.to(device)

    # 3. Load Weights
    logger.info(f"Loading model weights from {Config.MODEL_PATH}...")
    epoch, loss = load_checkpoint(Config.MODEL_PATH, model, device=device)
    if epoch == 0:
        logger.warning(
            "No checkpoint found or failed to load. Using random weights (expect poor results)."
        )
    else:
        logger.info(f"Loaded checkpoint from Epoch {epoch} with Val Loss {loss:.4f}")

    # 4. Run Inference
    raw_predictions_df = predict_on_test(model, test_loader, device)

    # 5. Select Answers and Format
    logger.info(
        f"Applying thresholds: TAU_LONG={Config.TAU_LONG}, TAU_SHORT={Config.TAU_SHORT}"
    )
    submission_dict = select_answers(
        raw_predictions_df, tau_long=Config.TAU_LONG, tau_short=Config.TAU_SHORT
    )

    # 6. Save Submission
    logger.info(f"Saving submission to {Config.SUBMISSION_FILE}...")
    format_submission_file(submission_dict, Config.SUBMISSION_FILE)

    logger.info("Inference completed successfully.")
