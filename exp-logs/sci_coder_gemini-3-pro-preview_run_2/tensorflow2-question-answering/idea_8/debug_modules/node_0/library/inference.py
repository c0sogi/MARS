import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import config
from library.data_utils import build_vocab
from library.dataset import NQDataset
from library.model import KernelPoolingNetwork


def run_inference(load_cached_data=True, limit_size=None, batch_size=None):
    """
    Runs inference on the test set and generates the submission file.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
        limit_size (int, optional): Limit dataset size for debugging.
        batch_size (int, optional): Batch size for inference. Defaults to config.BATCH_SIZE.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference using device: {device}")

    # 1. Load Resources
    # We need the vocab to initialize the model embedding layer correctly
    vocab = build_vocab(load_cached_data=load_cached_data)

    # 2. Initialize Model
    model = KernelPoolingNetwork(vocab).to(device)

    # 3. Load Model Weights
    checkpoint_path = config.MODEL_CHECKPOINT_PATH
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model checkpoint not found at {checkpoint_path}. Using random weights."
        )

    model.eval()

    # 4. Prepare Test Data
    # NQDataset handles caching internally based on the load_cached_data flag
    test_dataset = NQDataset(
        split="test",
        vocab=vocab,
        load_cached_data=load_cached_data,
        limit_size=limit_size,
    )

    b_size = batch_size if batch_size is not None else config.BATCH_SIZE
    test_loader = DataLoader(
        test_dataset, batch_size=b_size, shuffle=False, num_workers=0
    )

    print(f"Starting inference on {len(test_dataset)} samples...")

    # Store predictions: example_id -> list of candidate predictions
    all_predictions = {}

    with torch.no_grad():
        for batch in test_loader:
            question = batch["question"].to(device)
            candidate = batch["candidate"].to(device)

            # Forward pass
            outputs = model(question, candidate)

            # Extract scores
            # Ranking score: Sigmoid to get probability [0, 1]
            long_scores = torch.sigmoid(outputs["long_score"]).cpu().numpy().flatten()

            # Span indices: Argmax over sequence length
            start_idxs = torch.argmax(outputs["start_logits"], dim=1).cpu().numpy()
            end_idxs = torch.argmax(outputs["end_logits"], dim=1).cpu().numpy()

            # Yes/No: Argmax over classes
            yesno_idxs = torch.argmax(outputs["yesno_logits"], dim=1).cpu().numpy()

            # Metadata
            example_ids = batch["example_id"]  # List of strings
            cand_idxs = batch["candidate_index"].numpy()

            # Group by example_id
            for i in range(len(example_ids)):
                eid = example_ids[i]
                if eid not in all_predictions:
                    all_predictions[eid] = []

                all_predictions[eid].append(
                    {
                        "cand_idx": cand_idxs[i],
                        "long_score": long_scores[i],
                        "start_rel": start_idxs[i],
                        "end_rel": end_idxs[i],
                        "yesno_idx": yesno_idxs[i],
                    }
                )

    # 5. Generate Submission
    print("Processing predictions and generating submission file...")

    submission_rows = []
    yn_labels = {0: "", 1: "YES", 2: "NO"}  # 0 is NONE -> Blank

    # We iterate through the raw test file to ensure we cover all examples
    # and to get the global token offsets for the candidates.
    # This is necessary because the model operates on padded, truncated candidate segments.

    try:
        with open(config.TEST_DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                ex_id = str(entry["example_id"])

                # Default empty prediction
                long_pred_str = ""
                short_pred_str = ""

                if ex_id in all_predictions:
                    preds = all_predictions[ex_id]

                    # Select the best candidate based on the ranking score
                    best_pred = max(preds, key=lambda x: x["long_score"])

                    # Apply Confidence Threshold
                    if best_pred["long_score"] >= config.LONG_ANSWER_THRESHOLD:

                        # Retrieve candidate info to map back to document offsets
                        candidates = entry["long_answer_candidates"]
                        c_idx = best_pred["cand_idx"]

                        if c_idx < len(candidates):
                            cand_info = candidates[c_idx]
                            global_start = cand_info["start_token"]
                            global_end = cand_info["end_token"]

                            # Construct Long Answer String
                            long_pred_str = f"{global_start}:{global_end}"

                            # Determine Short Answer
                            # Check Yes/No first
                            yn_idx = best_pred["yesno_idx"]
                            yn_str = yn_labels.get(yn_idx, "")

                            if yn_str:
                                short_pred_str = yn_str
                            else:
                                # Construct Short Answer Span
                                s_rel = best_pred["start_rel"]
                                e_rel = best_pred["end_rel"]

                                # Map relative indices to global indices
                                s_global = global_start + s_rel
                                e_global = global_start + e_rel

                                # Validate span constraints
                                # 1. Must be within the candidate boundaries
                                # 2. Start must be <= End
                                if (
                                    s_global < global_end
                                    and e_global < global_end
                                    and s_global <= e_global
                                ):

                                    # Output start:end+1 for exclusive end index
                                    short_pred_str = f"{s_global}:{e_global + 1}"

                # Append rows for this example
                submission_rows.append([f"{ex_id}_long", long_pred_str])
                submission_rows.append([f"{ex_id}_short", short_pred_str])

    except FileNotFoundError:
        print(f"Error: Test data file not found at {config.TEST_DATA_PATH}")
        return

    # Create DataFrame and save
    sub_df = pd.DataFrame(submission_rows, columns=["example_id", "PredictionString"])

    # Ensure directory exists
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    sub_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
    print(f"Total predictions generated: {len(sub_df)}")
