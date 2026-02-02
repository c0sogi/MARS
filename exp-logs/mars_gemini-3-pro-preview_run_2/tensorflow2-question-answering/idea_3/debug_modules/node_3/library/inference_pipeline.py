import torch
import json
import os
import pandas as pd
from typing import Dict, List
from library.config import Config
from library.data_factory import DataFactory
from library.answer_extractor import SlidingWindowExtractor, detect_yes_no
from library.text_utils import TextUtils


def predict_and_format(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    vocab: Dict[str, int],
    device: torch.device,
    jsonl_path: str,
    output_path: str,
):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: The trained SiameseGatedConvRanker model.
        test_loader: DataLoader for the test set.
        vocab: Vocabulary mapping (token -> index).
        device: Torch device (CPU/GPU).
        jsonl_path: Path to the test JSONL file.
        output_path: Path to save the submission CSV.
    """
    model.eval()

    # Build File Index for random access to JSONL to retrieve token offsets and raw text
    # We use load_cached_data=True to utilize the index built during dataset creation
    file_index = DataFactory.build_file_index(jsonl_path, load_cached_data=True)

    # Initialize the short answer extractor
    extractor = SlidingWindowExtractor(threshold=Config.SHORT_OVERLAP_THRESHOLD)

    results = []

    print("Generating predictions...")

    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch
            # In inference mode, collate_fn returns a dict with these keys
            example_ids = batch["example_ids"]
            q_indices = batch["q_indices"].to(device)
            c_indices = batch["c_indices"].to(device)
            candidate_indices = batch["candidate_indices"]  # [batch, max_cands]
            mask = batch["mask"].to(device)  # [batch, max_cands]

            # Forward pass
            # logits shape: [batch, max_cands]
            logits = model(q_indices, c_indices)
            probs = torch.sigmoid(logits)

            # Mask out padding candidates
            # We set probs of invalid candidates to -1 to ensure they aren't selected
            probs = probs * mask.float() + (1.0 - mask.float()) * -1.0

            # Process each example in the batch
            for i, ex_id in enumerate(example_ids):
                # Find best candidate index within the batch dimension
                best_score, best_arg = torch.max(probs[i], dim=0)

                # Retrieve the original index in the JSON candidate list
                # candidate_indices contains the index into the 'long_answer_candidates' list
                best_cand_idx_in_json = candidate_indices[i, best_arg].item()

                # Initialize prediction strings
                long_pred_str = ""
                short_pred_str = ""

                # Check Confidence Threshold for Long Answer
                # If valid candidate exists (mask check implicit via max logic) and score is high enough
                if (
                    best_cand_idx_in_json != -1
                    and best_score.item() >= Config.LONG_CONFIDENCE_THRESHOLD
                ):

                    # Retrieve raw data for offsets and text extraction
                    if str(ex_id) in file_index:
                        offset = file_index[str(ex_id)]
                        with open(jsonl_path, "rb") as f:
                            f.seek(offset)
                            line = f.readline()
                            entry = json.loads(line.decode("utf-8"))

                        # Get Long Answer Offsets
                        c_info = entry["long_answer_candidates"][best_cand_idx_in_json]
                        long_start = c_info["start_token"]
                        long_end = c_info["end_token"]

                        long_pred_str = f"{long_start}:{long_end}"

                        # --- Short Answer Logic ---
                        doc_text = entry["document_text"]
                        # NQ dataset uses whitespace splitting for token indexing
                        doc_tokens = doc_text.split()

                        # Extract the text of the long answer
                        # Clip indices to be safe
                        ls = max(0, long_start)
                        le = min(len(doc_tokens), long_end)
                        long_answer_text = " ".join(doc_tokens[ls:le])

                        question_text = entry["question_text"]

                        # Use Extractor
                        # Returns relative offsets within the long_answer_text tokens
                        s_rel_start, s_rel_end, s_text = extractor.extract(
                            question_text, long_answer_text
                        )

                        if s_rel_start != -1:
                            # Check for Yes/No
                            yn_label = detect_yes_no(s_text)

                            if yn_label != "NONE":
                                short_pred_str = yn_label
                            else:
                                # Calculate absolute offsets
                                # Note: We assume additive offsets based on the logic in neural_ranker.py
                                abs_start = long_start + s_rel_start
                                abs_end = long_start + s_rel_end
                                short_pred_str = f"{abs_start}:{abs_end}"
                    else:
                        print(f"Warning: Example ID {ex_id} not found in file index.")

                # Append results
                results.append(f"{ex_id}_long,{long_pred_str}")
                results.append(f"{ex_id}_short,{short_pred_str}")

    # Save Submission to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("example_id,PredictionString\n")
        f.write("\n".join(results))

    print(f"Submission saved to {output_path}")
