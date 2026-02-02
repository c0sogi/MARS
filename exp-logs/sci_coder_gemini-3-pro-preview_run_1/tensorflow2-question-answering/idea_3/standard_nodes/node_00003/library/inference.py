import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.utils import load_checkpoint, load_metadata
from library.data_loader import get_dataloader, build_tokenizer
from library.modeling import DanTqpModel


class SubmissionGenerator:
    """
    Handles inference and submission generation for the NQ task.
    """

    def __init__(
        self, model, device, tokenizer, long_threshold=0.5, short_threshold=0.1
    ):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

    def predict(self, dataloader):
        """
        Runs inference on the dataloader and aggregates results by example_id.
        Returns a dictionary mapping example_id to the best prediction details.
        """
        self.model.eval()
        results = {}  # example_id -> {score, candidate_index, short_span}

        print("Running inference...")
        with torch.no_grad():
            for batch in dataloader:
                q_input_ids = batch["q_input_ids"].to(self.device)
                c_input_ids = batch["c_input_ids"].to(self.device)
                example_ids = batch["example_ids"]
                candidate_indices = batch["candidate_indices"]

                # Forward pass
                ranker_logits, extractor_logits = self.model(q_input_ids, c_input_ids)

                # Long Answer Probabilities
                long_probs = torch.sigmoid(ranker_logits).squeeze(-1).cpu().numpy()

                # Short Answer Probabilities
                # shape: (batch, seq_len, 3) -> softmax over last dim
                sa_probs = F.softmax(extractor_logits, dim=-1).cpu().numpy()

                for i, ex_id in enumerate(example_ids):
                    l_score = long_probs[i]
                    cand_idx = candidate_indices[i]

                    # Short Answer Extraction Logic
                    # Find best span (start, end) maximizing p_start * p_end
                    # Class 1 = Start, Class 2 = End
                    starts = sa_probs[i, :, 1]
                    ends = sa_probs[i, :, 2]

                    best_s_score = 0
                    best_span = None

                    # Simple greedy search for best span
                    # Limit span length to reasonable size (e.g., 30 tokens) for efficiency
                    seq_len = len(starts)

                    # Optimization: Only look at tokens with high start prob
                    start_candidates = np.where(starts > 0.01)[0]

                    for s_idx in start_candidates:
                        # Look for end in window [s_idx, s_idx + 30]
                        e_window = ends[s_idx : min(s_idx + 30, seq_len)]
                        if len(e_window) == 0:
                            continue

                        best_local_e = np.argmax(e_window)
                        e_idx = s_idx + best_local_e
                        score = starts[s_idx] * ends[e_idx]

                        if score > best_s_score:
                            best_s_score = score
                            best_span = (s_idx, e_idx)

                    # Aggregate results
                    # We keep the candidate with the highest Long Answer score for this example
                    if ex_id not in results or l_score > results[ex_id]["long_score"]:
                        results[ex_id] = {
                            "long_score": l_score,
                            "candidate_index": cand_idx,
                            "short_score": best_s_score,
                            "short_span": best_span,
                        }

        return results

    def generate_submission_file(self, predictions, test_file_path, output_path):
        """
        Reads the test file to map candidate indices to global token offsets and writes the submission CSV.
        """
        print(f"Generating submission file at {output_path}...")

        # Prepare output data
        submission_rows = []

        if not os.path.exists(test_file_path):
            raise FileNotFoundError(f"Test file not found: {test_file_path}")

        # We will process the test file line by line.
        with open(test_file_path, "rb") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ex_id = entry["example_id"]

                # Default predictions
                long_pred_str = ""
                short_pred_str = ""

                if ex_id in predictions:
                    pred = predictions[ex_id]

                    # 1. Long Answer Decision
                    if pred["long_score"] > self.long_threshold:
                        cand_idx = pred["candidate_index"]
                        candidates = entry.get("long_answer_candidates", [])

                        if cand_idx < len(candidates):
                            cand = candidates[cand_idx]
                            # Format: start_token:end_token
                            long_pred_str = f"{cand['start_token']}:{cand['end_token']}"

                            # 2. Short Answer Decision (Only if Long Answer is valid)
                            if (
                                pred["short_score"] > self.short_threshold
                                and pred["short_span"] is not None
                            ):
                                local_s, local_e = pred["short_span"]
                                # Map local to global
                                # c_ids corresponds to tokens starting at cand['start_token'].
                                global_s = cand["start_token"] + local_s
                                # NQ format is usually inclusive start, exclusive end for Python slicing,
                                # but the submission format examples (e.g., 6:18) typically imply
                                # token indices. The provided example shows 'start:end'.
                                # Assuming standard NQ convention where end is exclusive.
                                # However, our span search found e_idx as the index of the End token (inclusive).
                                # So exclusive end index is e_idx + 1.
                                global_e = cand["start_token"] + local_e + 1

                                # Verify bounds
                                if global_e <= cand["end_token"]:
                                    short_pred_str = f"{global_s}:{global_e}"

                # Append rows
                submission_rows.append(
                    {"example_id": f"{ex_id}_long", "PredictionString": long_pred_str}
                )
                submission_rows.append(
                    {"example_id": f"{ex_id}_short", "PredictionString": short_pred_str}
                )

        # Create DataFrame and save
        df = pd.DataFrame(submission_rows)

        # Ensure output directory exists
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        df.to_csv(output_path, index=False)
        print(f"Submission saved with {len(df)} rows.")


def run_inference(
    device="cpu",
    batch_size=32,
    num_workers=2,
    checkpoint_path="./working/idea_3/best_model.pth",
    output_csv="./submission/submission.csv",
    load_cached_data=True,
):
    """
    Main entry point for inference.
    """
    # 1. Setup
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Metadata to build tokenizer
    print("Loading metadata for tokenizer...")
    # We use train metadata to build vocab as test vocab might be unseen
    train_meta = load_metadata("train")
    tokenizer = build_tokenizer(train_meta, load_cached_data=load_cached_data)
    vocab_size = len(tokenizer)
    print(f"Tokenizer vocab size: {vocab_size}")

    # 3. Load Data
    print("Preparing test dataloader...")
    # Note: neg_ratio doesn't apply to test split
    test_loader = get_dataloader(
        split="test",
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        load_cached_data=load_cached_data,
    )

    # 4. Initialize and Load Model
    model = DanTqpModel(vocab_size=vocab_size)
    try:
        load_checkpoint(checkpoint_path, model, device=device)
    except FileNotFoundError:
        print(
            "Warning: Checkpoint not found. Using random weights (for debugging/testing flow)."
        )

    model.to(device)

    # 5. Run Prediction
    generator = SubmissionGenerator(model, device, tokenizer)
    predictions = generator.predict(test_loader)

    # 6. Generate Submission
    test_file_raw = "./input/simplified-nq-test.jsonl"
    generator.generate_submission_file(predictions, test_file_raw, output_csv)
