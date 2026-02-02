import os
import torch
import numpy as np
import pandas as pd
import json
from library.config import Config
from library.utils import seed_everything
from library.data import get_vocab, get_test_dataloader
from library.model import CQCRNN


class Predictor:
    """
    Manages the inference process for the Question Answering model.
    Loads the trained model and generates predictions for the test set.
    """

    def __init__(
        self, model_path, device, long_answer_threshold=Config.LONG_ANSWER_THRESHOLD
    ):
        self.model_path = model_path
        self.device = device
        self.threshold = long_answer_threshold
        self.vocab = get_vocab(load_cached_data=True)
        self.model = self._load_model()

    def _load_model(self):
        print(f"Loading model from {self.model_path}...")
        model = CQCRNN(
            vocab_size=len(self.vocab),
            embed_dim=Config.EMBED_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            num_layers=Config.NUM_LAYERS,
            dropout=Config.DROPOUT,
        ).to(self.device)

        if os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        else:
            print(
                f"Warning: Model checkpoint not found at {self.model_path}. Using initialized weights."
            )

        model.eval()
        return model

    def predict(self, test_loader, offset_map):
        """
        Runs inference on the test set and generates prediction strings.

        Args:
            test_loader: DataLoader providing batched test data.
            offset_map: Dictionary mapping example_id to list of candidate offsets.

        Returns:
            List of strings formatted for submission CSV.
        """
        results = []
        print("Running inference...")

        with torch.no_grad():
            for batch in test_loader:
                # Move inputs to device
                q_input = batch["q_input"].to(self.device)
                c_input = batch["c_input"].to(self.device)

                # Forward pass
                outputs = self.model(q_input, c_input)

                # Get probabilities/logits
                long_probs = (
                    torch.sigmoid(outputs["long_logits"]).cpu().numpy().flatten()
                )
                start_logits = outputs["start_logits"].cpu().numpy()
                end_logits = outputs["end_logits"].cpu().numpy()
                yn_logits = outputs["yn_logits"].cpu().numpy()

                # Batch metadata
                example_ids = batch["example_ids"]
                counts = batch["candidate_counts"]

                current_idx = 0
                for i, ex_id in enumerate(example_ids):
                    count = counts[i]

                    # Default empty predictions
                    pred_long = ""
                    pred_short = ""

                    if count > 0:
                        # Slice outputs for this specific example
                        sl_scores = long_probs[current_idx : current_idx + count]
                        sl_start = start_logits[current_idx : current_idx + count]
                        sl_end = end_logits[current_idx : current_idx + count]
                        sl_yn = yn_logits[current_idx : current_idx + count]

                        # 1. Select Best Candidate for Long Answer
                        best_cand_idx = np.argmax(sl_scores)
                        best_score = sl_scores[best_cand_idx]

                        # Apply Threshold
                        if best_score >= self.threshold:
                            # Retrieve offsets from map
                            # offset_map structure: ex_id -> list of dicts {'s': start, 'e': end}
                            if ex_id in offset_map and best_cand_idx < len(
                                offset_map[ex_id]
                            ):
                                cand_offsets = offset_map[ex_id][best_cand_idx]
                                c_start_doc = cand_offsets["s"]
                                c_end_doc = cand_offsets["e"]

                                # Set Long Answer Prediction
                                pred_long = f"{c_start_doc}:{c_end_doc}"

                                # 2. Determine Short Answer / Yes-No
                                # Get logits for the best candidate
                                s_idx = np.argmax(sl_start[best_cand_idx])
                                e_idx = np.argmax(sl_end[best_cand_idx])
                                yn_idx = np.argmax(sl_yn[best_cand_idx])

                                # Class 0: NONE, 1: YES, 2: NO
                                if yn_idx == 1:
                                    pred_short = "YES"
                                elif yn_idx == 2:
                                    pred_short = "NO"
                                else:
                                    # Span prediction
                                    # Index 0 is NULL token. Valid span: start > 0, start <= end
                                    if s_idx > 0 and e_idx > 0 and s_idx <= e_idx:
                                        # Map relative index (1-based in model output due to NULL token)
                                        # to absolute document index.
                                        # Model input: [NULL, token_0, token_1, ...]
                                        # s_idx=1 -> token_0 -> doc_index = c_start_doc
                                        abs_s = c_start_doc + (s_idx - 1)
                                        abs_e = c_start_doc + (e_idx - 1)

                                        # Sanity check: span must be within candidate bounds
                                        if abs_e < c_end_doc:
                                            pred_short = f"{abs_s}:{abs_e}"

                    # Append results in required format
                    results.append(f"{ex_id}_long,{pred_long}")
                    results.append(f"{ex_id}_short,{pred_short}")

                    current_idx += count

        return results


def get_cached_candidate_offsets(raw_data_path, load_cached_data=True):
    """
    Parses raw JSONL to get candidate offsets. Caches result as Parquet.
    Returns a dictionary mapping example_id to a list of candidate offset dictionaries.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "cache", "test_offsets.parquet")

    # 1. Load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading candidate offsets from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
        # Convert back to dict structure: ex_id -> list of dicts
        offset_map = {}
        for _, row in df.iterrows():
            starts = row["start_tokens"]
            ends = row["end_tokens"]
            offset_map[str(row["example_id"])] = [
                {"s": s, "e": e} for s, e in zip(starts, ends)
            ]
        return offset_map

    # 2. Compute from scratch
    print(f"Computing candidate offsets from {raw_data_path}...")
    data = []

    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(raw_data_path):
        with open(raw_data_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                ex_id = str(entry["example_id"])
                candidates = entry.get("long_answer_candidates", [])

                starts = [c["start_token"] for c in candidates]
                ends = [c["end_token"] for c in candidates]

                data.append(
                    {"example_id": ex_id, "start_tokens": starts, "end_tokens": ends}
                )

    df = pd.DataFrame(data)

    # 3. Save to cache
    df.to_parquet(cache_path)
    print(f"Saved candidate offsets to {cache_path}")

    # Convert to map
    offset_map = {}
    for _, row in df.iterrows():
        starts = row["start_tokens"]
        ends = row["end_tokens"]
        offset_map[str(row["example_id"])] = [
            {"s": s, "e": e} for s, e in zip(starts, ends)
        ]

    return offset_map


def run_inference_pipeline(
    model_path=Config.MODEL_SAVE_PATH,
    output_path=Config.SUBMISSION_PATH,
    threshold=Config.LONG_ANSWER_THRESHOLD,
    load_cached_data=True,
):
    """
    Orchestrates the full inference pipeline: loads data, runs model, saves submission.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Prepare Data
    # vocab is cached by library.data
    vocab = get_vocab(load_cached_data=load_cached_data)
    test_loader = get_test_dataloader(vocab, load_cached_data=load_cached_data)

    # Load Offsets (needed for absolute span reconstruction)
    offset_map = get_cached_candidate_offsets(
        Config.TEST_DATA_PATH, load_cached_data=load_cached_data
    )

    # Initialize Predictor
    predictor = Predictor(model_path, device, threshold)

    # Generate Predictions
    results = predictor.predict(test_loader, offset_map)

    # Save Submission
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("example_id,PredictionString\n")
        f.write("\n".join(results))

    print(
        f"Submission file generated at {output_path} with {len(results)//2} examples."
    )
