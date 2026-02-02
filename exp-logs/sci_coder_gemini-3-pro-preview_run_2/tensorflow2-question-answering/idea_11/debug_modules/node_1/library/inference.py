import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.vocab import Vocabulary
from library.dataset import NQDataset, collate_fn
from library.model import AGBoEModel


class InferenceEngine:
    def __init__(self, load_cached_data=True, sample_size=None):
        """
        Initializes the InferenceEngine.

        Args:
            load_cached_data (bool): Whether to load pre-processed features from cache.
            sample_size (int, optional): Limit the number of test samples for debugging.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_cached_data = load_cached_data
        self.sample_size = sample_size

        print(f"InferenceEngine initialized on device: {self.device}")

        # 1. Load Vocabulary and Embeddings to initialize model structure
        # We rely on the cache generated during training usually, or rebuild if missing (though unlikely in inference)
        self.vocab, self.embedding_matrix = Vocabulary.load_or_build(
            load_cached_data=True,  # Always try to load cache for inference
            sample_size=None,
        )

        # 2. Initialize Model
        self.model = AGBoEModel(self.embedding_matrix).to(self.device)

        # 3. Load Weights
        if os.path.exists(Config.MODEL_SAVE_PATH):
            print(f"Loading model weights from {Config.MODEL_SAVE_PATH}...")
            state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(
                f"Warning: Checkpoint not found at {Config.MODEL_SAVE_PATH}. Using random weights."
            )

        self.model.eval()

    def run_inference(self):
        """
        Runs the full inference pipeline: loads data, predicts, post-processes, and saves submission.
        """
        print("--- Starting Inference ---")

        # 1. Prepare Data Loader
        dataset = NQDataset(
            split="test",
            vocab=self.vocab,
            load_cached_data=self.load_cached_data,
            sample_size=self.sample_size,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2 if self.device.type == "cuda" else 0,
        )

        # 2. Prediction Loop
        # Store results grouped by example_id
        # Structure: example_id -> list of candidate result dicts
        grouped_results = {}

        with torch.no_grad():
            for batch in dataloader:
                # Move inputs to device
                q_indices = batch["q_indices"].to(self.device)
                c_indices = batch["c_indices"].to(self.device)

                # Metadata
                example_ids = batch["example_ids"]
                cand_starts = batch["cand_global_starts"]
                cand_ends = batch["cand_global_ends"]

                # Model Forward Pass
                ranking_logits, yesno_logits, attn_weights = self.model(
                    q_indices, c_indices
                )

                # Process Outputs
                ranking_probs = torch.sigmoid(ranking_logits).cpu().numpy()
                yesno_probs = torch.softmax(yesno_logits, dim=1).cpu().numpy()
                attn_weights_np = attn_weights.cpu().numpy()

                # Aggregate
                for i, eid in enumerate(example_ids):
                    if eid not in grouped_results:
                        grouped_results[eid] = []

                    grouped_results[eid].append(
                        {
                            "score": ranking_probs[i],
                            "yesno_probs": yesno_probs[i],
                            "attn_weights": attn_weights_np[i],
                            "global_start": cand_starts[i],
                            "global_end": cand_ends[i],
                        }
                    )

        # 3. Post-Processing and Formatting
        print("Processing predictions...")
        final_submission_data = []

        # We iterate over the grouped results.
        # Note: In a strict competition setting, one might iterate over the sample_submission
        # to ensure order, but here we process what we have.

        for eid, candidates in grouped_results.items():
            # Find best candidate by ranking score
            best_cand = max(candidates, key=lambda x: x["score"])

            long_pred_str = ""
            short_pred_str = ""

            # Apply Confidence Threshold
            if best_cand["score"] >= Config.LONG_ANSWER_THRESHOLD:
                # --- Long Answer ---
                # The long answer is the span of the candidate itself
                long_pred_str = f"{best_cand['global_start']}:{best_cand['global_end']}"

                # --- Short Answer ---
                # Check Yes/No first (0=NONE, 1=YES, 2=NO)
                yn_idx = np.argmax(best_cand["yesno_probs"])

                if yn_idx == 1:
                    short_pred_str = "YES"
                elif yn_idx == 2:
                    short_pred_str = "NO"
                else:
                    # Extract span using sliding window on attention weights
                    weights = best_cand["attn_weights"]

                    # Determine valid length of the candidate (excluding padding)
                    # The model input was padded to MAX_SEQ_LEN_C.
                    # The actual token count is global_end - global_start.
                    valid_len = best_cand["global_end"] - best_cand["global_start"]

                    # Safety clamp in case valid_len exceeds max seq len (truncation case)
                    valid_len = min(valid_len, Config.MAX_SEQ_LEN_C)

                    # Slice weights to valid tokens only
                    valid_weights = weights[:valid_len]

                    window_size = Config.SHORT_SPAN_WINDOW

                    if len(valid_weights) >= window_size:
                        # Sliding window sum
                        kernel = np.ones(window_size)
                        # mode='valid' returns output of length N - K + 1
                        sums = np.convolve(valid_weights, kernel, mode="valid")

                        # Find index of max window
                        best_rel_start = np.argmax(sums)
                        best_rel_end = best_rel_start + window_size
                    else:
                        # Candidate shorter than window: take the whole valid sequence
                        best_rel_start = 0
                        best_rel_end = len(valid_weights)

                    # Convert relative indices to global indices
                    s_global_start = best_cand["global_start"] + best_rel_start
                    s_global_end = best_cand["global_start"] + best_rel_end

                    short_pred_str = f"{s_global_start}:{s_global_end}"

            # Append rows for this example
            final_submission_data.append(
                {"example_id": f"{eid}_long", "PredictionString": long_pred_str}
            )
            final_submission_data.append(
                {"example_id": f"{eid}_short", "PredictionString": short_pred_str}
            )

        # 4. Save Submission
        submission_df = pd.DataFrame(final_submission_data)

        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Total predictions generated: {len(submission_df)}")
