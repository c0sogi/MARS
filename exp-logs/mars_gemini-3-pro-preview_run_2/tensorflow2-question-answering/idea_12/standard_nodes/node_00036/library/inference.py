import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import load_glove_embeddings
from library.data_processing import build_vocab
from library.dataset import get_test_dataloader, preprocess_and_cache, YESNO_MAP
from library.model import FiLMNetwork


class Predictor:
    """
    Handles the inference pipeline for the Natural Questions task.
    Loads the trained model, processes the test set, and generates the submission file.
    """

    def __init__(self, device=None, model_path=None):
        """
        Initialize the Predictor.

        Args:
            device (torch.device, optional): Device to run inference on. Defaults to auto-detect.
            model_path (str, optional): Path to the trained model weights. Defaults to Config.MODEL_SAVE_PATH.
        """
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_path = model_path if model_path else Config.MODEL_SAVE_PATH

        # Inverse mapping for Yes/No labels
        self.idx_to_yesno = {v: k for k, v in YESNO_MAP.items()}

        print(f"Initializing Predictor on device: {self.device}")

        # 1. Load Vocabulary and Embeddings to initialize model structure
        self.vocab = build_vocab(load_cached_data=True)
        embedding_matrix = load_glove_embeddings(
            self.vocab.stoi,
            Config.EMBEDDING_DIM,
            glove_path=None,  # Using random/cached for this baseline logic
        )

        # 2. Initialize Model
        self.model = FiLMNetwork(embedding_matrix)

        # 3. Load Weights
        if os.path.exists(self.model_path):
            print(f"Loading model weights from {self.model_path}")
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(
                f"Warning: Model checkpoint not found at {self.model_path}. Using random weights."
            )

        self.model.to(self.device)
        self.model.eval()

    def _get_best_span(self, start_logits, end_logits):
        """
        Finds the best valid span (start <= end) maximizing the sum of logits.

        Args:
            start_logits (torch.Tensor): Shape (Seq_Len,)
            end_logits (torch.Tensor): Shape (Seq_Len,)

        Returns:
            tuple: (start_idx, end_idx) relative to candidate.
        """
        # Simple constrained search
        # We only look at the top few start indices to keep it fast
        top_k = 5
        start_probs = F.softmax(start_logits, dim=0)
        end_probs = F.softmax(end_logits, dim=0)

        # Get top K start indices
        top_start_indices = torch.topk(
            start_probs, k=min(top_k, len(start_probs))
        ).indices.tolist()

        best_score = -float("inf")
        best_span = (0, 0)

        for s_idx in top_start_indices:
            # For this start index, find the best end index >= s_idx
            # We limit the max span length to avoid extremely long short answers (heuristic)
            max_span_len = 30
            e_search_end = min(len(end_probs), s_idx + max_span_len)

            if s_idx >= e_search_end:
                continue

            # Slice relevant end logits
            valid_end_logits = end_logits[s_idx:e_search_end]
            best_rel_e = torch.argmax(valid_end_logits).item()
            e_idx = s_idx + best_rel_e

            score = start_logits[s_idx].item() + end_logits[e_idx].item()

            if score > best_score:
                best_score = score
                best_span = (s_idx, e_idx)

        return best_span

    def generate_predictions(self, threshold=Config.LONG_ANSWER_THRESHOLD):
        """
        Runs inference on the test set and generates prediction strings.

        Args:
            threshold (float): Confidence threshold for predicting an answer.

        Returns:
            list: List of dictionaries containing submission rows.
        """
        print("Loading test data...")
        # We need the DataLoader for model input
        test_loader = get_test_dataloader(self.vocab, load_cached_data=True)

        # We also need the DataFrame to access raw candidate offsets,
        # which aren't passed through the standard Dataset __getitem__ for tensor efficiency.
        test_df = preprocess_and_cache(
            Config.TEST_META,
            Config.TEST_FILE,
            self.vocab,
            Config.TEST_CACHE,
            is_train=False,
        )
        # Create a lookup map for fast access to candidates by example_id
        # example_id is string in df
        test_data_map = test_df.set_index("example_id")["candidates"].to_dict()

        results = []
        print(f"Running inference on {len(test_loader)} examples...")

        with torch.no_grad():
            for batch in test_loader:
                # Unpack batch
                # batch size is 1, but contains N candidates
                example_id = batch["example_id"][0]  # str
                q_input = batch["q_input"].to(self.device)  # (1, Q_Len)
                candidates = batch["candidates"].to(self.device)  # (1, N, Ctx_Len)

                # Remove batch dimension
                candidates = candidates.squeeze(0)  # (N, Ctx_Len)
                num_cands = candidates.size(0)

                if num_cands == 0:
                    # No candidates found (edge case)
                    results.append(
                        {"example_id": f"{example_id}_long", "PredictionString": ""}
                    )
                    results.append(
                        {"example_id": f"{example_id}_short", "PredictionString": ""}
                    )
                    continue

                # Expand question to match candidates
                q_input_expanded = q_input.repeat(num_cands, 1)

                # Forward Pass
                outputs = self.model(q_input_expanded, candidates)

                # 1. Ranking
                rank_logits = outputs["rank_logits"].squeeze(1)  # (N,)
                rank_probs = torch.sigmoid(rank_logits)

                best_score, best_idx_tensor = torch.max(rank_probs, dim=0)
                best_idx = best_idx_tensor.item()
                best_score = best_score.item()

                # Default predictions
                long_pred_str = ""
                short_pred_str = ""

                if best_score >= threshold:
                    # Retrieve raw candidate offsets from dataframe
                    raw_candidates = test_data_map.get(example_id, [])
                    if best_idx < len(raw_candidates):
                        cand_start_token, cand_end_token = raw_candidates[best_idx]

                        # Set Long Answer Prediction
                        long_pred_str = f"{cand_start_token}:{cand_end_token}"

                        # 2. Yes/No Check
                        yesno_logits = outputs["yesno_logits"][best_idx]  # (3,)
                        yesno_probs = F.softmax(yesno_logits, dim=0)
                        yesno_class = torch.argmax(yesno_probs).item()
                        yesno_label = self.idx_to_yesno.get(yesno_class, "NONE")

                        if yesno_label in ["YES", "NO"]:
                            short_pred_str = yesno_label
                        else:
                            # 3. Span Prediction
                            start_logits = outputs["start_logits"][
                                best_idx
                            ]  # (Ctx_Len,)
                            end_logits = outputs["end_logits"][best_idx]  # (Ctx_Len,)

                            rel_start, rel_end = self._get_best_span(
                                start_logits, end_logits
                            )

                            # Convert relative to absolute
                            # Note: rel_end is inclusive from model perspective (dataset logic)
                            # Output format usually expects start:end (exclusive?) or token indices.
                            # Based on sample submission `6:18` and task description "start:end token indices",
                            # and standard NQ eval, it's usually `start:end` where end is exclusive (Python slice style)
                            # or `start:end` inclusive?
                            # Looking at dataset.py: `rel_end = s_end - p_start - 1`.
                            # So `s_end` (exclusive) = `rel_end + p_start + 1`.

                            abs_start = cand_start_token + rel_start
                            abs_end = cand_start_token + rel_end + 1

                            # Ensure within bounds
                            if abs_start < cand_end_token and abs_end <= cand_end_token:
                                short_pred_str = f"{abs_start}:{abs_end}"

                # Append results
                results.append(
                    {
                        "example_id": f"{example_id}_long",
                        "PredictionString": long_pred_str,
                    }
                )
                results.append(
                    {
                        "example_id": f"{example_id}_short",
                        "PredictionString": short_pred_str,
                    }
                )

        return results

    def save_submission(self, results):
        """
        Saves the results to a CSV file.
        """
        Config.setup()  # Ensure dir exists
        df = pd.DataFrame(results)

        # Ensure correct column order
        df = df[["example_id", "PredictionString"]]

        print(f"Saving submission to {Config.SUBMISSION_FILE}...")
        df.to_csv(Config.SUBMISSION_FILE, index=False)
        print("Submission saved successfully.")


def run_inference():
    """
    Main entry point for inference.
    """
    # 1. Setup
    predictor = Predictor()

    # 2. Predict
    results = predictor.generate_predictions(threshold=Config.LONG_ANSWER_THRESHOLD)

    # 3. Save
    predictor.save_submission(results)
