import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from library.config import Config
from library.text_utils import get_vocab_and_matrix
from library.ranker_model import DIPNRanker
from library.reader_model import QIRNReader
from library.data_loader import create_inference_data, pad_sequence


class PredictionPipeline:
    """
    End-to-end inference pipeline using trained Ranker and Reader models.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.vocab = None
        self.embedding_matrix = None
        self.ranker = None
        self.reader = None

    def load_resources(self, load_cached_data=True):
        """
        Loads vocabulary, embeddings, and model weights.
        """
        print("Loading resources for inference...")
        # Load Vocab and Embeddings
        self.vocab, self.embedding_matrix = get_vocab_and_matrix(
            texts=None, load_cached_data=load_cached_data
        )

        # Load Ranker
        print(f"Loading Ranker from {Config.RANKER_MODEL_PATH}")
        self.ranker = DIPNRanker(self.embedding_matrix).to(self.device)
        if os.path.exists(Config.RANKER_MODEL_PATH):
            self.ranker.load_state_dict(
                torch.load(Config.RANKER_MODEL_PATH, map_location=self.device)
            )
        else:
            print("Warning: Ranker model checkpoint not found. Using random weights.")
        self.ranker.eval()

        # Load Reader
        print(f"Loading Reader from {Config.READER_MODEL_PATH}")
        self.reader = QIRNReader(self.embedding_matrix).to(self.device)
        if os.path.exists(Config.READER_MODEL_PATH):
            self.reader.load_state_dict(
                torch.load(Config.READER_MODEL_PATH, map_location=self.device)
            )
        else:
            print("Warning: Reader model checkpoint not found. Using random weights.")
        self.reader.eval()

    def _get_best_span(self, start_probs, end_probs, max_span_len=30):
        """
        Finds the optimal span (start, end) maximizing P(start) * P(end).
        Constraints: start <= end, end - start < max_span_len.
        """
        # start_probs, end_probs: (Seq_Len,)
        # Create score matrix: score[i, j] = start[i] * end[j]
        # We use log probs for numerical stability: log(s) + log(e)
        # But inputs are probs, so just multiply.

        # Outer product: (Seq_Len, 1) * (1, Seq_Len) -> (Seq_Len, Seq_Len)
        score_mat = torch.ger(start_probs, end_probs)

        # Create mask for valid spans
        seq_len = len(start_probs)
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=self.device), diagonal=max_span_len
        )  # limit max len
        mask = torch.triu(mask, diagonal=0)  # start <= end

        # Apply mask
        score_mat = score_mat * mask

        # Find max
        max_score = score_mat.max()
        if max_score == 0:
            return 0, 0, 0.0

        flat_idx = score_mat.argmax()
        start_idx = flat_idx // seq_len
        end_idx = flat_idx % seq_len

        return start_idx.item(), end_idx.item(), max_score.item()

    def run_inference(self, test_metadata_path=Config.TEST_METADATA):
        """
        Generates predictions for the test set.
        """
        # Ensure resources are loaded
        if self.ranker is None:
            self.load_resources()

        print(f"Generating inference data from {test_metadata_path}...")
        # Note: create_inference_data is not cached via parquet due to complex nested structure,
        # but it uses the efficient metadata seeking method.
        inference_data = create_inference_data(test_metadata_path, self.vocab)

        results = []

        print(f"Running inference on {len(inference_data)} examples...")

        with torch.no_grad():
            for i, example in enumerate(inference_data):
                example_id = example["example_id"]
                q_indices = torch.tensor(example["q_indices"], dtype=torch.long).to(
                    self.device
                )
                candidates = example["candidates"]

                if not candidates:
                    results.append(
                        {
                            "example_id": example_id,
                            "long_answer": "",
                            "short_answer": "",
                        }
                    )
                    continue

                # --- 1. Ranker Step ---
                # Prepare batch: Question is repeated, Candidates vary
                q_batch = q_indices.unsqueeze(0).repeat(
                    len(candidates), 1
                )  # (Num_Cand, Q_Len)

                # Pad candidates
                p_seqs = [c["indices"] for c in candidates]
                p_batch = pad_sequence(p_seqs, max_len=Config.MAX_DOC_LEN).to(
                    self.device
                )

                # Forward Ranker
                ranker_logits = self.ranker(q_batch, p_batch)  # (Num_Cand,)
                ranker_probs = torch.sigmoid(ranker_logits)

                best_cand_idx = torch.argmax(ranker_probs).item()
                best_cand_score = ranker_probs[best_cand_idx].item()

                selected_candidate = candidates[best_cand_idx]

                # --- 2. Reader Step ---
                # Prepare input: Single Question, Single Best Paragraph
                q_input = q_indices.unsqueeze(0)  # (1, Q_Len)
                p_input = (
                    torch.tensor(selected_candidate["indices"], dtype=torch.long)
                    .unsqueeze(0)
                    .to(self.device)
                )  # (1, P_Len)

                # Forward Reader
                start_logits, end_logits = self.reader(q_input, p_input)

                # Softmax
                start_probs = torch.softmax(start_logits, dim=1).squeeze(0)
                end_probs = torch.softmax(end_logits, dim=1).squeeze(0)

                # Find best span
                rel_start, rel_end, span_score = self._get_best_span(
                    start_probs, end_probs
                )

                # --- 3. Logic & Thresholding ---
                long_ans_str = ""
                short_ans_str = ""

                # Check Long Answer Confidence
                if best_cand_score >= Config.CONFIDENCE_THRESHOLD:
                    # Construct Long Answer String: "start:end" (Global indices)
                    # Global start is token index in document
                    la_start = selected_candidate["global_start"]
                    la_end = selected_candidate["global_end"]
                    long_ans_str = f"{la_start}:{la_end}"

                    # Check Short Answer Confidence (Conditioned on Long Answer)
                    # Heuristic: Joint probability = RankerScore * ReaderSpanScore ?
                    # Or just ReaderSpanScore. Usually Reader score is enough if Ranker is confident.
                    if span_score >= Config.CONFIDENCE_THRESHOLD:
                        # Construct Short Answer String
                        # rel_end is inclusive in logic, but output format usually implies range.
                        # Standard: start_token:end_token (where end is exclusive in python slice,
                        # but NQ format usually asks for token indices.
                        # Looking at sample: "6:18".
                        # We will use global indices.
                        sa_start = la_start + rel_start
                        sa_end = (
                            la_start + rel_end + 1
                        )  # +1 for exclusive upper bound convention
                        short_ans_str = f"{sa_start}:{sa_end}"

                        # Handle YES/NO?
                        # Current architecture does not predict YES/NO class explicitly.
                        # We output span only.

                results.append(
                    {
                        "example_id": example_id,
                        "long_answer": long_ans_str,
                        "short_answer": short_ans_str,
                    }
                )

        return pd.DataFrame(results)

    def generate_submission(self, results_df):
        """
        Formats the results DataFrame into the submission format.
        Rows:
        {ID}_long, prediction
        {ID}_short, prediction
        """
        submission_rows = []
        for _, row in results_df.iterrows():
            eid = row["example_id"]

            # Long Answer Row
            submission_rows.append(
                {"example_id": f"{eid}_long", "PredictionString": row["long_answer"]}
            )

            # Short Answer Row
            # If short answer is empty, prediction is empty string (NaN in sample usually means blank)
            # Sample submission shows 'nan' or empty. We use empty string.
            submission_rows.append(
                {"example_id": f"{eid}_short", "PredictionString": row["short_answer"]}
            )

        sub_df = pd.DataFrame(submission_rows)

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")


def run_prediction_pipeline():
    pipeline = PredictionPipeline()
    pipeline.load_resources()

    # Run inference
    results_df = pipeline.run_inference()

    # Generate submission file
    pipeline.generate_submission(results_df)
