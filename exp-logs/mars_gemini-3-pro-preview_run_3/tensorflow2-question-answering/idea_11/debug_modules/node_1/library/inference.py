import os
import json
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.preprocessing import get_tokenizer, get_embedding_matrix
from library.dataset import get_test_dataset
from library.models import HistogramRanker, QRNNReader


class InferencePipeline:
    """
    Manages the inference process: loading data, running models,
    and generating the final submission file.
    """

    def __init__(self, load_cached_data=True):
        self.device = Config.DEVICE
        self.load_cached_data = load_cached_data

        print(f"Initializing Inference Pipeline on {self.device}...")

        # 1. Load Preprocessing Artifacts
        self.tokenizer = get_tokenizer(load_cached_data=load_cached_data)
        self.embedding_matrix = get_embedding_matrix(
            self.tokenizer, load_cached_data=load_cached_data
        )

        # 2. Initialize Models
        self.ranker = HistogramRanker(self.embedding_matrix).to(self.device)
        self.reader = QRNNReader(self.embedding_matrix).to(self.device)

        # 3. Load Model Weights
        self._load_weights()

        # Set to evaluation mode
        self.ranker.eval()
        self.reader.eval()

    def _load_weights(self):
        # Load Ranker
        if os.path.exists(Config.RANKER_MODEL_PATH):
            print(f"Loading Ranker weights from {Config.RANKER_MODEL_PATH}")
            state_dict = torch.load(Config.RANKER_MODEL_PATH, map_location=self.device)
            self.ranker.load_state_dict(state_dict)
        else:
            print("Warning: Ranker checkpoint not found. Using random initialization.")

        # Load Reader
        if os.path.exists(Config.READER_MODEL_PATH):
            print(f"Loading Reader weights from {Config.READER_MODEL_PATH}")
            state_dict = torch.load(Config.READER_MODEL_PATH, map_location=self.device)
            self.reader.load_state_dict(state_dict)
        else:
            print("Warning: Reader checkpoint not found. Using random initialization.")

    def get_best_span(self, start_probs, end_probs, q_len):
        """
        Finds the best valid text span (start, end) maximizing the joint probability.
        Constraints:
        1. start_index >= q_len (Answer must be in the document, not the question)
        2. start_index <= end_index
        3. end_index - start_index < MAX_ANSWER_LEN
        """
        best_score = -1.0
        best_start = -1
        best_end = -1

        # Convert tensors to lists for simple iteration
        start_probs = start_probs.tolist()
        end_probs = end_probs.tolist()
        seq_len = len(start_probs)

        # Iterate through all valid start positions
        for i in range(q_len, seq_len):
            # Optimization: Skip if start probability is negligible
            if start_probs[i] < 0.001:
                continue

            # Iterate through valid end positions
            max_j = min(i + Config.MAX_ANSWER_LEN, seq_len)
            for j in range(i, max_j):
                score = start_probs[i] * end_probs[j]

                if score > best_score:
                    best_score = score
                    best_start = i
                    best_end = j

        return best_start, best_end, best_score

    def predict(self):
        """
        Runs the inference loop over the test dataset.
        Returns a list of prediction dictionaries.
        """
        # Load Test Data
        test_dataset = get_test_dataset(
            self.tokenizer, load_cached_data=self.load_cached_data
        )
        results = []

        print(f"Running inference on {len(test_dataset)} samples...")

        with torch.no_grad():
            for idx in range(len(test_dataset)):
                sample = test_dataset[idx]
                example_id = sample["example_id"]
                q_ids = sample["q_ids"]
                candidates_json = sample["candidates"]

                candidates = json.loads(candidates_json)

                # Default predictions (Empty)
                long_ans_str = ""
                short_ans_str = ""

                if candidates:
                    # --- Step 1: Ranking ---
                    # Prepare batch: Question repeated for each candidate
                    num_cands = len(candidates)
                    q_tensor = torch.tensor([q_ids] * num_cands, dtype=torch.long).to(
                        self.device
                    )

                    # Pad candidate sequences
                    cand_seqs = [
                        torch.tensor(c["token_ids"], dtype=torch.long)
                        for c in candidates
                    ]
                    cand_tensor = torch.nn.utils.rnn.pad_sequence(
                        cand_seqs, batch_first=True, padding_value=0
                    ).to(self.device)

                    # Get Ranker Scores
                    rank_scores = self.ranker(q_tensor, cand_tensor)

                    # Select Best Candidate
                    best_idx = torch.argmax(rank_scores).item()
                    best_candidate = candidates[best_idx]

                    # Heuristic confidence from ranker (sigmoid of score)
                    ranker_conf = torch.sigmoid(rank_scores[best_idx]).item()

                    # --- Step 2: Reading ---
                    # Prepare input: Question + Best Candidate
                    curr_q_ids = q_ids[: Config.MAX_Q_LEN]
                    curr_p_ids = best_candidate["token_ids"][: Config.MAX_CTX_LEN]
                    input_ids = curr_q_ids + curr_p_ids

                    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(
                        self.device
                    )

                    # Get Reader Logits
                    start_logits, end_logits = self.reader(input_tensor)

                    # Softmax to get probabilities
                    start_probs = torch.softmax(start_logits, dim=1)[0]
                    end_probs = torch.softmax(end_logits, dim=1)[0]

                    # Find Best Span
                    q_len = len(curr_q_ids)
                    s_idx, e_idx, span_score = self.get_best_span(
                        start_probs, end_probs, q_len
                    )

                    # Combined Confidence
                    # We rely primarily on span_score but consider ranker confidence implicitly
                    # by selecting the best paragraph.

                    # --- Step 3: Thresholding & Formatting ---
                    if span_score > Config.CONFIDENCE_THRESHOLD:
                        # 1. Long Answer: The span of the selected paragraph in the document
                        la_start = best_candidate["start_token"]
                        la_end = best_candidate["end_token"]
                        long_ans_str = f"{la_start}:{la_end}"

                        # 2. Short Answer: Map relative indices back to document indices
                        if s_idx != -1:
                            # Relative offset in paragraph
                            rel_start = s_idx - q_len
                            rel_end = e_idx - q_len

                            # Absolute offset in document
                            doc_start = la_start + rel_start
                            # e_idx is inclusive in our logic, but output format usually expects
                            # exclusive end for range or inclusive token index.
                            # Based on NQ standard (start:end where end is exclusive), we add 1.
                            doc_end = la_start + rel_end + 1

                            short_ans_str = f"{doc_start}:{doc_end}"

                # Append results for both Long and Short answer IDs
                results.append(
                    {
                        "example_id": str(example_id) + "_long",
                        "PredictionString": long_ans_str,
                    }
                )
                results.append(
                    {
                        "example_id": str(example_id) + "_short",
                        "PredictionString": short_ans_str,
                    }
                )

        return results

    def generate_submission(self):
        """
        Runs prediction and saves the submission CSV.
        """
        predictions = self.predict()
        pred_df = pd.DataFrame(predictions)

        # Load sample submission to ensure all IDs are present and order is correct
        if os.path.exists(Config.SAMPLE_SUBMISSION_FILE):
            sample_df = pd.read_csv(Config.SAMPLE_SUBMISSION_FILE)
            # Merge predictions into sample layout
            final_df = sample_df[["example_id"]].merge(
                pred_df, on="example_id", how="left"
            )
            # Fill missing predictions with empty strings (blank prediction)
            final_df["PredictionString"] = final_df["PredictionString"].fillna("")
        else:
            print(
                "Sample submission file not found. Creating submission from predictions directly."
            )
            final_df = pred_df

        # Save
        final_df.to_csv(Config.FINAL_SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.FINAL_SUBMISSION_PATH}")
        print(f"Total rows: {len(final_df)}")
