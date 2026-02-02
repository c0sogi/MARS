import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import os
from library.config import Config
from library.data_utils import Tokenizer, HTMLParser
from library.models import SiameseRanker, SeparableConvReader


class QuestionAnsweringPredictor:
    """
    End-to-end predictor for the Question Answering task.
    Loads trained models and generates predictions for the test set.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)

        print(f"Initializing Predictor on {self.device}...")

        # 1. Load Tokenizer
        self.tokenizer = Tokenizer(config)
        if os.path.exists(config.VOCAB_CACHE_PATH):
            self.tokenizer.load(config.VOCAB_CACHE_PATH)
            print(f"Loaded vocabulary: {len(self.tokenizer.vocab)} tokens")
        else:
            raise FileNotFoundError(
                f"Vocabulary not found at {config.VOCAB_CACHE_PATH}"
            )

        # 2. Load Models
        self._load_models()

    def _load_models(self):
        # Load Ranker
        self.ranker = SiameseRanker(self.config).to(self.device)
        if os.path.exists(self.config.RANKER_MODEL_PATH):
            state_dict = torch.load(
                self.config.RANKER_MODEL_PATH, map_location=self.device
            )
            self.ranker.load_state_dict(state_dict)
            self.ranker.eval()
            print("Ranker model loaded.")
        else:
            print(
                f"Warning: Ranker model not found at {self.config.RANKER_MODEL_PATH}. Using random weights."
            )

        # Load Reader
        self.reader = SeparableConvReader(self.config).to(self.device)
        if os.path.exists(self.config.READER_MODEL_PATH):
            state_dict = torch.load(
                self.config.READER_MODEL_PATH, map_location=self.device
            )
            self.reader.load_state_dict(state_dict)
            self.reader.eval()
            print("Reader model loaded.")
        else:
            print(
                f"Warning: Reader model not found at {self.config.READER_MODEL_PATH}. Using random weights."
            )

    def _pad_sequence(self, id_list, max_len):
        """Pads a list of token IDs to max_len."""
        if len(id_list) > max_len:
            return id_list[:max_len]
        return id_list + [self.tokenizer.pad_token_id] * (max_len - len(id_list))

    def predict_single(self, data):
        """
        Generates prediction for a single JSON example.
        Returns: (long_answer_string, short_answer_string)
        """
        q_text = data["question_text"]
        doc_text = data["document_text"]
        q_tokens = q_text.split()
        doc_tokens = doc_text.split()
        candidates = HTMLParser.get_candidates(data)

        if not candidates:
            return "", ""

        # ---------------------------------------------------------
        # 1. Ranking Phase
        # ---------------------------------------------------------
        # Encode Question
        q_ids = self.tokenizer.encode(q_tokens, self.config.MAX_Q_LEN)
        q_tensor = torch.tensor([q_ids], dtype=torch.long).to(self.device)

        # Encode Candidates
        cand_tensors = []
        valid_candidates = []

        for cand in candidates:
            cand_text = HTMLParser.extract_candidate_text(doc_tokens, cand)
            # Skip empty candidates
            if not cand_text:
                continue

            cand_ids = self.tokenizer.encode(cand_text, self.config.MAX_DOC_LEN)
            cand_tensors.append(cand_ids)
            valid_candidates.append(cand)

        if not valid_candidates:
            return "", ""

        cand_tensor = torch.tensor(cand_tensors, dtype=torch.long).to(self.device)

        with torch.no_grad():
            # Efficient Ranking: Encode Q once, Encode all Cands, Dot Product
            q_vec = self.ranker.forward_one(q_tensor)  # (1, D)
            cand_vecs = self.ranker.forward_one(cand_tensor)  # (N, D)

            # Scores: (N,)
            scores = torch.sum(cand_vecs * q_vec, dim=1)
            best_cand_idx = torch.argmax(scores).item()

        best_candidate = valid_candidates[best_cand_idx]
        best_cand_text = HTMLParser.extract_candidate_text(doc_tokens, best_candidate)

        # ---------------------------------------------------------
        # 2. Reading Phase
        # ---------------------------------------------------------
        # Prepare Input: [Q] + [Best Candidate]
        combined_tokens = q_tokens + best_cand_text

        # Truncate if needed (though tokenizer.encode handles it, we need len for offset)
        if len(combined_tokens) > self.config.MAX_READER_SEQ_LEN:
            combined_tokens = combined_tokens[: self.config.MAX_READER_SEQ_LEN]

        input_ids = self.tokenizer.encode(
            combined_tokens, self.config.MAX_READER_SEQ_LEN
        )
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            start_logits, end_logits = self.reader(input_tensor)

            # Softmax to get probabilities
            start_probs = F.softmax(start_logits, dim=1)[0]  # (L,)
            end_probs = F.softmax(end_logits, dim=1)[0]  # (L,)

        # Find best span
        # Constraint: start < end (since end is exclusive in NQ logic derived from training data)
        # We construct a score matrix: score[i, j] = p_start[i] * p_end[j]
        # We only look at the upper triangle where j > i

        # Create grid of scores
        score_mat = torch.outer(start_probs, end_probs)  # (L, L)

        # Mask invalid spans (end <= start)
        # triu(..., 1) keeps diagonal offset 1 (j >= i+1), i.e., j > i
        score_mat = torch.triu(score_mat, diagonal=1)

        # Get max
        max_score = score_mat.max().item()

        # ---------------------------------------------------------
        # 3. Thresholding & Formatting
        # ---------------------------------------------------------
        if max_score < self.config.CONFIDENCE_THRESHOLD:
            return "", ""

        # Retrieve indices
        flat_idx = score_mat.argmax().item()
        # unravel_index equivalent for 2D
        L = score_mat.shape[0]
        pred_start = flat_idx // L
        pred_end = flat_idx % L

        # Map back to document indices
        # The input was [Q] + [Cand]
        # Indices 0 to len(q_tokens)-1 are Question
        # Indices len(q_tokens) to ... are Candidate

        q_len = len(q_tokens)

        # If prediction is inside the question part, ignore (or treat as null)
        if pred_start < q_len:
            return "", ""

        # Relative to candidate start
        rel_start = pred_start - q_len
        rel_end = pred_end - q_len

        # Absolute document indices
        cand_doc_start = best_candidate["start_token"]

        final_short_start = cand_doc_start + rel_start
        final_short_end = cand_doc_start + rel_end

        # Long Answer is the candidate itself
        final_long_start = best_candidate["start_token"]
        final_long_end = best_candidate["end_token"]

        long_ans_str = f"{final_long_start}:{final_long_end}"
        short_ans_str = f"{final_short_start}:{final_short_end}"

        return long_ans_str, short_ans_str

    def generate_submission(self):
        """
        Main inference loop. Generates predictions for the test set and saves to CSV.
        """
        print("Generating submission...")

        # Load Test Metadata
        if not os.path.exists(self.config.TEST_METADATA_PATH):
            raise FileNotFoundError(
                f"Test metadata not found at {self.config.TEST_METADATA_PATH}"
            )

        test_meta = pd.read_csv(self.config.TEST_METADATA_PATH)

        # Debugging option
        if self.config.DEBUG_SAMPLE_SIZE:
            print(
                f"Debug mode: Processing first {self.config.DEBUG_SAMPLE_SIZE} examples."
            )
            test_meta = test_meta.head(self.config.DEBUG_SAMPLE_SIZE)

        results = []

        with open(self.config.TEST_FILE, "rb") as f:
            for _, row in test_meta.iterrows():
                example_id = row["example_id"]
                offset = row["byte_offset"]

                # Load JSON
                f.seek(offset)
                line = f.readline()
                if not line:
                    continue

                try:
                    data = json.loads(line.decode("utf-8"))
                    # Cite debug_lesson_1: Prioritize Raw Data Over Derived Metadata for Critical Identifiers
                    if "example_id" in data:
                        example_id = str(data["example_id"])
                    long_pred, short_pred = self.predict_single(data)
                except Exception as e:
                    print(f"Error processing {example_id}: {e}")
                    long_pred, short_pred = "", ""

                # Append Long Answer Row
                results.append(
                    {"example_id": f"{example_id}_long", "PredictionString": long_pred}
                )

                # Append Short Answer Row
                results.append(
                    {
                        "example_id": f"{example_id}_short",
                        "PredictionString": short_pred,
                    }
                )

        # Create DataFrame and Save
        submission_df = pd.DataFrame(results)
        submission_df.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(
            f"Submission saved to {self.config.SUBMISSION_PATH}. Total rows: {len(submission_df)}"
        )


def run_inference():
    config = Config()
    predictor = QuestionAnsweringPredictor(config)
    predictor.generate_submission()
