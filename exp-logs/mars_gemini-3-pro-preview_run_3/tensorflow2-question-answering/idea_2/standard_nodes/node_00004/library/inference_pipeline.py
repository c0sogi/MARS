import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.config import Config
from library.text_processing import HTMLParser, build_vocab
from library.networks import SiameseTextCNN, AttentionMLPReader


class InferencePipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.ranker = None
        self.reader = None
        self.parser = HTMLParser()

        # Load resources
        self._load_resources()

    def _load_resources(self):
        """Loads vocabulary and trained models."""
        print("Loading Tokenizer...")
        self.tokenizer = build_vocab(load_cached_data=True)

        print(f"Loading Ranker model from {Config.RANKER_MODEL_PATH}...")
        self.ranker = SiameseTextCNN().to(self.device)
        if os.path.exists(Config.RANKER_MODEL_PATH):
            state_dict = torch.load(Config.RANKER_MODEL_PATH, map_location=self.device)
            self.ranker.load_state_dict(state_dict)
            self.ranker.eval()
        else:
            print(
                "Warning: Ranker model checkpoint not found. Using initialized weights."
            )

        print(f"Loading Reader model from {Config.READER_MODEL_PATH}...")
        self.reader = AttentionMLPReader().to(self.device)
        if os.path.exists(Config.READER_MODEL_PATH):
            state_dict = torch.load(Config.READER_MODEL_PATH, map_location=self.device)
            self.reader.load_state_dict(state_dict)
            self.reader.eval()
        else:
            print(
                "Warning: Reader model checkpoint not found. Using initialized weights."
            )

    def _predict_single_example(self, question_text, document_text, candidates_data):
        """
        Performs inference for a single example.
        Returns: (long_ans_str, short_ans_str)
        """
        # 1. Preprocessing
        # Tokenize Question
        q_seq = self.tokenizer.text_to_sequence(question_text)
        q_seq_padded = self.tokenizer.pad_sequence(q_seq, Config.MAX_Q_LEN)
        q_tensor = torch.tensor([q_seq_padded], dtype=torch.long).to(self.device)

        # Extract Candidates
        candidates = self.parser.extract_candidates(document_text, candidates_data)

        if not candidates:
            return "", ""

        # 2. Ranking
        # Prepare batch for ranker: (1 Question broadcasted, N Candidates)
        num_cands = len(candidates)

        # Tokenize all candidates
        cand_seqs = []
        for cand in candidates:
            seq = self.tokenizer.text_to_sequence(cand["text"])
            padded = self.tokenizer.pad_sequence(seq, Config.MAX_DOC_LEN)
            cand_seqs.append(padded)

        c_tensor = torch.tensor(cand_seqs, dtype=torch.long).to(self.device)
        # Broadcast question tensor to match number of candidates
        q_broadcast = q_tensor.repeat(num_cands, 1)

        with torch.no_grad():
            ranker_logits = self.ranker(q_broadcast, c_tensor)
            ranker_probs = torch.sigmoid(ranker_logits)

        # Get best candidate
        best_cand_idx = torch.argmax(ranker_probs).item()
        best_cand_prob = ranker_probs[best_cand_idx].item()
        best_candidate = candidates[best_cand_idx]

        # 3. Reading (Span Extraction)
        # Prepare input for reader (Question, Best Candidate)
        # We reuse q_tensor (1, Len) and select the specific candidate tensor (1, Len)
        c_best_tensor = c_tensor[best_cand_idx].unsqueeze(0)

        with torch.no_grad():
            start_logits, end_logits = self.reader(q_tensor, c_best_tensor)
            # Apply softmax to get probabilities over the sequence length
            start_probs = F.softmax(start_logits, dim=1)[0]  # Shape: (Seq_Len,)
            end_probs = F.softmax(end_logits, dim=1)[0]  # Shape: (Seq_Len,)

        # 4. Span Selection Logic
        # Find best (i, j) such that i <= j and j - i < MAX_ANSWER_LEN
        # We compute the outer product of probabilities to get a matrix of joint probs
        # joint_probs[i, j] = P_start(i) * P_end(j)
        joint_probs = torch.outer(start_probs, end_probs)  # Shape: (Seq_Len, Seq_Len)

        # Mask invalid spans
        seq_len = joint_probs.shape[0]
        # Create a mask where valid positions are 1, invalid are 0
        # triu(1) keeps upper triangle (j >= i)
        # tril(MAX_LEN) keeps band where j - i < MAX_LEN
        mask = torch.triu(torch.ones((seq_len, seq_len), device=self.device))
        mask = mask * torch.tril(
            torch.ones((seq_len, seq_len), device=self.device),
            diagonal=Config.MAX_ANSWER_LEN,
        )

        masked_probs = joint_probs * mask

        # Get best indices
        # flatten argmax
        best_flat_idx = torch.argmax(masked_probs).item()
        best_start_rel = best_flat_idx // seq_len
        best_end_rel = best_flat_idx % seq_len
        best_span_prob = masked_probs[best_start_rel, best_end_rel].item()

        # 5. Global Confidence & Thresholding
        final_score = best_cand_prob * best_span_prob

        long_ans_str = ""
        short_ans_str = ""

        if final_score >= Config.PREDICTION_THRESHOLD:
            # Construct Long Answer String: "start:end" (tokens)
            # NQ format usually expects token indices in the document
            la_start = best_candidate["start_token"]
            la_end = best_candidate["end_token"]
            long_ans_str = f"{la_start}:{la_end}"

            # Construct Short Answer String
            # Map relative indices to absolute document indices
            # Note: The model was trained with inclusive end index for classification.
            # Output format usually expects "start:end" where end is exclusive (Python style)
            # or inclusive? The task description says "start:end token indices".
            # Standard NQ evaluation usually treats the second number as exclusive.
            # Our training data processing: processed_rows[-1]["end_idx"] = rel_end - 1 (inclusive last token).
            # So `best_end_rel` is the index of the last token.
            # Absolute Start = la_start + best_start_rel
            # Absolute End (Exclusive) = la_start + best_end_rel + 1

            sa_abs_start = la_start + best_start_rel
            sa_abs_end = la_start + best_end_rel + 1

            # Ensure short answer is within long answer bounds
            if sa_abs_end <= la_end:
                short_ans_str = f"{sa_abs_start}:{sa_abs_end}"
            else:
                # Fallback: if prediction goes out of bounds, just return long answer or nothing
                short_ans_str = ""

        return long_ans_str, short_ans_str

    def run_inference(self, sample_size=None):
        """
        Runs the full inference pipeline on the test set and saves submission.csv.
        """
        print("Starting Inference...")

        # 1. Load Test Metadata
        if not os.path.exists(Config.TEST_METADATA_PATH):
            raise FileNotFoundError(
                f"Test metadata not found at {Config.TEST_METADATA_PATH}"
            )

        metadata = pd.read_csv(Config.TEST_METADATA_PATH)
        if sample_size:
            metadata = metadata.head(sample_size)

        test_file_path = None
        # Find the actual test file based on pattern in Config if not hardcoded
        # The metadata contains 'file_path', we can use the first row
        if not metadata.empty:
            fname = metadata.iloc[0]["file_path"]
            test_file_path = os.path.join(Config.INPUT_DIR, fname)

        if not test_file_path or not os.path.exists(test_file_path):
            # Fallback to finding it via glob if metadata path is stale or generic
            import glob

            files = glob.glob(os.path.join(Config.INPUT_DIR, Config.TEST_FILE_PATTERN))
            if files:
                test_file_path = files[0]
            else:
                raise FileNotFoundError("Could not locate test jsonl file.")

        results = []

        # 2. Iterate and Predict
        with open(test_file_path, "rb") as f:
            for i, row in metadata.iterrows():
                offset = row["byte_offset"]
                example_id = row["example_id"]

                f.seek(offset)
                line = f.readline()
                if not line:
                    continue

                try:
                    entry = json.loads(line.decode("utf-8"))

                    q_text = entry.get("question_text", "")
                    doc_text = entry.get("document_text", "")
                    candidates = entry.get("long_answer_candidates", [])

                    long_pred, short_pred = self._predict_single_example(
                        q_text, doc_text, candidates
                    )

                    # Append rows for submission format
                    # Format:
                    # example_id_long, prediction_string
                    # example_id_short, prediction_string

                    results.append(
                        {
                            "example_id": f"{example_id}_long",
                            "PredictionString": long_pred,
                        }
                    )
                    results.append(
                        {
                            "example_id": f"{example_id}_short",
                            "PredictionString": short_pred,
                        }
                    )

                except json.JSONDecodeError:
                    # In case of error, append empty predictions to maintain structure if needed
                    results.append(
                        {"example_id": f"{example_id}_long", "PredictionString": ""}
                    )
                    results.append(
                        {"example_id": f"{example_id}_short", "PredictionString": ""}
                    )
                    continue

        # 3. Save Submission
        submission_df = pd.DataFrame(results)
        Config.setup_directories()
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Processed {len(metadata)} examples.")


def generate_submission(sample_size=None):
    """
    Entry point for generating the submission file.
    """
    pipeline = InferencePipeline()
    pipeline.run_inference(sample_size=sample_size)
