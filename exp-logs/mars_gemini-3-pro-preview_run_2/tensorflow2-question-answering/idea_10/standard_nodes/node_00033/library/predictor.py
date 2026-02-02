import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from library.config import Config
from library.text_utils import build_or_load_tokenizer, segment_sentences, Tokenizer
from library.network import SentenceFactorizedModel
from library.data_loader import get_data_loader


class Predictor:
    def __init__(self, load_cached_data=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_cached_data = load_cached_data

        # 1. Load Tokenizer
        print("Loading tokenizer...")
        self.tokenizer = build_or_load_tokenizer(load_cached_data=load_cached_data)

        # 2. Load Model
        print("Loading model...")
        self.model = SentenceFactorizedModel(vocab_size=self.tokenizer.vocab_size)
        if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
            state_dict = torch.load(
                Config.MODEL_CHECKPOINT_PATH, map_location=self.device
            )
            self.model.load_state_dict(state_dict)
            print(f"Model weights loaded from {Config.MODEL_CHECKPOINT_PATH}")
        else:
            print("Warning: No model checkpoint found. Using random initialization.")

        self.model.to(self.device)
        self.model.eval()

        # 3. Build Raw Data Map (for span recovery)
        # We need this because the DataLoader provides token IDs, but we need
        # integer token indices (start:end) for the submission.
        self.raw_data_map = self._load_raw_test_data()

    def _load_raw_test_data(self):
        """
        Reads the raw test file to recover sentence spans and candidate spans.
        Returns a dict: example_id -> {
            'candidates': [{'start_token': int, 'end_token': int}, ...],
            'sentences': [{'start_token_idx': int, 'end_token_idx': int}, ...]
        }
        """
        print(
            f"Loading raw test data from {Config.TEST_DATA_PATH} for span recovery..."
        )
        data_map = {}

        if not os.path.exists(Config.TEST_DATA_PATH):
            print("Test data file not found. Skipping raw data map generation.")
            return data_map

        with open(Config.TEST_DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                ex_id = str(entry["example_id"])

                # Get Candidates
                candidates = entry.get("long_answer_candidates", [])

                # Get Sentence Spans
                doc_text = entry.get("document_text", "")
                sentences = segment_sentences(doc_text)

                # Truncate to match data loader logic
                if len(sentences) > Config.MAX_SENTS_PER_DOC:
                    sentences = sentences[: Config.MAX_SENTS_PER_DOC]

                # Store only necessary span info to save memory
                simple_candidates = [
                    {"start_token": c["start_token"], "end_token": c["end_token"]}
                    for c in candidates
                ]

                simple_sentences = [
                    {
                        "start_token_idx": s["start_token_idx"],
                        "end_token_idx": s["end_token_idx"],
                    }
                    for s in sentences
                ]

                data_map[ex_id] = {
                    "candidates": simple_candidates,
                    "sentences": simple_sentences,
                }

        print(f"Loaded span data for {len(data_map)} test examples.")
        return data_map

    def generate_submission(self, sample_size=None):
        """
        Runs inference and generates the submission.csv file.
        """
        print("Starting inference...")

        # Get DataLoader
        test_loader = get_data_loader(
            mode="test",
            tokenizer=self.tokenizer,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            sample_size=sample_size,
        )

        results = []

        # YES/NO Mapping: 0=NONE, 1=YES, 2=NO
        yn_map = {0: "", 1: "YES", 2: "NO"}

        with torch.no_grad():
            for batch in test_loader:
                # Move inputs to device
                questions = batch["questions"].to(self.device)
                sentences = batch["sentences"].to(self.device)
                doc_lengths = batch["doc_lengths"]
                example_ids = batch["example_ids"]
                candidate_maps = batch["candidate_maps"]  # List of lists of lists

                # Forward Pass
                # scores: (total_sentences,)
                # yn_logits: (batch_size, 3)
                scores, yn_logits = self.model(questions, sentences, doc_lengths)

                # Split scores back to documents
                scores_split = torch.split(scores, doc_lengths)

                # Process each document in the batch
                for i, ex_id in enumerate(example_ids):
                    doc_scores = scores_split[
                        i
                    ]  # Tensor of scores for sentences in this doc
                    doc_cand_map = candidate_maps[
                        i
                    ]  # List of [sent_indices] per candidate
                    doc_yn_logits = yn_logits[i]

                    # Retrieve raw spans
                    raw_info = self.raw_data_map.get(ex_id)
                    if not raw_info:
                        # Fallback if ID not found (shouldn't happen if files align)
                        results.append([f"{ex_id}_long", ""])
                        results.append([f"{ex_id}_short", ""])
                        continue

                    raw_candidates = raw_info["candidates"]
                    raw_sentences = raw_info["sentences"]

                    # 1. Determine Best Candidate (Long Answer)
                    best_cand_idx = -1
                    best_cand_score = -1.0
                    best_sent_idx_in_doc = -1

                    # Iterate over candidates to find max score
                    for c_idx, sent_indices in enumerate(doc_cand_map):
                        if not sent_indices:
                            continue

                        # Get scores for sentences in this candidate
                        # sent_indices are local to the document (0 to num_sents-1)
                        # ensure indices are valid
                        valid_indices = [
                            idx for idx in sent_indices if idx < len(doc_scores)
                        ]
                        if not valid_indices:
                            continue

                        cand_sent_scores = doc_scores[valid_indices]
                        max_score_val, max_score_arg = torch.max(
                            cand_sent_scores, dim=0
                        )
                        max_score = max_score_val.item()

                        if max_score > best_cand_score:
                            best_cand_score = max_score
                            best_cand_idx = c_idx
                            # The specific sentence that triggered this score
                            best_sent_idx_in_doc = valid_indices[max_score_arg.item()]

                    # 2. Apply Threshold logic
                    long_pred_str = ""
                    short_pred_str = ""

                    if (
                        best_cand_score >= Config.CONFIDENCE_THRESHOLD
                        and best_cand_idx != -1
                    ):
                        # Formulate Long Answer String
                        c_start = raw_candidates[best_cand_idx]["start_token"]
                        c_end = raw_candidates[best_cand_idx]["end_token"]
                        long_pred_str = f"{c_start}:{c_end}"

                        # 3. Determine Short Answer
                        # Check Yes/No prediction first
                        yn_pred_idx = torch.argmax(doc_yn_logits).item()
                        yn_str = yn_map.get(yn_pred_idx, "")

                        if yn_str != "":
                            short_pred_str = yn_str
                        elif best_sent_idx_in_doc != -1:
                            # Use the best sentence span
                            s_start = raw_sentences[best_sent_idx_in_doc][
                                "start_token_idx"
                            ]
                            s_end = raw_sentences[best_sent_idx_in_doc]["end_token_idx"]
                            short_pred_str = f"{s_start}:{s_end}"

                    # Append to results
                    results.append([f"{ex_id}_long", long_pred_str])
                    results.append([f"{ex_id}_short", short_pred_str])

        # Save Submission
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        sub_df = pd.DataFrame(results, columns=["example_id", "PredictionString"])
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission generated successfully.")


def predict_answers(load_cached_data=True, sample_size=None):
    """
    Wrapper function to execute the prediction pipeline.
    """
    predictor = Predictor(load_cached_data=load_cached_data)
    predictor.generate_submission(sample_size=sample_size)
