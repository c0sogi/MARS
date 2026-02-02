import os
import csv
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.data_utils import Tokenizer, build_embedding_matrix
from library.dataset import NQDataset
from library.model import GlobalContextPointwiseNet


class Evaluator:
    def __init__(self, load_cached_data=True):
        """
        Initializes the Evaluator with model, tokenizer, and resources.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Evaluator initialized on device: {self.device}")
        self.load_cached_data = load_cached_data

        # 1. Load Tokenizer
        self.tokenizer = Tokenizer()
        if os.path.exists(Config.VOCAB_CACHE_FILE) and load_cached_data:
            self.tokenizer.load(Config.VOCAB_CACHE_FILE)
        else:
            print(
                f"Warning: Vocab file not found at {Config.VOCAB_CACHE_FILE}. Inference may fail if not built."
            )

        # 2. Build/Load Embeddings
        self.embedding_matrix = build_embedding_matrix(
            self.tokenizer.word_index,
            embedding_dim=Config.EMBEDDING_DIM,
            load_cached_data=load_cached_data,
        )

        # 3. Initialize Model
        self.model = GlobalContextPointwiseNet(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=Config.EMBEDDING_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            dropout_rate=Config.DROPOUT_RATE,
            embedding_matrix=self.embedding_matrix,
        ).to(self.device)

        # 4. Load Weights
        model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model weights from {model_path}")
        else:
            print("Warning: No trained model weights found. Using initialized weights.")

        self.model.eval()

    def _get_candidate_offsets(self, load_cached_data):
        """
        Retrieves candidate offsets (start/end tokens) for test examples.
        Implements caching using Parquet.
        """
        cache_file = os.path.join(Config.WORKING_DIR, "test_candidate_offsets.parquet")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading candidate offsets from {cache_file}...")
            try:
                df = pd.read_parquet(cache_file)
                # Convert back to nested dictionary structure for O(1) lookup
                # Structure: {example_id: {candidate_index: {'start': s, 'end': e}}}
                offsets_map = {}
                for _, row in df.iterrows():
                    eid = str(row["example_id"])
                    c_idx = int(row["candidate_index"])
                    if eid not in offsets_map:
                        offsets_map[eid] = {}
                    offsets_map[eid][c_idx] = {
                        "start_token": int(row["start_token"]),
                        "end_token": int(row["end_token"]),
                    }
                return offsets_map
            except Exception as e:
                print(f"Error loading offsets cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print("Extracting candidate offsets from raw test file...")
        data_rows = []

        with open(Config.TEST_DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                eid = str(entry["example_id"])
                candidates = entry["long_answer_candidates"]

                for i, cand in enumerate(candidates):
                    data_rows.append(
                        {
                            "example_id": eid,
                            "candidate_index": i,
                            "start_token": cand["start_token"],
                            "end_token": cand["end_token"],
                        }
                    )

        df = pd.DataFrame(data_rows)

        # 3. Save to cache
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        df.to_parquet(cache_file, index=False)
        print(f"Saved candidate offsets to {cache_file}")

        # Convert to dict
        offsets_map = {}
        for _, row in df.iterrows():
            eid = str(row["example_id"])
            c_idx = int(row["candidate_index"])
            if eid not in offsets_map:
                offsets_map[eid] = {}
            offsets_map[eid][c_idx] = {
                "start_token": int(row["start_token"]),
                "end_token": int(row["end_token"]),
            }

        return offsets_map

    def generate_submission(self):
        """
        Runs inference on the test set and generates the submission CSV.
        """
        # Prepare Test Dataset
        test_dataset = NQDataset(
            metadata_path=Config.TEST_META_PATH,
            raw_data_path=Config.TEST_DATA_PATH,
            tokenizer=self.tokenizer,
            is_train=False,
            load_cached_data=self.load_cached_data,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
        )

        # Get Offsets for mapping back to global indices
        candidate_offsets = self._get_candidate_offsets(self.load_cached_data)

        # Store best results per example_id
        results = {}

        print("Running inference on test set...")
        with torch.no_grad():
            for batch in test_loader:
                q_seq = batch["q_seq"].to(self.device)
                c_seq = batch["c_seq"].to(self.device)
                example_ids = batch["example_id"]
                cand_indices = batch["candidate_index"].numpy()

                l_logits, s_logits, e_logits, yn_logits = self.model(q_seq, c_seq)

                l_probs = torch.sigmoid(l_logits).cpu().numpy()
                s_probs = F.softmax(s_logits, dim=1).cpu().numpy()
                e_probs = F.softmax(e_logits, dim=1).cpu().numpy()
                yn_probs = F.softmax(yn_logits, dim=1).cpu().numpy()

                for i, eid in enumerate(example_ids):
                    eid = str(eid)
                    c_idx = cand_indices[i]
                    l_score = l_probs[i]

                    # Initialize if new
                    if eid not in results:
                        results[eid] = {
                            "best_score": -1.0,
                            "long_ans": "",
                            "short_ans": "",
                        }

                    # Update if this candidate has a higher long answer score
                    if l_score > results[eid]["best_score"]:

                        # Get global offsets
                        # candidate_offsets structure: {eid: {c_idx: {'start_token': x, 'end_token': y}}}
                        cand_info = candidate_offsets.get(eid, {}).get(c_idx)
                        if not cand_info:
                            continue

                        global_c_start = cand_info["start_token"]
                        global_c_end = cand_info["end_token"]

                        # 1. Long Answer
                        long_ans_str = ""
                        if l_score > Config.LONG_CONFIDENCE_THRESHOLD:
                            long_ans_str = f"{global_c_start}:{global_c_end}"

                        # 2. Short Answer
                        short_ans_str = ""
                        # Only predict short if long is confident enough
                        if l_score > Config.LONG_CONFIDENCE_THRESHOLD:
                            # Check Yes/No
                            yn_idx = np.argmax(yn_probs[i])

                            if yn_idx == 1:  # YES
                                short_ans_str = "YES"
                            elif yn_idx == 2:  # NO
                                short_ans_str = "NO"
                            else:
                                # Span prediction
                                s_idx = np.argmax(s_probs[i])
                                e_idx = np.argmax(e_probs[i])

                                # Validate span (start <= end)
                                if s_idx <= e_idx:
                                    span_score = s_probs[i][s_idx] * e_probs[i][e_idx]

                                    if span_score > Config.SHORT_CONFIDENCE_THRESHOLD:
                                        # Convert to global indices
                                        # s_idx is relative to candidate start
                                        global_s_start = global_c_start + s_idx
                                        # e_idx is relative to candidate start.
                                        # Output format is token indices.
                                        # If e_idx points to the last token, the range usually implies exclusive end in some formats,
                                        # but NQ evaluation often treats "start:end" as token indices.
                                        # The sample submission uses "6:18".
                                        # Assuming standard python slice notation (inclusive start, exclusive end).
                                        global_s_end = global_c_start + e_idx + 1
                                        short_ans_str = (
                                            f"{global_s_start}:{global_s_end}"
                                        )

                        results[eid] = {
                            "best_score": l_score,
                            "long_ans": long_ans_str,
                            "short_ans": short_ans_str,
                        }

        # Generate Submission File
        print("Generating submission file...")
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        submission_data = []
        # Iterate over sample submission to ensure all IDs are present and in order
        for _, row in sample_sub.iterrows():
            full_id = row["example_id"]
            # Extract base ID (e.g., "-12345_long" -> "-12345")
            if "_long" in full_id:
                eid = full_id.replace("_long", "")
                pred_type = "long"
            else:
                eid = full_id.replace("_short", "")
                pred_type = "short"

            prediction = ""
            if eid in results:
                if pred_type == "long":
                    prediction = results[eid]["long_ans"]
                else:
                    prediction = results[eid]["short_ans"]

            submission_data.append(
                {"example_id": full_id, "PredictionString": prediction}
            )

        submission_df = pd.DataFrame(submission_data)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        # Use strict quoting to prevent CSV parsing errors (Cite debug_lesson_1)
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False, quoting=csv.QUOTE_ALL)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
