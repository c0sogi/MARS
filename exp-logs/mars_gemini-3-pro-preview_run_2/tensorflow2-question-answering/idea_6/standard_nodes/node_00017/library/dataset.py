import os
import torch
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset
from library.config import Config
from library.utils import load_jsonl
from library.preprocessing import Tokenizer


class NQDataset(Dataset):
    def __init__(
        self,
        mode="train",
        tokenizer=None,
        sample_size=0,
        load_cached_data=True,
        expand_candidates=False,
    ):
        """
        PyTorch Dataset for Natural Questions.

        Args:
            mode (str): 'train', 'val', or 'test'.
            tokenizer (Tokenizer): Instance of library.preprocessing.Tokenizer.
            sample_size (int): Limit the number of examples (for debugging).
            load_cached_data (bool): Whether to load features from cache.
            expand_candidates (bool): If True, flattens the dataset so each item is a (Question, Candidate) pair.
                                      Used for evaluation/inference. If False, performs negative sampling (Training).
        """
        self.mode = mode
        self.tokenizer = tokenizer
        self.expand_candidates = expand_candidates
        self.sample_size = sample_size

        # Determine paths based on mode
        if mode == "train":
            self.metadata_path = Config.TRAIN_META
            self.jsonl_path = Config.TRAIN_FILE
            self.cache_path = Config.TRAIN_FEATURES_CACHE
        elif mode == "val":
            self.metadata_path = Config.VAL_META
            self.jsonl_path = Config.TRAIN_FILE  # Val is subset of train file
            self.cache_path = Config.VAL_FEATURES_CACHE
        elif mode == "test":
            self.metadata_path = Config.TEST_META
            self.jsonl_path = Config.TEST_FILE
            self.cache_path = Config.TEST_FEATURES_CACHE
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Load or Compute Data
        self.data = self._load_or_create_data(load_cached_data)

        # If expanding candidates (for inference/eval), we need a mapping
        if self.expand_candidates:
            self.expanded_indices = []
            for i, row in self.data.iterrows():
                candidates = row["candidates"]
                # candidates is a numpy array of dicts or list of dicts
                # In parquet read, it comes as array of structs/maps usually, or list
                for c_idx in range(len(candidates)):
                    self.expanded_indices.append((i, c_idx))

    def _load_or_create_data(self, load_cached_data):
        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                # print(f"Loading {self.mode} features from {self.cache_path}...")
                df = pd.read_parquet(self.cache_path)
                if self.sample_size > 0:
                    df = df.head(self.sample_size)
                return df
            except Exception:
                pass  # Fallback to compute

        # 2. Compute from Scratch
        # print(f"Processing {self.mode} data from scratch...")

        # Load Metadata to filter/order
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        meta_df = pd.read_csv(self.metadata_path)

        # Create a set of relevant IDs for O(1) lookup
        relevant_ids = set(meta_df["example_id"].astype(str))

        # We need to preserve metadata info (labels)
        meta_dict = meta_df.set_index("example_id").to_dict("index")

        processed_rows = []

        # Iterate through JSONL
        count = 0
        # Determine limit for loading if sample size is set and we are creating cache
        # Note: We must scan to find relevant IDs, so we can't strictly limit load_jsonl
        # unless we know the file is sorted same as metadata (which it isn't strictly for val).
        # We iterate all but stop storing if we hit sample_size limit.

        for entry in load_jsonl(self.jsonl_path):
            eid = str(entry["example_id"])

            if eid not in relevant_ids:
                continue

            # Tokenize Question
            q_text = entry.get("question_text", "")
            q_ids = self.tokenizer.encode(q_text, Config.MAX_Q_LEN)

            # Tokenize Document
            doc_text = entry.get("document_text", "")
            doc_tokens = doc_text.split()
            doc_ids = [
                self.tokenizer.token_to_id.get(t, self.tokenizer.unk_id)
                for t in doc_tokens
            ]

            # Process Candidates
            raw_candidates = entry.get("long_answer_candidates", [])
            candidates_list = []
            for c in raw_candidates:
                if c.get("top_level", False):
                    candidates_list.append(
                        {"start_token": c["start_token"], "end_token": c["end_token"]}
                    )

            # Retrieve Labels from Metadata (if available)
            meta_row = meta_dict.get(eid, {})
            long_idx = meta_row.get("long_answer_index", -1)
            yes_no = meta_row.get("yes_no_answer", "NONE")

            # For short answers, we need the raw annotations from JSONL to get spans
            short_spans = []
            annotations = entry.get("annotations", [])
            if annotations:
                ann = annotations[0]
                shorts = ann.get("short_answers", [])
                for s in shorts:
                    short_spans.append([s["start_token"], s["end_token"]])

            processed_rows.append(
                {
                    "example_id": eid,
                    "q_ids": q_ids,
                    "doc_ids": doc_ids,
                    "candidates": candidates_list,
                    "long_idx": long_idx,
                    "short_spans": short_spans,
                    "yes_no": yes_no,
                }
            )

            count += 1
            if self.sample_size > 0 and count >= self.sample_size:
                break

        # Create DataFrame
        df = pd.DataFrame(processed_rows)

        # Save to Cache
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        df.to_parquet(self.cache_path, index=False)

        return df

    def __len__(self):
        if self.expand_candidates:
            return len(self.expanded_indices)
        return len(self.data)

    def __getitem__(self, idx):
        if self.expand_candidates:
            # Inference/Eval Mode: Return specific (Question, Candidate) pair
            row_idx, cand_idx = self.expanded_indices[idx]
            row = self.data.iloc[row_idx]

            q_ids = np.array(row["q_ids"], dtype=np.int64)
            doc_ids = row["doc_ids"]
            candidate = row["candidates"][cand_idx]

            # Extract Candidate Tokens
            start = candidate["start_token"]
            end = candidate["end_token"]

            # Safe slicing
            if start < len(doc_ids):
                c_ids_seq = doc_ids[start:end]
            else:
                c_ids_seq = []

            # Pad Candidate
            c_ids = np.zeros(Config.MAX_C_LEN, dtype=np.int64)
            # Fill with pad_id (0) is default for zeros, assuming pad_id is 0
            length = min(len(c_ids_seq), Config.MAX_C_LEN)
            if length > 0:
                c_ids[:length] = c_ids_seq[:length]

            # Metadata for reconstruction
            example_id = row["example_id"]

            return {
                "q_ids": torch.tensor(q_ids, dtype=torch.long),
                "c_ids": torch.tensor(c_ids, dtype=torch.long),
                "example_id": str(example_id),
                "candidate_index": cand_idx,
                "token_start": start,
                "token_end": end,
            }

        else:
            # Training Mode: Sampling
            row = self.data.iloc[idx]

            q_ids = np.array(row["q_ids"], dtype=np.int64)
            doc_ids = row["doc_ids"]
            candidates = row["candidates"]
            true_long_idx = row["long_idx"]
            short_spans = row["short_spans"]  # List of [start, end]
            yes_no = row["yes_no"]

            is_positive_sample = False
            selected_cand_idx = -1

            # Check if there is a valid long answer
            has_positive = (true_long_idx != -1) and (true_long_idx < len(candidates))

            if has_positive and random.random() < 0.5:
                # Sample Positive
                selected_cand_idx = true_long_idx
                is_positive_sample = True
            else:
                # Sample Negative
                neg_indices = [i for i in range(len(candidates)) if i != true_long_idx]
                if not neg_indices:
                    # Fallback if only 1 candidate and it's positive
                    selected_cand_idx = true_long_idx
                    is_positive_sample = True
                else:
                    selected_cand_idx = random.choice(neg_indices)
                    is_positive_sample = False

            if len(candidates) > 0:
                candidate = candidates[selected_cand_idx]
                c_start = candidate["start_token"]
                c_end = candidate["end_token"]

                # Extract and Pad Candidate Sequence
                if c_start < len(doc_ids):
                    c_ids_seq = doc_ids[c_start:c_end]
                else:
                    c_ids_seq = []
            else:
                # Edge case: no candidates
                c_ids_seq = []
                c_start = 0
                c_end = 0

            c_ids = np.zeros(Config.MAX_C_LEN, dtype=np.int64)
            length = min(len(c_ids_seq), Config.MAX_C_LEN)
            if length > 0:
                c_ids[:length] = c_ids_seq[:length]

            # --- Targets ---

            # 1. Rank Label (Binary)
            rank_label = 1.0 if is_positive_sample else 0.0

            # 2. Yes/No Label (Categorical)
            yn_map = {"NONE": 0, "YES": 1, "NO": 2}
            yn_label = yn_map.get(yes_no, 0) if is_positive_sample else 0

            # 3. Short Answer Span (Start/End relative to candidate)
            span_start = 0
            span_end = 0

            if is_positive_sample and len(short_spans) > 0:
                # Find a short span that falls within this candidate
                for s_span in short_spans:
                    s_abs_start, s_abs_end = s_span
                    if s_abs_start >= c_start and s_abs_end <= c_end:
                        # Convert to relative
                        rel_start = s_abs_start - c_start
                        rel_end = s_abs_end - c_start

                        # Check bounds (truncation)
                        if rel_end < Config.MAX_C_LEN:
                            span_start = rel_start
                            span_end = rel_end
                            break

            return {
                "q_ids": torch.tensor(q_ids, dtype=torch.long),
                "c_ids": torch.tensor(c_ids, dtype=torch.long),
                "rank_label": torch.tensor(rank_label, dtype=torch.float),
                "span_start": torch.tensor(span_start, dtype=torch.long),
                "span_end": torch.tensor(span_end, dtype=torch.long),
                "yn_label": torch.tensor(yn_label, dtype=torch.long),
            }


def process_data(load_cached_data=True):
    """
    Helper function to initialize tokenizer and ensure data is ready.
    """
    tokenizer = Tokenizer()
    tokenizer.fit(Config.TRAIN_FILE, load_cached_data=load_cached_data)
    return tokenizer
