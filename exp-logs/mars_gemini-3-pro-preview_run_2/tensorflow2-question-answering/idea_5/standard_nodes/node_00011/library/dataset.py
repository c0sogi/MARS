import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config
from library.data_processing import DataProcessor


class NQDataset(Dataset):
    """
    PyTorch Dataset for Natural Questions.
    Handles loading, caching, and processing of Question-Candidate pairs.
    """

    def __init__(
        self,
        config: Config,
        processor: DataProcessor,
        split: str = "train",
        load_cached_data: bool = True,
    ):
        """
        Args:
            config: Configuration object.
            processor: DataProcessor object for tokenization/vocab.
            split: 'train', 'val', or 'test'.
            load_cached_data: Whether to load pre-processed parquet cache.
        """
        self.config = config
        self.processor = processor
        self.split = split
        self.load_cached_data = load_cached_data

        # 1. Load Metadata
        if split == "train":
            self.meta_path = config.TRAIN_META_PATH
        elif split == "val":
            self.meta_path = config.VAL_META_PATH
        else:
            self.meta_path = config.TEST_META_PATH

        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

        self.metadata = pd.read_csv(self.meta_path)

        # Apply Debug Limit
        if self.config.DEBUG_SAMPLE_SIZE:
            print(
                f"[{split}] Debug mode: Limiting to {self.config.DEBUG_SAMPLE_SIZE} samples."
            )
            self.metadata = self.metadata.head(self.config.DEBUG_SAMPLE_SIZE)

        # 2. Load Data (with Caching)
        self.data_df = self._load_data()

        # Create fast lookup map
        self.id_to_data = {
            str(row["example_id"]): row for _, row in self.data_df.iterrows()
        }

        # Yes/No Label Mapping
        self.yn_map = {"YES": 0, "NO": 1, "NONE": 2}

    def _load_data(self):
        """
        Loads raw JSONL data, filters by metadata IDs, and caches to Parquet.
        """
        cache_file = f"cached_{self.split}.parquet"
        cache_path = os.path.join(self.config.WORKING_DIR, cache_file)

        # Try loading cache
        if self.load_cached_data and os.path.exists(cache_path):
            print(f"[{self.split}] Loading cached data from {cache_path}...")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"[{self.split}] Failed to load cache: {e}. Recomputing...")

        # Compute from scratch
        print(f"[{self.split}] Processing raw data from scratch...")

        # Identify source file
        # Note: Validation split comes from train file, Test split from test file
        if self.split in ["train", "val"]:
            source_path = self.config.TRAIN_DATA_PATH
        else:
            source_path = self.config.TEST_DATA_PATH

        target_ids = set(self.metadata["example_id"].astype(str))

        data_rows = []
        chunksize = 5000

        # Read JSONL
        reader = pd.read_json(source_path, lines=True, chunksize=chunksize)

        for chunk in reader:
            chunk["example_id"] = chunk["example_id"].astype(str)
            # Filter rows that exist in our metadata split
            filtered_chunk = chunk[chunk["example_id"].isin(target_ids)]

            if filtered_chunk.empty:
                continue

            for _, row in filtered_chunk.iterrows():
                # Serialize nested structures to string for Parquet storage
                # This avoids pickle and complex object types
                record = {
                    "example_id": str(row["example_id"]),
                    "question_text": row["question_text"],
                    "document_text": row["document_text"],
                    "long_answer_candidates": json.dumps(row["long_answer_candidates"]),
                    # Test set might not have annotations
                    "annotations": (
                        json.dumps(row["annotations"]) if "annotations" in row else "[]"
                    ),
                }
                data_rows.append(record)

        df = pd.DataFrame(data_rows)

        # Save cache
        print(f"[{self.split}] Saving cache to {cache_path}...")
        df.to_parquet(cache_path)

        return df

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns a list of dictionaries, where each dictionary is a (Question, Candidate) pair.
        """
        # Get metadata info
        meta_row = self.metadata.iloc[idx]
        ex_id = str(meta_row["example_id"])

        # Retrieve data
        if ex_id not in self.id_to_data:
            # Should not happen if logic is correct
            raise IndexError(f"Example ID {ex_id} not found in loaded data.")

        data_row = self.id_to_data[ex_id]

        # Parse fields
        question_text = data_row["question_text"]
        doc_text = data_row["document_text"]
        candidates = json.loads(data_row["long_answer_candidates"])
        annotations = json.loads(data_row["annotations"])

        # Tokenize Document (Whitespace split as per NQ)
        # We need this to extract candidate text by token indices
        doc_tokens = doc_text.split()

        # Tokenize Question
        q_indices = self.processor.text_to_indices(question_text, self.config.Q_MAX_LEN)

        # Determine targets (only for train/val)
        pos_cand_idx = -1
        short_start_token = -1
        short_end_token = -1
        yn_label_str = "NONE"

        if self.split in ["train", "val"]:
            # Use metadata for fast label access
            pos_cand_idx = int(meta_row["long_answer_index"])
            yn_label_str = meta_row["yes_no_answer"]

            # Extract short answer tokens if they exist
            if meta_row["has_short_answer"] and len(annotations) > 0:
                short_anns = annotations[0].get("short_answers", [])
                if short_anns:
                    short_start_token = short_anns[0]["start_token"]
                    short_end_token = short_anns[0]["end_token"]

        # Select Candidates
        selected_candidates = []

        if self.split == "train":
            # Training: Positive + Negative Sampling
            if pos_cand_idx != -1 and pos_cand_idx < len(candidates):
                # Add Positive
                selected_candidates.append(
                    (candidates[pos_cand_idx], 1)
                )  # (candidate, is_positive)

            # Add Negatives
            # Get indices of all candidates except positive
            all_indices = list(range(len(candidates)))
            if pos_cand_idx != -1:
                if pos_cand_idx in all_indices:
                    all_indices.remove(pos_cand_idx)

            # Sample negatives
            if all_indices:
                # If we have a positive, we need NEG_SAMPLE_RATIO negatives
                # If we don't have a positive, we might take NEG_SAMPLE_RATIO + 1 negatives
                # (though usually we focus on positive examples)
                num_negs = self.config.NEG_SAMPLE_RATIO
                if len(all_indices) > num_negs:
                    neg_indices = np.random.choice(all_indices, num_negs, replace=False)
                else:
                    neg_indices = all_indices

                for ni in neg_indices:
                    selected_candidates.append((candidates[ni], 0))
        else:
            # Val/Test: Use ALL candidates for ranking
            for c in candidates:
                # Label is 1 if it matches pos_cand_idx, else 0 (useful for Val evaluation)
                label = (
                    1
                    if (pos_cand_idx != -1 and candidates.index(c) == pos_cand_idx)
                    else 0
                )
                selected_candidates.append((c, label))

        # Process selected candidates
        samples = []

        for cand, long_label in selected_candidates:
            # Extract text
            c_start = cand["start_token"]
            c_end = cand["end_token"]

            # Safety check
            c_start = max(0, min(c_start, len(doc_tokens)))
            c_end = max(0, min(c_end, len(doc_tokens)))

            cand_tokens_list = doc_tokens[c_start:c_end]
            cand_text = " ".join(cand_tokens_list)

            # Indices
            c_indices = self.processor.text_to_indices(cand_text, self.config.C_MAX_LEN)

            # Targets
            # Short Answer (Relative to candidate)
            # Default to (0, 0) which is usually [PAD] or ignored
            s_start_rel = 0
            s_end_rel = 0

            if long_label == 1 and short_start_token != -1:
                # Calculate relative position
                rel_s = short_start_token - c_start
                rel_e = short_end_token - c_start

                # Check if valid (within this candidate and within max len)
                # Note: config.C_MAX_LEN includes padding, but text_to_indices truncates.
                # We need to check against the truncated length.
                if (
                    0 <= rel_s < self.config.C_MAX_LEN
                    and 0 < rel_e <= self.config.C_MAX_LEN
                ):
                    s_start_rel = rel_s
                    s_end_rel = rel_e - 1  # Inclusive end index for classification
                else:
                    # Short answer exists but is cut off by truncation or logic error
                    s_start_rel = 0
                    s_end_rel = 0

            # Yes/No Target
            # Only applies if this is the positive candidate
            yn_target = (
                self.yn_map.get(yn_label_str, 2) if long_label == 1 else 2
            )  # 2=NONE

            samples.append(
                {
                    "example_id": ex_id,
                    "q_indices": q_indices,
                    "c_indices": c_indices,
                    "long_label": long_label,
                    "short_start": s_start_rel,
                    "short_end": s_end_rel,
                    "yn_label": yn_target,
                    # Metadata for inference reconstruction
                    "cand_start_token": c_start,
                    "cand_end_token": c_end,
                }
            )

        return samples

    @staticmethod
    def collate_fn(batch):
        """
        Collates a list of list of samples into a batch of tensors.
        """
        # Flatten the batch (list of lists -> list)
        flat_batch = [item for sublist in batch for item in sublist]

        if not flat_batch:
            return {}

        # Extract fields
        q_indices = [x["q_indices"] for x in flat_batch]
        c_indices = [x["c_indices"] for x in flat_batch]
        long_labels = [x["long_label"] for x in flat_batch]
        short_starts = [x["short_start"] for x in flat_batch]
        short_ends = [x["short_end"] for x in flat_batch]
        yn_labels = [x["yn_label"] for x in flat_batch]
        example_ids = [x["example_id"] for x in flat_batch]
        cand_starts = [x["cand_start_token"] for x in flat_batch]
        cand_ends = [x["cand_end_token"] for x in flat_batch]

        # Convert to tensors
        return {
            "q_input": torch.tensor(q_indices, dtype=torch.long),
            "c_input": torch.tensor(c_indices, dtype=torch.long),
            "long_labels": torch.tensor(
                long_labels, dtype=torch.float
            ),  # Float for BCE
            "short_starts": torch.tensor(short_starts, dtype=torch.long),
            "short_ends": torch.tensor(short_ends, dtype=torch.long),
            "yn_labels": torch.tensor(yn_labels, dtype=torch.long),
            "example_ids": example_ids,
            "cand_starts": cand_starts,
            "cand_ends": cand_ends,
        }
