import json
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.vocab import Vocabulary


class NQDataset(Dataset):
    def __init__(self, metadata_path, vocab, mode="train", debug_limit=None):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            vocab (Vocabulary): Loaded vocabulary object.
            mode (str): 'train', 'val', or 'test'.
            debug_limit (int, optional): Limit number of samples for debugging.
        """
        self.metadata = pd.read_csv(metadata_path)
        if debug_limit:
            self.metadata = self.metadata.head(debug_limit)

        self.vocab = vocab
        self.mode = mode

        # Create a set of relevant IDs for filtering raw data
        self.relevant_ids = set(self.metadata["example_id"].astype(str).values)

        # Determine which raw file to load based on metadata file path
        # Assuming metadata contains 'file_path' column pointing to source
        # We process unique file paths found in metadata
        self.data_map = {}
        unique_files = self.metadata["file_path"].unique()

        for rel_path in unique_files:
            full_path = os.path.join(Config.INPUT_DIR, rel_path)
            self._load_raw_data(full_path)

        # Yes/No Label Mapping
        self.yn_map = {"NONE": 0, "YES": 1, "NO": 2}

    def _load_raw_data(self, file_path):
        """Loads and pre-tokenizes relevant entries from JSONL into memory."""
        if not os.path.exists(file_path):
            print(f"Warning: Data file {file_path} not found.")
            return

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                eid = str(entry["example_id"])

                if eid in self.relevant_ids:
                    # Pre-split document text to save time in __getitem__
                    # We keep the document as a list of tokens
                    doc_text = entry.get("document_text", "")
                    entry["doc_tokens"] = doc_text.split()
                    self.data_map[eid] = entry

    def _encode_sequence(self, tokens, max_len):
        """Converts token list to indices and pads/truncates."""
        indices = [self.vocab.lookup_token(t) for t in tokens]

        # Truncate
        if len(indices) > max_len:
            indices = indices[:max_len]

        # Pad
        padding = [0] * (max_len - len(indices))  # 0 is PAD_TOKEN index
        return indices + padding

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        eid = str(row["example_id"])

        if eid not in self.data_map:
            # Fallback for missing data (should not happen if setup correctly)
            return self._empty_sample()

        entry = self.data_map[eid]
        doc_tokens = entry["doc_tokens"]
        question_tokens = entry["question_text"].split()
        candidates = entry["long_answer_candidates"]

        # Encode Question once
        q_indices = self._encode_sequence(question_tokens, Config.MAX_Q_LEN)

        samples = []

        # Determine which candidates to process
        target_candidates = []

        if self.mode in ["train"]:
            # Training: Negative Sampling
            pos_idx = (
                int(row["long_answer_index"]) if "long_answer_index" in row else -1
            )

            # Identify positive candidate
            pos_candidate = None
            if pos_idx != -1 and pos_idx < len(candidates):
                pos_candidate = candidates[pos_idx]
                pos_candidate["is_positive"] = True
                pos_candidate["global_idx"] = pos_idx
                target_candidates.append(pos_candidate)

            # Identify negative candidates
            neg_indices = [i for i in range(len(candidates)) if i != pos_idx]

            # Sample negatives
            num_neg = Config.NEGATIVE_SAMPLE_RATIO
            if len(neg_indices) > num_neg:
                selected_negs = random.sample(neg_indices, num_neg)
            else:
                selected_negs = neg_indices  # Take all if fewer than ratio

            for ni in selected_negs:
                cand = candidates[ni]
                cand["is_positive"] = False
                cand["global_idx"] = ni
                target_candidates.append(cand)

            # Shuffle to prevent model from learning position bias (e.g. pos always first)
            random.shuffle(target_candidates)

        else:
            # Validation / Test: Use ALL candidates
            # We limit to a reasonable number if too many to avoid OOM during inference,
            # though NQ usually has < 50 candidates per doc.
            for i, cand in enumerate(candidates):
                cand["is_positive"] = False  # Default
                # If validation, we can check ground truth
                if "long_answer_index" in row and int(row["long_answer_index"]) == i:
                    cand["is_positive"] = True
                cand["global_idx"] = i
                target_candidates.append(cand)

        # Process selected candidates
        for cand in target_candidates:
            start_token = cand["start_token"]
            end_token = cand["end_token"]

            # Extract candidate text from document
            # Note: end_token is exclusive in NQ
            cand_tokens = doc_tokens[start_token:end_token]
            c_indices = self._encode_sequence(cand_tokens, Config.MAX_C_LEN)

            # --- Labels ---

            # 1. Ranking Label
            rank_label = 1.0 if cand.get("is_positive", False) else 0.0

            # 2. Short Answer Span Labels
            # Default to (0, 0) which is <PAD>, effectively "no answer"
            s_start, s_end = 0, 0

            if rank_label == 1.0 and row.get("has_short_answer", False):
                # We need to find the short answer span in the raw annotations
                # Since we don't have raw annotations in metadata, we rely on the fact
                # that if has_short_answer is True, the raw data entry has it.
                # However, metadata row has flags. We need to look at raw entry['annotations']
                # But wait, metadata generation dropped annotations.
                # We must rely on the raw JSON loaded in data_map.

                raw_anns = entry.get("annotations", [])
                if raw_anns:
                    short_anns = raw_anns[0].get("short_answers", [])
                    if short_anns:
                        # Use the first short answer
                        sa = short_anns[0]
                        global_s_start = sa["start_token"]
                        global_s_end = sa["end_token"]

                        # Convert to local indices
                        local_start = global_s_start - start_token
                        local_end = global_s_end - start_token

                        # Check if valid within this candidate and within truncation limit
                        # Note: local_end is exclusive.
                        # If local_end is 5, the span is indices 0,1,2,3,4.
                        # Config.MAX_C_LEN is the limit.
                        if (
                            local_start >= 0
                            and local_end <= len(cand_tokens)
                            and local_end <= Config.MAX_C_LEN
                        ):
                            s_start = local_start
                            s_end = (
                                local_end - 1
                            )  # Convert to inclusive index for CrossEntropy
                        else:
                            # Short answer exists but was truncated or outside this candidate
                            # (though NQ guarantees short is inside long, truncation might cut it)
                            s_start, s_end = 0, 0

            # 3. Yes/No Label
            yn_label = 0  # NONE
            if rank_label == 1.0:
                yn_str = row.get("yes_no_answer", "NONE")
                yn_label = self.yn_map.get(yn_str, 0)

            samples.append(
                {
                    "q_ids": torch.tensor(q_indices, dtype=torch.long),
                    "c_ids": torch.tensor(c_indices, dtype=torch.long),
                    "rank_label": torch.tensor(rank_label, dtype=torch.float),
                    "span_start": torch.tensor(s_start, dtype=torch.long),
                    "span_end": torch.tensor(s_end, dtype=torch.long),
                    "yn_label": torch.tensor(yn_label, dtype=torch.long),
                    "global_cand_idx": cand["global_idx"],
                    "example_id": eid,
                }
            )

        return samples

    def _empty_sample(self):
        """Returns a dummy sample to handle edge cases."""
        return []


def collate_fn(batch):
    """
    Flattens the list of lists returned by Dataset.__getitem__ and batches tensors.

    Args:
        batch: List of lists of sample dictionaries.

    Returns:
        dict: Batched tensors.
    """
    # Flatten the batch (list of lists -> list of dicts)
    flat_batch = [item for sublist in batch for item in sublist]

    if not flat_batch:
        return {}

    # Stack tensors
    q_ids = torch.stack([x["q_ids"] for x in flat_batch])
    c_ids = torch.stack([x["c_ids"] for x in flat_batch])
    rank_labels = torch.stack([x["rank_label"] for x in flat_batch])
    span_starts = torch.stack([x["span_start"] for x in flat_batch])
    span_ends = torch.stack([x["span_end"] for x in flat_batch])
    yn_labels = torch.stack([x["yn_label"] for x in flat_batch])

    # Metadata lists (not tensors)
    global_cand_idxs = [x["global_cand_idx"] for x in flat_batch]
    example_ids = [x["example_id"] for x in flat_batch]

    return {
        "q_input_ids": q_ids,
        "c_input_ids": c_ids,
        "rank_labels": rank_labels,
        "span_start_labels": span_starts,
        "span_end_labels": span_ends,
        "yn_labels": yn_labels,
        "global_cand_idxs": global_cand_idxs,
        "example_ids": example_ids,
    }
