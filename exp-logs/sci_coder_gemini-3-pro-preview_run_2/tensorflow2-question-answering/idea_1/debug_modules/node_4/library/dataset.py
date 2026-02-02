import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config
from library.data_utils import Tokenizer, extract_candidate_text


def get_file_offsets(data_path, load_cached_data=True):
    """
    Creates or loads a mapping from example_id to byte offset in the JSONL file.
    Follows the strict caching logic requirements.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Construct cache filename based on data filename
    base_name = os.path.basename(data_path)
    cache_name = f"{base_name}_offsets.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_name)

    # 1. Try to load
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading file offsets from {cache_path}...")
            df = pd.read_parquet(cache_path)
            # Convert to dictionary for O(1) lookup
            return dict(zip(df["example_id"], df["offset"]))
        except Exception as e:
            print(f"Failed to load offsets: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Computing file offsets for {data_path}...")
    offsets = {}

    try:
        with open(data_path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break

                try:
                    entry = json.loads(line)
                    ex_id = entry["example_id"]
                    # Ensure ID is string
                    offsets[str(ex_id)] = offset
                except json.JSONDecodeError:
                    continue

    except FileNotFoundError:
        print(f"Error: Data file {data_path} not found.")
        return {}

    # 3. Save to cache
    print(f"Saving offsets to {cache_path}...")
    df = pd.DataFrame(list(offsets.items()), columns=["example_id", "offset"])
    df.to_parquet(cache_path, index=False)

    return offsets


class NQTrainDataset(Dataset):
    def __init__(
        self, metadata_path, data_path, tokenizer, limit=None, load_cached_data=True
    ):
        """
        Dataset for training the Neural BoE Ranker.

        Args:
            metadata_path: Path to the metadata CSV (train or val).
            data_path: Path to the source JSONL file.
            tokenizer: Fitted Tokenizer instance.
            limit: Optional limit on number of samples (for debugging).
            load_cached_data: Whether to use cached file offsets.
        """
        self.data_path = data_path
        self.tokenizer = tokenizer

        # Load metadata
        print(f"Loading metadata from {metadata_path}...")
        self.meta = pd.read_csv(metadata_path)

        # Filter: For training/validation of the ranker, we focus on examples
        # that actually HAVE a long answer to learn the positive signal.
        # long_answer_index != -1
        initial_len = len(self.meta)
        self.meta = self.meta[self.meta["long_answer_index"] != -1].reset_index(
            drop=True
        )
        print(
            f"Filtered dataset from {initial_len} to {len(self.meta)} samples (kept only those with long answers)."
        )

        if limit:
            self.meta = self.meta.iloc[:limit]
            print(f"Limited dataset to {limit} samples.")

        # Load offsets for random access
        self.offsets = get_file_offsets(data_path, load_cached_data=load_cached_data)

        # Pre-convert example_ids in meta to string to match offsets keys
        self.meta["example_id"] = self.meta["example_id"].astype(str)

        # Filter out metadata entries that don't have offsets (mismatched files)
        valid_ids = set(self.offsets.keys())
        self.meta = self.meta[self.meta["example_id"].isin(valid_ids)].reset_index(
            drop=True
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        ex_id = row["example_id"]
        label_idx = int(row["long_answer_index"])

        offset = self.offsets[ex_id]

        with open(self.data_path, "rb") as f:
            f.seek(offset)
            line = f.readline()
            entry = json.loads(line)

        # Extract Question
        q_text = entry.get("question_text", "")
        q_tokens = q_text.split()
        q_seq = self.tokenizer.text_to_sequence(q_tokens, max_len=Config.MAX_SEQ_LEN)

        # Extract Document Tokens
        doc_text = entry.get("document_text", "")
        doc_tokens = doc_text.split()

        candidates = entry.get("long_answer_candidates", [])

        samples = []

        # 1. Positive Sample
        if 0 <= label_idx < len(candidates):
            cand = candidates[label_idx]
            c_text = extract_candidate_text(
                doc_tokens, cand["start_token"], cand["end_token"]
            )
            c_seq = self.tokenizer.text_to_sequence(
                c_text.split(), max_len=Config.MAX_SEQ_LEN
            )
            samples.append({"q_seq": q_seq, "c_seq": c_seq, "label": 1.0})

        # 2. Negative Sampling
        # Get indices of all candidates except the correct one
        neg_indices = [i for i in range(len(candidates)) if i != label_idx]

        # Randomly sample negatives
        if len(neg_indices) > 0:
            # If we have fewer negatives than requested, take them all (maybe repeat)
            # If we have more, sample without replacement
            num_negs = Config.NEGATIVE_SAMPLES_RATIO
            if len(neg_indices) >= num_negs:
                selected_negs = np.random.choice(neg_indices, num_negs, replace=False)
            else:
                # Sample with replacement if not enough candidates
                selected_negs = np.random.choice(neg_indices, num_negs, replace=True)

            for neg_idx in selected_negs:
                cand = candidates[neg_idx]
                c_text = extract_candidate_text(
                    doc_tokens, cand["start_token"], cand["end_token"]
                )
                c_seq = self.tokenizer.text_to_sequence(
                    c_text.split(), max_len=Config.MAX_SEQ_LEN
                )
                samples.append({"q_seq": q_seq, "c_seq": c_seq, "label": 0.0})

        # If for some reason no samples were generated (e.g. invalid label index), return empty list
        # The collate_fn will handle this.
        return samples


class NQInferenceDataset(Dataset):
    def __init__(
        self, metadata_path, data_path, tokenizer, limit=None, load_cached_data=True
    ):
        """
        Dataset for inference. Returns all candidates for ranking.
        """
        self.data_path = data_path
        self.tokenizer = tokenizer

        print(f"Loading inference metadata from {metadata_path}...")
        self.meta = pd.read_csv(metadata_path)

        if limit:
            self.meta = self.meta.iloc[:limit]
            print(f"Limited inference dataset to {limit} samples.")

        self.offsets = get_file_offsets(data_path, load_cached_data=load_cached_data)
        self.meta["example_id"] = self.meta["example_id"].astype(str)

        # Filter
        valid_ids = set(self.offsets.keys())
        self.meta = self.meta[self.meta["example_id"].isin(valid_ids)].reset_index(
            drop=True
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        ex_id = row["example_id"]
        offset = self.offsets[ex_id]

        with open(self.data_path, "rb") as f:
            f.seek(offset)
            line = f.readline()
            entry = json.loads(line)

        q_text = entry.get("question_text", "")
        q_tokens = q_text.split()
        q_seq = self.tokenizer.text_to_sequence(q_tokens, max_len=Config.MAX_SEQ_LEN)

        doc_text = entry.get("document_text", "")
        doc_tokens = doc_text.split()

        candidates = entry.get("long_answer_candidates", [])

        candidate_data = []
        for i, cand in enumerate(candidates):
            c_text = extract_candidate_text(
                doc_tokens, cand["start_token"], cand["end_token"]
            )
            c_seq = self.tokenizer.text_to_sequence(
                c_text.split(), max_len=Config.MAX_SEQ_LEN
            )

            # Store token indices for submission formatting
            token_indices = f"{cand['start_token']}:{cand['end_token']}"

            candidate_data.append(
                {
                    "c_seq": c_seq,
                    "token_indices": token_indices,
                    "raw_text": c_text,  # Useful for short answer extraction later
                }
            )

        return {
            "example_id": ex_id,
            "q_seq": q_seq,
            "q_text": q_text,  # Passed for short answer extraction
            "candidates": candidate_data,
        }


def collate_fn(batch):
    """
    Custom collate function to handle the list of samples returned by NQTrainDataset
    and the complex structure of NQInferenceDataset.
    """
    # Check if this is a training batch (list of lists of dicts)
    if isinstance(batch[0], list):
        # Flatten the batch
        flat_samples = [sample for item in batch for sample in item]

        if not flat_samples:
            return None

        q_seqs = torch.tensor([s["q_seq"] for s in flat_samples], dtype=torch.long)
        c_seqs = torch.tensor([s["c_seq"] for s in flat_samples], dtype=torch.long)
        labels = torch.tensor([s["label"] for s in flat_samples], dtype=torch.float)

        return {"q_seqs": q_seqs, "c_seqs": c_seqs, "labels": labels}

    # Check if this is an inference batch (list of dicts with 'candidates' list)
    elif "candidates" in batch[0]:
        # For inference, we usually process one example at a time or handle variable candidates.
        # To keep it simple for the model, we can just return the list of dicts
        # and let the inference loop handle the internal batching of candidates.
        return batch

    return None
