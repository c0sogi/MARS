import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from library.config import Config
from library.text_utils import TextUtils


class DataFactory:
    """
    Manages data loading, caching, and batch creation for the NQ dataset.
    """

    @staticmethod
    def build_file_index(
        jsonl_path: str, load_cached_data: bool = True
    ) -> Dict[str, int]:
        """
        Builds or loads a mapping from example_id to file byte offset.
        Strictly follows the caching logic required.
        """
        # Construct cache path based on input filename
        filename = os.path.basename(jsonl_path)
        cache_filename = f"{filename}.index.parquet"
        cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        # Ensure directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # 1. Try to load
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading file index from {cache_path}...")
            try:
                df = pd.read_parquet(cache_path)
                # Convert to dict
                return dict(zip(df["example_id"], df["offset"]))
            except Exception as e:
                print(f"Failed to load index cache: {e}. Rebuilding...")

        # 2. Compute from scratch
        print(f"Building file index for {jsonl_path}...")
        offsets = {}

        with open(jsonl_path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break

                # Parse just enough to get the ID
                try:
                    entry = json.loads(line.decode("utf-8"))
                    ex_id = str(entry["example_id"])
                    offsets[ex_id] = offset
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"Error parsing line at offset {offset}: {e}")
                    continue

        if len(offsets) == 0:
            raise RuntimeError(
                f"Failed to build file index for {jsonl_path}. No valid entries found."
            )

        # 3. Save to cache
        print(f"Saving file index to {cache_path}...")
        df = pd.DataFrame(list(offsets.items()), columns=["example_id", "offset"])
        df.to_parquet(cache_path, index=False)

        return offsets


class NQDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        jsonl_path: str,
        vocab: Dict[str, int],
        is_train: bool = True,
        debug: bool = False,
        debug_size: int = 1000,
        load_cached_index: bool = True,
    ):
        """
        Args:
            metadata_path: Path to the metadata CSV.
            jsonl_path: Path to the source JSONL file.
            vocab: Vocabulary mapping.
            is_train: Whether this is for training (enables negative sampling).
            debug: If True, limit dataset size.
            debug_size: Number of samples if debug is True.
            load_cached_index: Whether to use cached file offsets.
        """
        self.is_train = is_train
        self.vocab = vocab
        self.max_q_len = Config.MAX_Q_LEN
        self.max_c_len = Config.MAX_C_LEN
        self.jsonl_path = jsonl_path

        # Load metadata
        print(f"Loading metadata from {metadata_path}...")
        self.meta_df = pd.read_csv(metadata_path)

        # Handle string type for IDs
        self.meta_df["example_id"] = self.meta_df["example_id"].astype(str)

        if debug:
            print(f"Debug mode: sampling {debug_size} examples.")
            self.meta_df = self.meta_df.head(debug_size)

        # Build/Load File Index
        self.file_index = DataFactory.build_file_index(
            jsonl_path, load_cached_data=load_cached_index
        )

        # Filter metadata to ensure we only have IDs present in the file (safety check)
        initial_len = len(self.meta_df)
        self.meta_df = self.meta_df[self.meta_df["example_id"].isin(self.file_index)]
        if len(self.meta_df) < initial_len:
            print(
                f"Filtered {initial_len - len(self.meta_df)} metadata entries not found in JSONL index."
            )

        if len(self.meta_df) == 0:
            raise ValueError(
                "Dataset is empty after filtering! Check if file index matches metadata IDs."
            )

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        ex_id = row["example_id"]

        # Retrieve raw data using offset
        offset = self.file_index[ex_id]

        with open(self.jsonl_path, "rb") as f:
            f.seek(offset)
            line = f.readline()
            entry = json.loads(line.decode("utf-8"))

        # Process Question
        question_text = entry["question_text"]
        q_indices = TextUtils.text_to_indices(question_text, self.vocab, self.max_q_len)

        # Process Candidates
        candidates_data = entry["long_answer_candidates"]
        doc_text = entry["document_text"]
        tokens = doc_text.split()  # Document text is space-separated tokens

        # Helper to get text from token indices
        def get_cand_text(start, end):
            # Basic safety clipping
            start = max(0, start)
            end = min(len(tokens), end)
            return " ".join(tokens[start:end])

        selected_candidates = []
        labels = []
        candidate_indices = []  # To track original indices in the candidates list

        if self.is_train:
            # Training Logic: Negative Sampling
            correct_idx = (
                int(row["long_answer_index"]) if "long_answer_index" in row else -1
            )

            # Prepare pool of indices
            num_candidates = len(candidates_data)
            all_indices = list(range(num_candidates))

            # Identify Positive
            if correct_idx != -1 and correct_idx < num_candidates:
                # Add Positive
                c_info = candidates_data[correct_idx]
                c_text = get_cand_text(c_info["start_token"], c_info["end_token"])
                selected_candidates.append(c_text)
                labels.append(1.0)
                candidate_indices.append(correct_idx)

                # Remove from pool for negative sampling
                if correct_idx in all_indices:
                    all_indices.remove(correct_idx)

                # Sample Negatives
                n_negs = Config.NEG_SAMPLES
            else:
                # No positive answer (or invalid index)
                # We will sample (NEG_SAMPLES + 1) negatives to keep batch size consistent
                n_negs = Config.NEG_SAMPLES + 1

            # Sample Negatives
            if len(all_indices) > 0:
                if len(all_indices) >= n_negs:
                    neg_indices = np.random.choice(all_indices, n_negs, replace=False)
                else:
                    neg_indices = np.random.choice(all_indices, n_negs, replace=True)

                for ni in neg_indices:
                    c_info = candidates_data[ni]
                    c_text = get_cand_text(c_info["start_token"], c_info["end_token"])
                    selected_candidates.append(c_text)
                    labels.append(0.0)
                    candidate_indices.append(ni)
            else:
                # Edge case: No candidates at all? Or only positive?
                # Fill with placeholders if necessary
                while len(selected_candidates) < (Config.NEG_SAMPLES + 1):
                    selected_candidates.append("")
                    labels.append(0.0)
                    candidate_indices.append(-1)

        else:
            # Inference Logic: Return all candidates
            for i, c_info in enumerate(candidates_data):
                c_text = get_cand_text(c_info["start_token"], c_info["end_token"])
                selected_candidates.append(c_text)
                labels.append(0.0)
                candidate_indices.append(i)

            # If no candidates, add a dummy
            if not selected_candidates:
                selected_candidates.append("")
                labels.append(0.0)
                candidate_indices.append(-1)

        # Tokenize Candidates
        c_indices_list = [
            TextUtils.text_to_indices(ct, self.vocab, self.max_c_len)
            for ct in selected_candidates
        ]

        return {
            "example_id": ex_id,
            "q_indices": torch.tensor(q_indices, dtype=torch.long),
            "c_indices": torch.tensor(c_indices_list, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
            "candidate_indices": torch.tensor(candidate_indices, dtype=torch.long),
            "raw_candidates": selected_candidates,
        }

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable number of candidates in inference mode,
        or flatten the batch in training mode.
        """
        # Check if we are in training mode (fixed size) or inference (variable size)
        sizes = [item["c_indices"].shape[0] for item in batch]
        is_fixed_size = all(s == sizes[0] for s in sizes)

        example_ids = [item["example_id"] for item in batch]
        q_indices = torch.stack([item["q_indices"] for item in batch])

        if is_fixed_size:
            # Training mode: Stack typically [Batch, Num_Cand, Seq_Len]
            c_indices = torch.stack([item["c_indices"] for item in batch])
            labels = torch.stack([item["labels"] for item in batch])
            candidate_ids = torch.stack([item["candidate_indices"] for item in batch])
            raw_candidates = [item["raw_candidates"] for item in batch]
        else:
            # Inference mode: Pad to max candidates in batch
            max_cands = max(sizes)
            batch_size = len(batch)
            seq_len = batch[0]["c_indices"].shape[1]

            padded_c_indices = torch.zeros(
                (batch_size, max_cands, seq_len), dtype=torch.long
            )
            padded_labels = torch.zeros((batch_size, max_cands), dtype=torch.float)
            padded_cand_ids = torch.full((batch_size, max_cands), -1, dtype=torch.long)

            # Mask to indicate valid candidates
            mask = torch.zeros((batch_size, max_cands), dtype=torch.bool)

            raw_candidates = []

            for i, item in enumerate(batch):
                n = item["c_indices"].shape[0]
                padded_c_indices[i, :n, :] = item["c_indices"]
                padded_labels[i, :n] = item["labels"]
                padded_cand_ids[i, :n] = item["candidate_indices"]
                mask[i, :n] = True
                raw_candidates.append(item["raw_candidates"])

            c_indices = padded_c_indices
            labels = padded_labels
            candidate_ids = padded_cand_ids

            return {
                "example_ids": example_ids,
                "q_indices": q_indices,
                "c_indices": c_indices,
                "labels": labels,
                "candidate_indices": candidate_ids,
                "mask": mask,
                "raw_candidates": raw_candidates,
            }

        return {
            "example_ids": example_ids,
            "q_indices": q_indices,
            "c_indices": c_indices,
            "labels": labels,
            "candidate_indices": candidate_ids,
            "raw_candidates": raw_candidates,
        }


def get_dataloaders(
    vocab: Dict[str, int], debug: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for train, validation, and test sets.
    """

    # Train Set
    train_dataset = NQDataset(
        metadata_path=Config.TRAIN_META_PATH,
        jsonl_path=Config.TRAIN_DATA_PATH,
        vocab=vocab,
        is_train=True,
        debug=debug,
        debug_size=Config.DEBUG_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=NQDataset.collate_fn,
        pin_memory=True,
    )

    # Validation Set
    val_dataset = NQDataset(
        metadata_path=Config.VAL_META_PATH,
        jsonl_path=Config.TRAIN_DATA_PATH,  # Validation comes from train file
        vocab=vocab,
        is_train=False,  # Inference mode for accurate metrics
        debug=debug,
        debug_size=Config.DEBUG_SIZE,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=NQDataset.collate_fn,
        pin_memory=True,
    )

    # Test Set
    test_dataset = NQDataset(
        metadata_path=Config.TEST_META_PATH,
        jsonl_path=Config.TEST_DATA_PATH,
        vocab=vocab,
        is_train=False,
        debug=debug,
        debug_size=Config.DEBUG_SIZE,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=NQDataset.collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
