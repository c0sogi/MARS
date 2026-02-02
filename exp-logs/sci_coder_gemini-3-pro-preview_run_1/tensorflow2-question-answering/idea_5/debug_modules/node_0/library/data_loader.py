import os
import json
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.text_processing import text_to_indices


def process_split(split, load_cached_data=True):
    """
    Processes the raw data into a flattened format suitable for training/inference.
    Handles caching and negative subsampling for the training set.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        pd.DataFrame: Processed dataframe containing flattened examples.
    """

    # Determine paths based on split
    if split == "train":
        meta_path = Config.TRAIN_META_PATH
        data_path = Config.TRAIN_DATA_PATH
        cache_path = Config.PROCESSED_TRAIN_PATH
        is_train = True
    elif split == "val":
        meta_path = Config.VAL_META_PATH
        data_path = Config.TRAIN_DATA_PATH  # Val comes from train file
        cache_path = Config.PROCESSED_VAL_PATH
        is_train = False
    elif split == "test":
        meta_path = Config.TEST_META_PATH
        data_path = Config.TEST_DATA_PATH
        cache_path = Config.PROCESSED_TEST_PATH
        is_train = False
    else:
        raise ValueError(f"Invalid split: {split}")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"[DataLoader] Loading processed {split} data from {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"[DataLoader] Failed to load cache: {e}. Re-processing...")

    # 2. Process from scratch
    print(f"[DataLoader] Processing {split} data from scratch...")

    if not os.path.exists(meta_path):
        print(f"[DataLoader] Metadata {meta_path} missing. Returning empty DataFrame.")
        return pd.DataFrame()

    meta_df = pd.read_parquet(meta_path)

    # Debugging subsample
    if Config.DEBUG:
        meta_df = meta_df.head(Config.DEBUG_SAMPLE_SIZE)
        print(f"[DataLoader] DEBUG: Subsampled metadata to {len(meta_df)} rows.")

    flattened_data = []

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, "rb") as f:
        for _, row in meta_df.iterrows():
            offset = row["byte_offset"]
            f.seek(offset)
            line = f.readline()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            example_id = entry["example_id"]
            question_text = entry["question_text"]
            doc_text = entry["document_text"]
            doc_tokens = doc_text.split()  # Basic whitespace split to match indices
            candidates = entry["long_answer_candidates"]

            # Parse annotations for ground truth (Train/Val only)
            correct_long_ranges = []
            correct_short_ranges = []

            if is_train or split == "val":
                # Use raw entry annotations
                anns = entry.get("annotations", [])
                for ann in anns:
                    # Long answer
                    la = ann.get("long_answer", {})
                    if la.get("start_token", -1) != -1:
                        correct_long_ranges.append((la["start_token"], la["end_token"]))

                    # Short answers
                    sas = ann.get("short_answers", [])
                    for sa in sas:
                        correct_short_ranges.append(
                            (sa["start_token"], sa["end_token"])
                        )

            # Process candidates
            for idx, cand in enumerate(candidates):
                c_start = cand["start_token"]
                c_end = cand["end_token"]

                # Extract text
                cand_text = " ".join(doc_tokens[c_start:c_end])

                # Determine Labels
                is_long_answer = 0
                short_start = -1
                short_end = -1

                if is_train or split == "val":
                    # Check Long Answer Match
                    for gt_start, gt_end in correct_long_ranges:
                        if c_start == gt_start and c_end == gt_end:
                            is_long_answer = 1
                            break

                    # Check Short Answer Match (only if it's a long answer)
                    if is_long_answer:
                        for s_start, s_end in correct_short_ranges:
                            # Short answer must be contained within candidate
                            if s_start >= c_start and s_end <= c_end:
                                # Calculate relative indices
                                short_start = s_start - c_start
                                short_end = s_end - c_start
                                break

                flattened_data.append(
                    {
                        "example_id": example_id,
                        "candidate_index": idx,
                        "question_text": question_text,
                        "candidate_text": cand_text,
                        "long_label": is_long_answer,
                        "short_start": short_start,
                        "short_end": short_end,
                    }
                )

    df = pd.DataFrame(flattened_data)

    # Negative Subsampling for Training
    if is_train and not df.empty:
        positives = df[df["long_label"] == 1]
        negatives = df[df["long_label"] == 0]

        if not negatives.empty:
            # Sample a percentage of negatives
            n_sample = int(len(negatives) * Config.NEGATIVE_SAMPLING_RATIO)
            # Ensure a minimum number if dataset is very small
            n_sample = max(n_sample, min(len(negatives), 10))

            negatives_sampled = negatives.sample(n=n_sample, random_state=Config.SEED)
            df = (
                pd.concat([positives, negatives_sampled])
                .sample(frac=1, random_state=Config.SEED)
                .reset_index(drop=True)
            )
            print(
                f"[DataLoader] Subsampling applied. Train size: {len(df)} (Pos: {len(positives)}, Neg: {len(negatives_sampled)})"
            )

    # Cache results
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if not df.empty:
        df.to_parquet(cache_path, index=False)
        print(f"[DataLoader] Saved processed {split} data to {cache_path}")
    else:
        print(f"[DataLoader] Warning: Processed dataframe for {split} is empty.")

    return df


class NQDataset(Dataset):
    """
    PyTorch Dataset for Natural Questions.
    Reads from the processed DataFrame and converts text to indices.
    """

    def __init__(self, dataframe, vocab):
        self.data = dataframe
        self.vocab = vocab
        self.max_q_len = Config.MAX_Q_LEN
        self.max_cand_len = Config.MAX_SEQ_LEN

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Tokenize and encode using library function
        q_indices = text_to_indices(row["question_text"], self.vocab, self.max_q_len)
        c_indices = text_to_indices(
            row["candidate_text"], self.vocab, self.max_cand_len
        )

        # Labels
        long_label = float(row["long_label"]) if "long_label" in row else 0.0

        # Handle Short Answer Labels
        # -1 indicates no short answer or ignore
        s_start = int(row["short_start"]) if "short_start" in row else -1
        s_end = int(row["short_end"]) if "short_end" in row else -1

        # Adjust for truncation if the answer falls outside max_cand_len
        if s_start >= self.max_cand_len:
            s_start = -1
            s_end = -1
        elif s_end >= self.max_cand_len:
            s_end = self.max_cand_len - 1  # Clamp end index

        return {
            "example_id": row["example_id"],
            "candidate_index": row["candidate_index"],
            "question_ids": torch.tensor(q_indices, dtype=torch.long),
            "candidate_ids": torch.tensor(c_indices, dtype=torch.long),
            "long_label": torch.tensor(long_label, dtype=torch.float),
            "short_start_label": torch.tensor(s_start, dtype=torch.long),
            "short_end_label": torch.tensor(s_end, dtype=torch.long),
        }


def collate_fn(batch):
    """
    Collate function to batch data.
    Stacks tensors since padding is handled in dataset via text_to_indices.
    """
    question_ids = torch.stack([item["question_ids"] for item in batch])
    candidate_ids = torch.stack([item["candidate_ids"] for item in batch])
    long_labels = torch.stack([item["long_label"] for item in batch])
    short_start_labels = torch.stack([item["short_start_label"] for item in batch])
    short_end_labels = torch.stack([item["short_end_label"] for item in batch])

    # Metadata lists (not tensors)
    example_ids = [item["example_id"] for item in batch]
    candidate_indices = [item["candidate_index"] for item in batch]

    return {
        "question_ids": question_ids,
        "candidate_ids": candidate_ids,
        "long_labels": long_labels,
        "short_start_labels": short_start_labels,
        "short_end_labels": short_end_labels,
        "example_ids": example_ids,
        "candidate_indices": candidate_indices,
    }
