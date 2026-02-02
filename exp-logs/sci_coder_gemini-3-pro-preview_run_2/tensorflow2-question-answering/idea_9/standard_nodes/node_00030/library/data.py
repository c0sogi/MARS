import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Tuple, Optional, Union

from library.config import Config
from library.utils import build_vocab, tokenize

# Constants for Yes/No mapping
YES_NO_MAP = {"NONE": 0, "YES": 1, "NO": 2}


def preprocess_data(
    metadata_path: str,
    raw_data_path: str,
    output_path: str,
    vocab: Dict[str, int],
    is_train: bool = True,
    load_cached_data: bool = True,
    sample_fraction: float = 1.0,
) -> pd.DataFrame:
    """
    Reads raw JSONL data, tokenizes text, maps to indices, and saves/loads from Parquet.
    Strict caching logic implementation.
    """
    # 1. Try to load from cache
    if load_cached_data and os.path.exists(output_path):
        try:
            print(f"Loading processed data from {output_path}...")
            df = pd.read_parquet(output_path)

            # Cite debug_lesson_9: Enforce type consistency immediately after loading cached data.
            # Parquet deserialization often produces NumPy arrays for sequence columns, which causes
            # broadcasting errors during list concatenation in the Collator.
            list_cols = ["question_ids", "document_ids", "candidates", "short_answers"]
            for col in list_cols:
                if col in df.columns:
                    # Ensure column contains Python lists, not NumPy arrays
                    df[col] = df[col].apply(
                        lambda x: x.tolist() if isinstance(x, np.ndarray) else list(x)
                    )

            print(f"Loaded {len(df)} records.")
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing data from {raw_data_path}...")

    # Load metadata to filter and get labels
    meta_df = pd.read_csv(metadata_path)

    # Handle debugging/subsampling
    if sample_fraction < 1.0:
        meta_df = meta_df.sample(
            frac=sample_fraction, random_state=Config.SEED
        ).reset_index(drop=True)

    valid_ids = set(meta_df["example_id"].astype(str))

    # Create a lookup for labels if training
    meta_lookup = {}
    if is_train:
        for _, row in meta_df.iterrows():
            eid = str(row["example_id"])
            meta_lookup[eid] = {
                "long_answer_index": row["long_answer_index"],
                "yes_no_answer": row["yes_no_answer"],
            }

    processed_records = []

    # Pre-fetch special token IDs
    unk_id = vocab.get(Config.UNK_TOKEN, 1)

    with open(raw_data_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            eid = str(entry["example_id"])

            if eid not in valid_ids:
                continue

            # Tokenize Question
            q_tokens = tokenize(entry["question_text"])
            q_ids = [vocab.get(t, unk_id) for t in q_tokens]

            # Tokenize Document
            # NQ document_text is space-separated tokens, so split() works perfectly
            doc_text = entry["document_text"]
            doc_tokens = doc_text.split()
            doc_ids = [vocab.get(t, unk_id) for t in doc_tokens]

            # Process Candidates
            # Store as list of tuples: (start_token, end_token, top_level)
            candidates = []
            for cand in entry["long_answer_candidates"]:
                candidates.append(
                    (cand["start_token"], cand["end_token"], int(cand["top_level"]))
                )

            record = {
                "example_id": eid,
                "question_ids": q_ids,
                "document_ids": doc_ids,
                "candidates": candidates,
            }

            # Add labels for training data
            if is_train:
                labels = meta_lookup[eid]
                record["long_answer_index"] = labels["long_answer_index"]
                record["yes_no_answer"] = YES_NO_MAP.get(labels["yes_no_answer"], 0)

                # Extract short answer annotations from raw JSON to get token offsets
                # The metadata CSV only has boolean has_short_answer, we need coordinates
                short_answers = []
                for ann in entry["annotations"]:
                    for sa in ann.get("short_answers", []):
                        short_answers.append((sa["start_token"], sa["end_token"]))
                record["short_answers"] = short_answers

            processed_records.append(record)

    df = pd.DataFrame(processed_records)

    # 3. Save to cache
    print(f"Saving processed data to {output_path}...")
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)

    return df


class NQDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, is_train: bool = True):
        self.data = data_df
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        item = {
            "example_id": row["example_id"],
            "question_ids": row["question_ids"],
            "document_ids": row["document_ids"],
            # Candidates are stored as numpy array of objects or lists in dataframe
            # Need to ensure they are python lists of tuples/lists
            "candidates": row["candidates"],
        }

        if self.is_train:
            item["long_answer_index"] = row["long_answer_index"]
            item["yes_no_answer"] = row["yes_no_answer"]
            item["short_answers"] = row["short_answers"]

        return item


class NQCollator:
    def __init__(self, vocab: Dict[str, int], is_train: bool = True):
        self.pad_id = vocab[Config.PAD_TOKEN]
        self.sep_id = vocab[Config.SEP_TOKEN]
        self.is_train = is_train
        self.max_len = Config.MAX_SEQ_LEN
        self.neg_ratio = Config.NEGATIVE_SAMPLING_RATIO

    def __call__(self, batch: List[Dict]):
        # Outputs
        batch_input_ids = []
        batch_ranking_labels = []
        batch_start_labels = []
        batch_end_labels = []
        batch_yes_no_labels = []

        # Metadata for inference mapping
        batch_example_ids = []
        batch_candidate_indices = []

        for item in batch:
            q_ids = item["question_ids"]
            doc_ids = item["document_ids"]
            candidates = item["candidates"]  # List of (start, end, top_level)

            # Determine which candidates to process
            selected_indices = []
            labels_map = {}  # cand_idx -> (rank_label, start_span, end_span, yes_no)

            if self.is_train:
                correct_idx = item["long_answer_index"]

                # 1. Positive Sample
                if correct_idx != -1 and correct_idx < len(candidates):
                    selected_indices.append(correct_idx)

                    # Determine short answer targets relative to candidate
                    # Default: 0 (no answer)
                    s_start_rel, s_end_rel = 0, 0

                    # If there are short answers, find one that fits in this candidate
                    c_start, c_end, _ = candidates[correct_idx]

                    # NQ annotations list multiple valid short answers, pick first valid one
                    # that is contained within the long answer candidate
                    for sa_start, sa_end in item["short_answers"]:
                        if sa_start >= c_start and sa_end <= c_end:
                            # Calculate relative offset
                            # Input: [Q] [SEP] [Cand]
                            # Offset = len(Q) + 1
                            # Relative in Cand = sa_start - c_start
                            offset = len(q_ids) + 1
                            s_start_rel = offset + (sa_start - c_start)
                            s_end_rel = (
                                offset + (sa_end - c_start) - 1
                            )  # Inclusive end index

                            # Check if truncated
                            if s_end_rel >= self.max_len:
                                s_start_rel, s_end_rel = 0, 0
                            else:
                                break  # Found a valid one

                    labels_map[correct_idx] = (
                        1.0,
                        s_start_rel,
                        s_end_rel,
                        item["yes_no_answer"],
                    )

                # 2. Negative Sampling
                # Filter candidates to top_level only for negatives usually, but NQ provides all.
                # We simply exclude the correct index.
                all_indices = list(range(len(candidates)))
                if correct_idx != -1:
                    if correct_idx in all_indices:
                        all_indices.remove(correct_idx)

                # How many negatives?
                num_pos = 1 if correct_idx != -1 else 0
                num_neg = int(max(1, num_pos * self.neg_ratio))

                if len(all_indices) > 0:
                    neg_indices = random.sample(
                        all_indices, min(len(all_indices), num_neg)
                    )
                    selected_indices.extend(neg_indices)
                    for ni in neg_indices:
                        # Negative labels: Rank=0, Span=0, YesNo=0 (NONE)
                        labels_map[ni] = (0.0, 0, 0, 0)

            else:
                # Inference: Process all candidates (or filter top_level if desired)
                # Processing all candidates ensures we don't miss anything
                selected_indices = list(range(len(candidates)))

            # Construct Sequences
            for cand_idx in selected_indices:
                c_start, c_end, _ = candidates[cand_idx]

                # Extract candidate text
                # Handle edge cases where indices might be out of bounds (though unlikely in clean data)
                c_start = max(0, min(c_start, len(doc_ids)))
                c_end = max(0, min(c_end, len(doc_ids)))
                cand_tokens = doc_ids[c_start:c_end]

                # Concatenate: [Q] + [SEP] + [Cand]
                input_seq = q_ids + [self.sep_id] + cand_tokens

                # Truncate
                if len(input_seq) > self.max_len:
                    input_seq = input_seq[: self.max_len]

                batch_input_ids.append(torch.tensor(input_seq, dtype=torch.long))

                if self.is_train:
                    rank, s_start, s_end, yn = labels_map[cand_idx]
                    batch_ranking_labels.append(rank)
                    batch_start_labels.append(s_start)
                    batch_end_labels.append(s_end)
                    batch_yes_no_labels.append(yn)
                else:
                    batch_example_ids.append(item["example_id"])
                    batch_candidate_indices.append(cand_idx)

        # Padding
        padded_inputs = pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.pad_id
        )

        result = {"input_ids": padded_inputs}

        if self.is_train:
            result["ranking_labels"] = torch.tensor(
                batch_ranking_labels, dtype=torch.float
            )
            result["start_labels"] = torch.tensor(batch_start_labels, dtype=torch.long)
            result["end_labels"] = torch.tensor(batch_end_labels, dtype=torch.long)
            result["yes_no_labels"] = torch.tensor(
                batch_yes_no_labels, dtype=torch.long
            )
        else:
            result["example_ids"] = batch_example_ids
            result["candidate_indices"] = batch_candidate_indices

        return result


def get_dataloaders(
    load_cached_data: bool = True, debug: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Factory function to create dataloaders for train, val, and test.
    Also returns the vocabulary.
    """
    # 1. Build/Load Vocabulary
    # We need text to build vocab if not cached.
    # To avoid reading the huge file just for vocab if not needed, we assume cache exists
    # or we read a subset of train data.

    vocab = None
    if load_cached_data and os.path.exists(Config.VOCAB_CACHE_PATH):
        vocab = build_vocab(None, load_cached_data=True)

    if vocab is None:
        # If no vocab cache, we must read train data to build it.
        # This is expensive but necessary if cache is missing.
        print("Building vocab from training data...")
        texts = []
        # Read a subset of train data for vocab building to save time
        with open(Config.TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i > 50000:
                    break  # Limit vocab corpus size
                entry = json.loads(line)
                texts.append(entry["question_text"])
                texts.append(entry["document_text"])
        vocab = build_vocab(texts, load_cached_data=False)

    # 2. Preprocess Data
    sample_frac = Config.DATASET_FRACTION if debug else 1.0

    train_df = preprocess_data(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_DATA_PATH,
        Config.PROCESSED_TRAIN_DATA_PATH,
        vocab,
        is_train=True,
        load_cached_data=load_cached_data,
        sample_fraction=sample_frac,
    )

    val_df = preprocess_data(
        Config.VAL_METADATA_PATH,
        Config.TRAIN_DATA_PATH,  # Val is subset of train file
        Config.PROCESSED_VAL_DATA_PATH,
        vocab,
        is_train=True,
        load_cached_data=load_cached_data,
        sample_fraction=sample_frac,
    )

    test_df = preprocess_data(
        Config.TEST_METADATA_PATH,
        Config.TEST_DATA_PATH,
        Config.PROCESSED_TEST_DATA_PATH,
        vocab,
        is_train=False,
        load_cached_data=load_cached_data,
        sample_fraction=1.0,  # Always full test
    )

    # 3. Create Datasets
    train_ds = NQDataset(train_df, is_train=True)
    val_ds = NQDataset(val_df, is_train=True)
    test_ds = NQDataset(test_df, is_train=False)

    # 4. Create Collators
    train_collate = NQCollator(vocab, is_train=True)
    val_collate = NQCollator(vocab, is_train=True)  # Val uses train logic for loss calc
    test_collate = NQCollator(vocab, is_train=False)

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=train_collate,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=val_collate,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=test_collate,
        num_workers=2,
    )

    return train_loader, val_loader, test_loader, vocab
