import os
import json
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.data_processing import Tokenizer, pad_sequence

# Mapping for Yes/No answers
YESNO_MAP = {"NONE": 0, "YES": 1, "NO": 2}


def preprocess_and_cache(
    metadata_path, data_path, vocab, cache_path, is_train=True, filter_negatives=True
):
    """
    Reads metadata and raw JSONL, tokenizes text, maps to indices, and caches to Parquet.
    Implements the strict caching logic required.
    """
    # 1. Try to load from cache
    if os.path.exists(cache_path):
        try:
            # We assume if it exists, it's valid.
            # Using pyarrow engine for efficient list column handling
            df = pd.read_parquet(cache_path, engine="pyarrow")
            return df
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    # 2. Compute from scratch
    # Load metadata to know which examples to process
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Enforce string type for example_id to ensure correct matching with JSON strings (Cite debug_lesson_9)
    meta_df = pd.read_csv(metadata_path, dtype={"example_id": str})

    # For training, we filter to examples that actually have a long answer
    # to ensure we can form Positive-Negative pairs.
    if is_train and filter_negatives:
        # candidate_index != -1 means a long answer exists
        meta_df = meta_df[meta_df["long_answer_index"] != -1].copy()

    target_ids = set(meta_df["example_id"].astype(str))

    tokenizer = Tokenizer()
    processed_rows = []

    # Stream the large JSONL file
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            ex_id = str(entry["example_id"])

            if ex_id not in target_ids:
                continue

            # Tokenize Question
            q_text = entry.get("question_text", "")
            q_tokens = tokenizer.tokenize(q_text)
            q_indices = vocab.lookup_indices(q_tokens)

            # Tokenize Document
            doc_text = entry.get("document_text", "")
            doc_tokens = tokenizer.tokenize(doc_text)
            doc_indices = vocab.lookup_indices(doc_tokens)

            # Extract Candidates (start, end)
            raw_candidates = entry.get("long_answer_candidates", [])
            candidates_list = []
            for c in raw_candidates:
                candidates_list.append([c["start_token"], c["end_token"]])

            # Base row data
            row = {
                "example_id": ex_id,
                "q_indices": q_indices,
                "doc_indices": doc_indices,
                "candidates": candidates_list,
            }

            # Extract Labels if training data
            # We extract directly from JSON to ensure accuracy
            if is_train:
                anns = entry.get("annotations", [])
                la_idx = -1
                yn_label = 0
                short_span = [-1, -1]

                if anns:
                    ann = anns[0]
                    # Long Answer Index
                    la_idx = ann.get("long_answer", {}).get("candidate_index", -1)

                    # Yes/No
                    yes_no = ann.get("yes_no_answer", "NONE")
                    yn_label = YESNO_MAP.get(yes_no, 0)

                    # Short Answer
                    sas = ann.get("short_answers", [])
                    if sas:
                        # Take the first short answer
                        short_span = [sas[0]["start_token"], sas[0]["end_token"]]

                row["label_la_idx"] = la_idx
                row["label_yn"] = yn_label
                row["label_short_span"] = short_span

            processed_rows.append(row)

    df = pd.DataFrame(processed_rows)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    # Use pyarrow to support list columns
    df.to_parquet(cache_path, engine="pyarrow")

    return df


class NQDataset(Dataset):
    def __init__(self, dataframe, vocab, is_train=True):
        self.data = dataframe
        self.vocab = vocab
        self.is_train = is_train
        self.max_q_len = Config.MAX_Q_LEN
        self.max_ctx_len = Config.MAX_CTX_LEN
        self.pad_idx = vocab.pad_index

    def __len__(self):
        return len(self.data)

    def _get_candidate_seq(self, doc_indices, start, end):
        # Slice document indices
        span = doc_indices[start:end]
        # Pad/Truncate
        return pad_sequence(span, self.max_ctx_len, self.pad_idx)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Prepare Question
        q_indices = row["q_indices"]
        # Ensure it's a list (parquet might return numpy array)
        if isinstance(q_indices, np.ndarray):
            q_indices = q_indices.tolist()
        q_seq = pad_sequence(q_indices, self.max_q_len, self.pad_idx)

        doc_indices = row["doc_indices"]
        if isinstance(doc_indices, np.ndarray):
            doc_indices = doc_indices.tolist()

        candidates = row["candidates"]
        if isinstance(candidates, np.ndarray):
            candidates = candidates.tolist()

        if self.is_train:
            # Training Mode: Return Positive and Negative Pair
            la_idx = int(row["label_la_idx"])

            # --- Positive Sample ---
            pos_cand = candidates[la_idx]
            p_start, p_end = pos_cand
            pos_seq = self._get_candidate_seq(doc_indices, p_start, p_end)

            # Short Answer Targets (Relative to candidate start)
            s_start, s_end = row["label_short_span"]
            if s_start != -1:
                # Calculate relative offset
                # Check if short answer is actually within this long answer candidate
                if s_start >= p_start and s_end <= p_end:
                    rel_start = s_start - p_start
                    # End index is exclusive in slice, but usually inclusive in span prediction
                    # We'll make it inclusive for the model target
                    rel_end = s_end - p_start - 1

                    # Clamp to max length
                    if rel_end >= self.max_ctx_len:
                        rel_start, rel_end = 0, 0
                else:
                    rel_start, rel_end = 0, 0
            else:
                rel_start, rel_end = 0, 0

            # Yes/No Target
            yn_label = int(row["label_yn"])

            # --- Negative Sample ---
            # Randomly sample a candidate that is NOT the correct one
            num_cands = len(candidates)
            if num_cands > 1:
                neg_idx = la_idx
                while neg_idx == la_idx:
                    neg_idx = random.randint(0, num_cands - 1)

                n_start, n_end = candidates[neg_idx]
                neg_seq = self._get_candidate_seq(doc_indices, n_start, n_end)
            else:
                # Fallback if only 1 candidate (rare/impossible if filtered correctly)
                neg_seq = np.full(self.max_ctx_len, self.pad_idx, dtype=np.int64)

            return {
                "q_input": torch.tensor(q_seq, dtype=torch.long),
                "pos_cand_input": torch.tensor(pos_seq, dtype=torch.long),
                "neg_cand_input": torch.tensor(neg_seq, dtype=torch.long),
                "pos_label": torch.tensor(1.0, dtype=torch.float),
                "neg_label": torch.tensor(0.0, dtype=torch.float),
                "short_start": torch.tensor(rel_start, dtype=torch.long),
                "short_end": torch.tensor(rel_end, dtype=torch.long),
                "yn_label": torch.tensor(yn_label, dtype=torch.long),
                "has_short": torch.tensor(
                    1.0 if s_start != -1 else 0.0, dtype=torch.float
                ),
            }

        else:
            # Inference/Validation Mode: Return Question and ALL candidates
            # We limit the number of candidates to avoid OOM
            MAX_EVAL_CANDS = 30

            cands_batch = []
            for i, (cs, ce) in enumerate(candidates):
                if i >= MAX_EVAL_CANDS:
                    break
                c_seq = self._get_candidate_seq(doc_indices, cs, ce)
                cands_batch.append(c_seq)

            cands_tensor = np.array(cands_batch, dtype=np.int64)

            # If validation, we might have labels
            la_idx = -1
            if "label_la_idx" in row:
                la_idx = int(row["label_la_idx"])

            return {
                "example_id": row["example_id"],
                "q_input": torch.tensor(q_seq, dtype=torch.long),
                "candidates": torch.tensor(cands_tensor, dtype=torch.long),
                "label_idx": torch.tensor(la_idx, dtype=torch.long),
            }


def get_dataloaders(vocab, load_cached_data=True):
    """
    Generates DataLoaders for training and validation.
    """
    # Train Data
    train_df = preprocess_and_cache(
        Config.TRAIN_META,
        Config.TRAIN_FILE,
        vocab,
        Config.TRAIN_CACHE,
        is_train=True,
        filter_negatives=True,
    )
    train_ds = NQDataset(train_df, vocab, is_train=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Validation Data
    # We use is_train=True in preprocessing to extract labels,
    # but filter_negatives=False to keep ALL examples for valid metric calculation.
    # We use is_train=False in Dataset to get inference structure (all candidates)
    val_df = preprocess_and_cache(
        Config.VAL_META,
        Config.TRAIN_FILE,
        vocab,
        Config.VAL_CACHE,
        is_train=True,
        filter_negatives=False,
    )
    val_ds = NQDataset(val_df, vocab, is_train=False)

    # Batch size 1 for validation because number of candidates varies per example
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader


def get_test_dataloader(vocab, load_cached_data=True):
    """
    Generates DataLoader for the test set.
    """
    test_df = preprocess_and_cache(
        Config.TEST_META,
        Config.TEST_FILE,
        vocab,
        Config.TEST_CACHE,
        is_train=False,
        filter_negatives=False,
    )
    test_ds = NQDataset(test_df, vocab, is_train=False)

    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    return test_loader
