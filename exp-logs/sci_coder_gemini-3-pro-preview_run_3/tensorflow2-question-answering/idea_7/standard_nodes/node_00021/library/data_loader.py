import os
import json
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from library.utils import (
    Tokenizer,
    parse_candidates,
    load_jsonl_sample,
    ensure_dir,
    CACHE_DIR,
    INPUT_DIR,
    DEFAULT_MAX_SEQ_LEN,
    PAD_TOKEN,
)

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def get_tokenizer(
    metadata_df, load_cached_data=True, vocab_size=50000, sample_size=None
):
    """
    Builds or loads a tokenizer based on the training data.
    """
    vocab_path = os.path.join(CACHE_DIR, "vocab.parquet")
    tokenizer = Tokenizer(vocab_size=vocab_size)

    if load_cached_data and os.path.exists(vocab_path):
        try:
            tokenizer.load(vocab_path)
            return tokenizer
        except Exception as e:
            print(f"Failed to load vocab from cache: {e}. Rebuilding...")

    print("Building vocabulary from training data...")
    texts = []

    # Use a subset for vocab building to save time if sample_size is set
    df = metadata_df
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    for _, row in df.iterrows():
        file_path = os.path.join(INPUT_DIR, row["file_path"])
        data = load_jsonl_sample(file_path, row["byte_offset"])
        if data:
            texts.append(data.get("question_text", ""))
            # Add a portion of document text to capture vocabulary
            doc_text = data.get("document_text", "")
            if doc_text:
                texts.append(doc_text[:5000])

    tokenizer.fit_on_texts(texts)
    tokenizer.save(vocab_path)
    return tokenizer


def process_ranker_data(
    metadata_df,
    tokenizer,
    load_cached_data=True,
    split="train",
    sample_size=None,
    neg_ratio=1,
):
    """
    Preprocesses data for the Ranker model.
    Generates (question, paragraph, label) triplets.
    """
    cache_path = os.path.join(CACHE_DIR, f"ranker_{split}_data.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading ranker data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing ranker data for {split}...")

    records = []

    if sample_size and len(metadata_df) > sample_size:
        metadata_df = metadata_df.sample(n=sample_size, random_state=42)

    for _, row in metadata_df.iterrows():
        file_path = os.path.join(INPUT_DIR, row["file_path"])
        data = load_jsonl_sample(file_path, row["byte_offset"])
        if not data:
            continue

        question_text = data["question_text"]
        doc_tokens = data["document_text"].split()

        # Get Ground Truth Long Answer
        annotations = data.get("annotations", [])
        positive_candidate = None
        la_start = -1

        # Find the positive candidate index if available
        for ann in annotations:
            la = ann["long_answer"]
            if la["start_token"] != -1:
                la_start = la["start_token"]
                la_end = la["end_token"]
                positive_candidate = " ".join(doc_tokens[la_start:la_end])
                break

        candidates = data["long_answer_candidates"]

        # If we have a positive label (train/val), add it and sample negatives
        if positive_candidate:
            records.append(
                {"q_text": question_text, "p_text": positive_candidate, "label": 1}
            )

            # Collect valid negatives (top level candidates that are NOT the ground truth)
            neg_candidates = []
            for cand in candidates:
                if cand["top_level"]:
                    c_start = cand["start_token"]
                    c_end = cand["end_token"]
                    if c_start != la_start:
                        # Ensure we don't crash on bounds
                        if c_end <= len(doc_tokens):
                            neg_text = " ".join(doc_tokens[c_start:c_end])
                            neg_candidates.append(neg_text)

            if neg_candidates:
                num_neg = min(len(neg_candidates), neg_ratio)
                selected_negs = random.sample(neg_candidates, num_neg)
                for neg in selected_negs:
                    records.append({"q_text": question_text, "p_text": neg, "label": 0})
        # Note: For strict test set inference without labels, logic would differ (all candidates),
        # but this function focuses on dataset creation for training/validation.

    df = pd.DataFrame(records)

    print("Tokenizing ranker data...")
    if not df.empty:
        df["q_ids"] = tokenizer.texts_to_sequences(df["q_text"].tolist())
        df["p_ids"] = tokenizer.texts_to_sequences(df["p_text"].tolist())
        # Drop text columns to save memory/disk space
        df = df.drop(columns=["q_text", "p_text"])
    else:
        df["q_ids"] = []
        df["p_ids"] = []

    ensure_dir(CACHE_DIR)
    df.to_parquet(cache_path, index=False)
    print(f"Saved ranker data to {cache_path}")
    return df


def process_reader_data(
    metadata_df, tokenizer, load_cached_data=True, split="train", sample_size=None
):
    """
    Preprocesses data for the Reader model.
    Generates (concatenated_ids, start_idx, end_idx) samples.
    Only uses samples with valid short answers.
    """
    cache_path = os.path.join(CACHE_DIR, f"reader_{split}_data.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading reader data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing reader data for {split}...")

    records = []

    # Filter for short answers if label info is present
    if "has_short_answer" in metadata_df.columns:
        df_filtered = metadata_df[metadata_df["has_short_answer"] == True]
    else:
        df_filtered = metadata_df

    if sample_size and len(df_filtered) > sample_size:
        df_filtered = df_filtered.sample(n=sample_size, random_state=42)

    for _, row in df_filtered.iterrows():
        file_path = os.path.join(INPUT_DIR, row["file_path"])
        data = load_jsonl_sample(file_path, row["byte_offset"])
        if not data:
            continue

        question_text = data["question_text"]
        doc_tokens = data["document_text"].split()
        annotations = data.get("annotations", [])

        for ann in annotations:
            short_answers = ann["short_answers"]
            long_answer = ann["long_answer"]

            # Skip if no short answer or no associated long answer context
            if not short_answers or long_answer["start_token"] == -1:
                continue

            # Take the first short answer span
            sa = short_answers[0]
            s_start = sa["start_token"]
            s_end = sa["end_token"]

            l_start = long_answer["start_token"]
            l_end = long_answer["end_token"]

            # Extract Long Answer Text (Paragraph)
            if l_end > len(doc_tokens):
                l_end = len(doc_tokens)
            paragraph_tokens = doc_tokens[l_start:l_end]
            paragraph_text = " ".join(paragraph_tokens)

            # Calculate relative indices within the paragraph
            rel_start = s_start - l_start
            rel_end = s_end - l_start

            # Validate indices
            if rel_start < 0 or rel_end > len(paragraph_tokens):
                continue

            records.append(
                {
                    "q_text": question_text,
                    "p_text": paragraph_text,
                    "rel_start": rel_start,
                    "rel_end": rel_end,
                }
            )
            # Use one valid annotation per document
            break

    df = pd.DataFrame(records)

    print("Tokenizing reader data...")
    if not df.empty:
        q_seqs = tokenizer.texts_to_sequences(df["q_text"].tolist())
        p_seqs = tokenizer.texts_to_sequences(df["p_text"].tolist())

        combined_ids = []
        start_indices = []
        end_indices = []

        for i in range(len(df)):
            q = q_seqs[i]
            p = p_seqs[i]
            rel_s = df.iloc[i]["rel_start"]
            rel_e = df.iloc[i]["rel_end"]

            # Concatenate: Q + P
            combined = q + p

            # Adjust indices: The paragraph starts after len(q)
            # End index is usually exclusive in dataset but inclusive for classification
            final_start = len(q) + rel_s
            final_end = len(q) + rel_e - 1

            # Filter if sequence is too long or indices are out of bounds
            if final_end < DEFAULT_MAX_SEQ_LEN:
                combined_ids.append(combined)
                start_indices.append(final_start)
                end_indices.append(final_end)

        final_df = pd.DataFrame(
            {
                "input_ids": combined_ids,
                "start_idx": start_indices,
                "end_idx": end_indices,
            }
        )
    else:
        final_df = pd.DataFrame(columns=["input_ids", "start_idx", "end_idx"])

    ensure_dir(CACHE_DIR)
    final_df.to_parquet(cache_path, index=False)
    print(f"Saved reader data to {cache_path}")
    return final_df


class NQRankerDataset(Dataset):
    def __init__(self, data_df):
        self.data = data_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Data loaded from parquet might be numpy arrays, convert to list or tensor
        return {
            "q_ids": list(row["q_ids"]),
            "p_ids": list(row["p_ids"]),
            "label": row["label"],
        }


class NQReaderDataset(Dataset):
    def __init__(self, data_df):
        self.data = data_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            "input_ids": list(row["input_ids"]),
            "start_idx": row["start_idx"],
            "end_idx": row["end_idx"],
        }


def ranker_collate_fn(batch):
    """
    Pads batch of variable length sequences for Ranker.
    """
    q_lens = [len(x["q_ids"]) for x in batch]
    p_lens = [len(x["p_ids"]) for x in batch]
    max_q = max(q_lens) if q_lens else 0
    max_p = max(p_lens) if p_lens else 0

    # Cap at default max len
    max_q = min(max_q, DEFAULT_MAX_SEQ_LEN)
    max_p = min(max_p, DEFAULT_MAX_SEQ_LEN)

    q_batch = []
    p_batch = []
    labels = []

    for x in batch:
        # Pad Q
        q = x["q_ids"][:max_q]
        q_pad = q + [0] * (max_q - len(q))
        q_batch.append(q_pad)

        # Pad P
        p = x["p_ids"][:max_p]
        p_pad = p + [0] * (max_p - len(p))
        p_batch.append(p_pad)

        labels.append(x["label"])

    return (
        torch.tensor(q_batch, dtype=torch.long),
        torch.tensor(p_batch, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32).unsqueeze(1),
    )


def reader_collate_fn(batch):
    """
    Pads batch for Reader.
    """
    lens = [len(x["input_ids"]) for x in batch]
    max_len = max(lens) if lens else 0
    max_len = min(max_len, DEFAULT_MAX_SEQ_LEN)

    input_batch = []
    start_batch = []
    end_batch = []

    for x in batch:
        seq = x["input_ids"][:max_len]
        seq_pad = seq + [0] * (max_len - len(seq))
        input_batch.append(seq_pad)

        # Clamp indices to valid range
        s_idx = min(x["start_idx"], max_len - 1)
        e_idx = min(x["end_idx"], max_len - 1)

        start_batch.append(s_idx)
        end_batch.append(e_idx)

    return (
        torch.tensor(input_batch, dtype=torch.long),
        torch.tensor(start_batch, dtype=torch.long),
        torch.tensor(end_batch, dtype=torch.long),
    )
