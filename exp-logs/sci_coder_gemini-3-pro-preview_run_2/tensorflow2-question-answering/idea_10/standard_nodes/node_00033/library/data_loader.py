import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.text_utils import build_or_load_tokenizer, segment_sentences

# Constants for Yes/No labels
YES_NO_MAP = {"NONE": 0, "YES": 1, "NO": 2}


def serialize_list(data_list):
    """Serializes a list (or list of lists) to a JSON string for Parquet storage."""
    return json.dumps(data_list)


def deserialize_list(json_str):
    """Deserializes a JSON string back to a list."""
    return json.loads(json_str)


def check_overlap(span1, span2):
    """Checks if two spans (start, end) overlap."""
    # span is [start, end)
    return max(span1[0], span2[0]) < min(span1[1], span2[1])


def process_and_cache_data(mode, tokenizer, load_cached_data=True, sample_size=None):
    """
    Reads raw data, processes it into sentence-level features, and caches to Parquet.
    """
    # Determine paths based on mode
    if mode == "train":
        raw_path = Config.TRAIN_DATA_PATH
        meta_path = Config.TRAIN_META_PATH
        cache_path = Config.TRAIN_CACHE_PATH
    elif mode == "val":
        raw_path = Config.TRAIN_DATA_PATH  # Val is a subset of train file
        meta_path = Config.VAL_META_PATH
        cache_path = Config.VAL_CACHE_PATH
    elif mode == "test":
        raw_path = Config.TEST_DATA_PATH
        meta_path = Config.TEST_META_PATH
        cache_path = Config.TEST_CACHE_PATH
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {mode} data from cache: {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            # Deserialize complex columns
            list_cols = [
                "question_ids",
                "sentence_ids",
                "sentence_labels",
                "candidate_map",
            ]
            for col in list_cols:
                if col in df.columns:
                    df[col] = df[col].apply(deserialize_list)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing...")

    # 2. Process from scratch
    print(f"Processing {mode} data from {raw_path}...")

    # Load metadata to filter examples
    meta_df = pd.read_csv(meta_path)
    valid_ids = set(meta_df["example_id"].astype(str))

    # Pre-fetch labels for train/val from metadata if available (faster than parsing annotations again)
    # However, we need precise spans from the raw JSONL for sentence labeling.

    processed_rows = []
    count = 0

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            ex_id = str(entry["example_id"])

            if ex_id not in valid_ids:
                continue

            if sample_size and count >= sample_size:
                break

            # --- Text Processing ---
            doc_text = entry.get("document_text", "")
            question_text = entry.get("question_text", "")

            # Tokenize Question
            q_ids = tokenizer.text_to_sequence(question_text, max_len=Config.MAX_Q_LEN)

            # Segment Document
            sentences = segment_sentences(doc_text)

            # Limit sentences per doc to avoid OOM
            if len(sentences) > Config.MAX_SENTS_PER_DOC:
                sentences = sentences[: Config.MAX_SENTS_PER_DOC]

            # Tokenize Sentences
            sent_ids_list = []
            for sent in sentences:
                s_ids = tokenizer.text_to_sequence(
                    sent["text"], max_len=Config.MAX_SENT_LEN
                )
                sent_ids_list.append(s_ids)

            # --- Labeling & Mapping ---
            sent_labels = [0] * len(sentences)
            yes_no_label = 0
            candidate_map = []  # List of lists: candidate_index -> [sent_indices]

            candidates = entry.get("long_answer_candidates", [])

            # Initialize candidate map
            candidate_map = [[] for _ in range(len(candidates))]

            # Map sentences to candidates
            for s_idx, sent in enumerate(sentences):
                s_start = sent["start_token_idx"]
                s_end = sent["end_token_idx"]

                for c_idx, cand in enumerate(candidates):
                    c_start = cand["start_token"]
                    c_end = cand["end_token"]

                    # Check if sentence is contained in candidate
                    # Relaxed condition: if they overlap significantly
                    if check_overlap((s_start, s_end), (c_start, c_end)):
                        candidate_map[c_idx].append(s_idx)

            if mode in ["train", "val"]:
                annotations = entry.get("annotations", [])
                if annotations:
                    ann = annotations[0]

                    # Yes/No Label
                    yn_str = ann.get("yes_no_answer", "NONE")
                    yes_no_label = YES_NO_MAP.get(yn_str, 0)

                    # Short Answer Labels
                    short_answers = ann.get("short_answers", [])
                    if short_answers:
                        for sa in short_answers:
                            sa_start = sa["start_token"]
                            sa_end = sa["end_token"]

                            # Mark sentences that overlap with short answer
                            for s_idx, sent in enumerate(sentences):
                                s_start = sent["start_token_idx"]
                                s_end = sent["end_token_idx"]
                                if check_overlap((s_start, s_end), (sa_start, sa_end)):
                                    sent_labels[s_idx] = 1

            processed_rows.append(
                {
                    "example_id": ex_id,
                    "question_ids": serialize_list(q_ids),
                    "sentence_ids": serialize_list(sent_ids_list),
                    "sentence_labels": serialize_list(sent_labels),
                    "candidate_map": serialize_list(candidate_map),
                    "yes_no": yes_no_label,
                }
            )

            count += 1
            if count % 5000 == 0:
                print(f"Processed {count} records for {mode}...")

    df = pd.DataFrame(processed_rows)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"Saved {len(df)} rows to {cache_path}")

    # Deserialize for immediate use
    for col in ["question_ids", "sentence_ids", "sentence_labels", "candidate_map"]:
        df[col] = df[col].apply(deserialize_list)

    return df


class NQSentenceDataset(Dataset):
    def __init__(self, dataframe, mode="train"):
        self.data = dataframe
        self.mode = mode
        self.neg_ratio = Config.NEGATIVE_SAMPLE_RATIO

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        q_ids = row["question_ids"]
        all_sents = row["sentence_ids"]
        all_labels = row["sentence_labels"]
        yes_no = row["yes_no"]

        # If no sentences (empty doc), return placeholders
        if not all_sents:
            return {
                "question": q_ids,
                "sentences": [[0] * Config.MAX_SENT_LEN],  # 1 dummy sentence
                "labels": [0],
                "yes_no": yes_no,
                "candidate_map": [],
                "example_id": row["example_id"],
                "original_indices": [0],
            }

        if self.mode == "train":
            # Random Negative Sampling
            pos_indices = [i for i, label in enumerate(all_labels) if label == 1]
            neg_indices = [i for i, label in enumerate(all_labels) if label == 0]

            selected_indices = []

            # Always include all positives
            selected_indices.extend(pos_indices)

            # Sample negatives
            num_pos = len(pos_indices)
            # If we have positives, sample k * num_pos negatives
            # If no positives, sample a fixed small number to learn "no answer"
            num_neg_to_sample = (
                max(1, num_pos * self.neg_ratio) if num_pos > 0 else self.neg_ratio
            )

            if len(neg_indices) > num_neg_to_sample:
                selected_negs = random.sample(neg_indices, num_neg_to_sample)
            else:
                selected_negs = neg_indices

            selected_indices.extend(selected_negs)

            # Shuffle to mix pos and neg
            random.shuffle(selected_indices)

            batch_sents = [all_sents[i] for i in selected_indices]
            batch_labels = [all_labels[i] for i in selected_indices]

            return {
                "question": q_ids,
                "sentences": batch_sents,
                "labels": batch_labels,
                "yes_no": yes_no,
                "example_id": row["example_id"],
            }

        else:
            # Inference/Validation: Return all sentences
            return {
                "question": q_ids,
                "sentences": all_sents,
                "labels": all_labels,
                "yes_no": yes_no,
                "candidate_map": row["candidate_map"],
                "example_id": row["example_id"],
            }


def collate_fn(batch):
    """
    Custom collate function to handle variable number of sentences per document.
    """
    # batch is a list of dicts

    questions = []
    flat_sentences = []
    flat_labels = []
    yes_no_labels = []

    # Metadata for reconstruction/tracking
    doc_lengths = []  # How many sentences in each doc
    example_ids = []
    candidate_maps = []  # Only needed for inference

    for item in batch:
        questions.append(item["question"])
        flat_sentences.extend(item["sentences"])
        flat_labels.extend(item["labels"])
        yes_no_labels.append(item["yes_no"])

        doc_lengths.append(len(item["sentences"]))
        example_ids.append(item["example_id"])

        if "candidate_map" in item:
            candidate_maps.append(item["candidate_map"])

    # Convert to tensors
    # Questions: (batch_size, max_q_len)
    q_tensor = torch.tensor(questions, dtype=torch.long)

    # Sentences: (total_sentences_in_batch, max_sent_len)
    s_tensor = torch.tensor(flat_sentences, dtype=torch.long)

    # Labels: (total_sentences_in_batch)
    l_tensor = torch.tensor(flat_labels, dtype=torch.float)

    # Yes/No: (batch_size)
    yn_tensor = torch.tensor(yes_no_labels, dtype=torch.long)

    return {
        "questions": q_tensor,
        "sentences": s_tensor,
        "labels": l_tensor,
        "yes_no": yn_tensor,
        "doc_lengths": doc_lengths,
        "example_ids": example_ids,
        "candidate_maps": candidate_maps,
    }


def get_data_loader(
    mode, tokenizer, batch_size=Config.BATCH_SIZE, shuffle=True, sample_size=None
):
    """
    Factory function to create DataLoaders.
    """
    # Load Data
    df = process_and_cache_data(
        mode, tokenizer, load_cached_data=True, sample_size=sample_size
    )

    # Create Dataset
    dataset = NQSentenceDataset(df, mode=mode)

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    return loader
