import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from library.utils import manage_caching, load_metadata

# Constants
INPUT_DIR = "./input"
TRAIN_FILE = "simplified-nq-train.jsonl"
TEST_FILE = "simplified-nq-test.jsonl"
CACHE_DIR = "./working/idea_3/"


class SimpleTokenizer:
    def __init__(self, vocab=None, min_freq=2, pad_token="<PAD>", unk_token="<UNK>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = {pad_token: 0, unk_token: 1}

        self.pad_id = self.vocab[self.pad_token]
        self.unk_id = self.vocab[self.unk_token]
        self.id_to_word = {v: k for k, v in self.vocab.items()}
        self.min_freq = min_freq

    def fit_on_texts(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        idx = len(self.vocab)
        # Ensure PAD and UNK are preserved at 0 and 1
        if self.pad_token not in self.vocab:
            self.vocab[self.pad_token] = 0
            idx = max(idx, 1)
        if self.unk_token not in self.vocab:
            self.vocab[self.unk_token] = 1
            idx = max(idx, 2)

        for word, count in counter.items():
            if count >= self.min_freq:
                if word not in self.vocab:
                    self.vocab[word] = idx
                    self.id_to_word[idx] = word
                    idx += 1

    def encode(self, text):
        return [self.vocab.get(w, self.unk_id) for w in text.split()]

    def __len__(self):
        return len(self.vocab)


def build_tokenizer(metadata_df, sample_size=20000, load_cached_data=True):
    """
    Builds a SimpleTokenizer from a sample of the training data.
    Caches the vocabulary as a parquet file.
    """

    def _generate_vocab_df():
        # Sample rows to build vocab efficiently
        if len(metadata_df) > sample_size:
            sample = metadata_df.sample(n=sample_size, random_state=42)
        else:
            sample = metadata_df

        texts = []

        # Group by file path to optimize IO
        for file_path, group in sample.groupby("file_path"):
            abs_path = os.path.join(INPUT_DIR, file_path)
            if not os.path.exists(abs_path):
                continue

            with open(abs_path, "rb") as f:
                for _, row in group.iterrows():
                    f.seek(row["byte_offset"])
                    line = f.readline()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Add question text
                        texts.append(data.get("question_text", ""))
                        # Add partial document text (first 1000 tokens) to capture common domain vocab
                        doc_text = data.get("document_text", "")
                        texts.append(" ".join(doc_text.split()[:1000]))
                    except:
                        continue

        temp_tokenizer = SimpleTokenizer()
        temp_tokenizer.fit_on_texts(texts)

        # Convert dict to DataFrame for parquet caching
        df = pd.DataFrame(list(temp_tokenizer.vocab.items()), columns=["token", "id"])
        return df

    vocab_df = manage_caching(
        "vocab.parquet", _generate_vocab_df, load_cached_data=load_cached_data
    )

    # Reconstruct dictionary from DataFrame
    vocab = dict(zip(vocab_df["token"], vocab_df["id"]))
    return SimpleTokenizer(vocab=vocab)


def process_flattened_data(metadata_df, split, neg_ratio=0.2, load_cached_data=True):
    """
    Flattens the dataset: (Question, Candidate) pairs.
    For train: Subsamples negatives.
    For val/test: Keeps top-level candidates.
    """
    cache_filename = f"{split}_flattened.parquet"

    def _generate():
        records = []

        # Group by file path to optimize IO
        for file_path, group in metadata_df.groupby("file_path"):
            abs_path = os.path.join(INPUT_DIR, file_path)
            if not os.path.exists(abs_path):
                continue

            with open(abs_path, "rb") as f:
                for _, row in group.iterrows():
                    f.seek(row["byte_offset"])
                    line = f.readline()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except:
                        continue

                    example_id = data["example_id"]
                    candidates = data.get("long_answer_candidates", [])

                    # Parse Annotations (if available)
                    correct_candidate_idx = -1
                    short_answers = []

                    # In 'train'/'val' splits, we have annotations in the metadata dataframe
                    # But here we are reading raw JSON to get candidates.
                    # The metadata row actually contains 'annotations' as a JSON string.
                    # We can use that instead of parsing raw file annotations if we trust metadata.
                    # However, for consistency with test (which has no annotations), let's rely on
                    # the logic that extracts labels.

                    # For training/val, we need ground truth.
                    if split != "test":
                        # The metadata row has annotations
                        if row["annotations"] is not None:
                            try:
                                anns = json.loads(row["annotations"])
                                for ann in anns:
                                    la = ann.get("long_answer", {})
                                    idx = la.get("candidate_index", -1)
                                    if idx != -1:
                                        correct_candidate_idx = idx
                                        short_answers = ann.get("short_answers", [])
                                        break
                            except:
                                pass

                    # Select candidates to include in the dataset
                    selected_indices = []

                    if split == "train":
                        # 1. Positive Sample
                        if correct_candidate_idx != -1:
                            selected_indices.append(correct_candidate_idx)

                            # 2. Negative Samples
                            all_indices = list(range(len(candidates)))
                            all_indices.remove(correct_candidate_idx)

                            # Subsample
                            num_neg = max(1, int(len(all_indices) * neg_ratio))
                            num_neg = min(
                                num_neg, 5
                            )  # Cap negatives to prevent explosion

                            if all_indices:
                                negs = random.sample(
                                    all_indices, min(len(all_indices), num_neg)
                                )
                                selected_indices.extend(negs)
                        else:
                            # No answer question. Include a few random candidates as negatives.
                            if candidates:
                                num_neg = min(len(candidates), 2)
                                selected_indices = random.sample(
                                    range(len(candidates)), num_neg
                                )

                    else:
                        # Val/Test: Select candidates for inference.
                        # We select all "top_level" candidates (usually paragraphs).
                        for i, c in enumerate(candidates):
                            if c.get("top_level", False):
                                selected_indices.append(i)

                        # Fallback if no top level found
                        if not selected_indices and candidates:
                            selected_indices = list(range(min(len(candidates), 20)))

                    # Create records for selected candidates
                    for cand_idx in selected_indices:
                        is_correct = cand_idx == correct_candidate_idx

                        # Short Answer Logic
                        sa_start_local = -1
                        sa_end_local = -1
                        sa_label = 0  # 0: None, 1: Exists

                        if is_correct and short_answers:
                            cand_start = candidates[cand_idx]["start_token"]
                            cand_end = candidates[cand_idx]["end_token"]

                            # Use first short answer
                            sa = short_answers[0]
                            s_start = sa["start_token"]
                            s_end = sa["end_token"]

                            # Check containment
                            if s_start >= cand_start and s_end <= cand_end:
                                sa_start_local = s_start - cand_start
                                sa_end_local = s_end - cand_start - 1  # Inclusive index
                                sa_label = 1

                        records.append(
                            {
                                "example_id": example_id,
                                "file_path": file_path,
                                "byte_offset": row["byte_offset"],
                                "candidate_index": cand_idx,
                                "label_long": 1 if is_correct else 0,
                                "label_short_exists": sa_label,
                                "sa_start_local": sa_start_local,
                                "sa_end_local": sa_end_local,
                            }
                        )

        return pd.DataFrame(records)

    return manage_caching(cache_filename, _generate, load_cached_data=load_cached_data)


class NQDataset(Dataset):
    def __init__(
        self, split, tokenizer, max_len=256, neg_ratio=0.2, load_cached_data=True
    ):
        self.split = split
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Load Metadata
        self.metadata = load_metadata(split)

        # Flatten Data
        self.data = process_flattened_data(
            self.metadata, split, neg_ratio, load_cached_data
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = os.path.join(INPUT_DIR, row["file_path"])

        # Read raw data
        with open(file_path, "rb") as f:
            f.seek(row["byte_offset"])
            line = f.readline()
            json_data = json.loads(line)

        question = json_data["question_text"]
        doc_tokens = json_data["document_text"].split()

        cand_idx = int(row["candidate_index"])
        candidates = json_data["long_answer_candidates"]

        # Extract Candidate Text
        if cand_idx < len(candidates):
            cand_struct = candidates[cand_idx]
            start = cand_struct["start_token"]
            end = cand_struct["end_token"]
            # Bounds check
            start = max(0, start)
            end = min(len(doc_tokens), end)
            candidate_tokens = doc_tokens[start:end]
            candidate_text = " ".join(candidate_tokens)
            token_offset = start
        else:
            candidate_text = ""
            token_offset = 0

        # Tokenize
        q_ids = self.tokenizer.encode(question)
        c_ids = self.tokenizer.encode(candidate_text)

        # Truncate Candidate if necessary
        if len(c_ids) > self.max_len:
            c_ids = c_ids[: self.max_len]

        # Prepare Labels
        label_long = float(row["label_long"])

        # Short Answer Labels (Per Token)
        # 0: Neither, 1: Start, 2: End
        sa_labels = [0] * len(c_ids)

        if row["label_short_exists"] == 1:
            s_start = int(row["sa_start_local"])
            s_end = int(row["sa_end_local"])

            if 0 <= s_start < len(c_ids):
                sa_labels[s_start] = 1
            if 0 <= s_end < len(c_ids):
                sa_labels[s_end] = 2

        return {
            "q_ids": torch.tensor(q_ids, dtype=torch.long),
            "c_ids": torch.tensor(c_ids, dtype=torch.long),
            "label_long": torch.tensor(label_long, dtype=torch.float),
            "sa_labels": torch.tensor(sa_labels, dtype=torch.long),
            "example_id": row["example_id"],
            "candidate_index": cand_idx,
            "token_offset": token_offset,
        }


def collate_fn(batch):
    q_ids = [item["q_ids"] for item in batch]
    c_ids = [item["c_ids"] for item in batch]
    label_long = [item["label_long"] for item in batch]
    sa_labels = [item["sa_labels"] for item in batch]
    example_ids = [item["example_id"] for item in batch]
    candidate_indices = [item["candidate_index"] for item in batch]
    token_offsets = [item["token_offset"] for item in batch]

    # Pad sequences
    # Padding value is 0 (which corresponds to <PAD> in SimpleTokenizer)
    q_ids_padded = pad_sequence(q_ids, batch_first=True, padding_value=0)
    c_ids_padded = pad_sequence(c_ids, batch_first=True, padding_value=0)

    # Pad labels with -1 or 0?
    # For CrossEntropyLoss, we usually use -100 to ignore.
    # But here 0 is 'Neither'. If we pad with 0, the model learns 'Neither' for pads.
    # Ideally use -1 and ignore_index in loss.
    sa_labels_padded = pad_sequence(sa_labels, batch_first=True, padding_value=-1)

    label_long = torch.stack(label_long)

    return {
        "q_input_ids": q_ids_padded,
        "c_input_ids": c_ids_padded,
        "label_long": label_long,
        "sa_labels": sa_labels_padded,
        "example_ids": example_ids,
        "candidate_indices": candidate_indices,
        "token_offsets": token_offsets,
    }


def get_dataloader(
    split,
    tokenizer,
    batch_size=32,
    max_len=256,
    neg_ratio=0.2,
    num_workers=2,
    load_cached_data=True,
):
    dataset = NQDataset(split, tokenizer, max_len, neg_ratio, load_cached_data)
    shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
