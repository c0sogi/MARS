import os
import json
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter
import glob

# Import from provided library files
from library.config import Config
from library.utils import tokenize_text, parse_html_candidates

# Set seeds for reproducibility
random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


class Vocabulary:
    """
    Handles mapping between tokens and integer indices.
    """

    def __init__(self):
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.unk_token = Config.UNK_TOKEN
        self.pad_token = Config.PAD_TOKEN

    def build(self, texts, max_size=Config.VOCAB_SIZE):
        """
        Builds vocabulary from a list of tokenized texts.
        """
        counter = Counter()
        for tokens in texts:
            counter.update(tokens)

        # Start with special tokens
        self.token_to_idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx_to_token = {0: self.pad_token, 1: self.unk_token}

        # Add most common tokens
        for token, _ in counter.most_common(max_size - 2):
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

    def convert_tokens_to_ids(self, tokens, max_len=None):
        """
        Converts a list of tokens to a list of IDs, with truncation and padding.
        """
        ids = [
            self.token_to_idx.get(token, self.token_to_idx[self.unk_token])
            for token in tokens
        ]
        if max_len:
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids += [self.token_to_idx[self.pad_token]] * (max_len - len(ids))
        return ids

    def save(self, path):
        """
        Saves vocabulary to a parquet file.
        """
        data = {
            "token": list(self.token_to_idx.keys()),
            "idx": list(self.token_to_idx.values()),
        }
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"Vocabulary saved to {path}")

    def load(self, path):
        """
        Loads vocabulary from a parquet file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")
        df = pd.read_parquet(path)
        self.token_to_idx = dict(zip(df["token"], df["idx"]))
        self.idx_to_token = dict(zip(df["idx"], df["token"]))
        print(f"Vocabulary loaded from {path}. Size: {len(self.token_to_idx)}")


def get_vocabulary(load_cached_data=True):
    """
    Factory function to get or build the vocabulary.
    """
    vocab = Vocabulary()
    if load_cached_data and os.path.exists(Config.VOCAB_CACHE_PATH):
        vocab.load(Config.VOCAB_CACHE_PATH)
    else:
        print("Building vocabulary from scratch...")
        # Load training metadata to sample texts
        if not os.path.exists(Config.TRAIN_METADATA_PATH):
            raise FileNotFoundError(
                f"Metadata not found at {Config.TRAIN_METADATA_PATH}"
            )

        metadata = pd.read_csv(Config.TRAIN_METADATA_PATH)

        # Limit sample size for vocab building
        sample_size = (
            min(len(metadata), Config.TRAIN_SAMPLE_SIZE)
            if Config.TRAIN_SAMPLE_SIZE
            else len(metadata)
        )
        sample_meta = metadata.sample(n=sample_size, random_state=Config.SEED)

        texts = []
        with open(Config.TRAIN_FILE, "rb") as f:
            for _, row in sample_meta.iterrows():
                f.seek(row["byte_offset"])
                line = f.readline()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Add question tokens
                    texts.append(tokenize_text(data.get("question_text", "")))
                    # Add document tokens (truncated)
                    doc_tokens = tokenize_text(data.get("document_text", ""))
                    texts.append(doc_tokens[: Config.MAX_DOC_LEN])
                except json.JSONDecodeError:
                    continue

        vocab.build(texts)
        vocab.save(Config.VOCAB_CACHE_PATH)

    return vocab


def _process_ranker_data_logic(metadata_path, raw_file_path, sample_size=None):
    """
    Generates (Question, Candidate, Label) triplets for the ranker.
    Label 1: Candidate contains the long answer.
    Label 0: Candidate does not contain the answer (negative sample).
    """
    metadata = pd.read_csv(metadata_path)

    # Filter for examples that have a long answer
    train_df = metadata[metadata["has_long_answer"] == True].copy()

    if sample_size and len(train_df) > sample_size:
        train_df = train_df.sample(n=sample_size, random_state=Config.SEED)

    data_samples = []

    with open(raw_file_path, "rb") as f:
        for _, row in train_df.iterrows():
            f.seek(row["byte_offset"])
            line = f.readline()
            if not line:
                continue

            try:
                entry = json.loads(line)
                question_text = entry["question_text"]
                doc_text = entry["document_text"]
                doc_tokens = tokenize_text(doc_text)

                # Get ground truth long answer
                annotation = entry["annotations"][0]
                la_start = annotation["long_answer"]["start_token"]
                la_end = annotation["long_answer"]["end_token"]

                if la_start == -1:
                    continue

                # Parse candidates
                candidates = parse_html_candidates(doc_tokens)

                # Find positive candidate index
                pos_cand_idx = -1
                for idx, (c_start, c_end) in enumerate(candidates):
                    if c_start == la_start and c_end == la_end:
                        pos_cand_idx = idx
                        break

                if pos_cand_idx == -1:
                    continue

                # Positive sample
                pos_tokens = doc_tokens[
                    candidates[pos_cand_idx][0] : candidates[pos_cand_idx][1]
                ]
                data_samples.append(
                    {
                        "q_text": question_text,
                        "c_text": " ".join(pos_tokens),
                        "label": 1,
                    }
                )

                # Negative sampling (1 negative per positive)
                neg_indices = [i for i in range(len(candidates)) if i != pos_cand_idx]
                if neg_indices:
                    neg_idx = random.choice(neg_indices)
                    neg_tokens = doc_tokens[
                        candidates[neg_idx][0] : candidates[neg_idx][1]
                    ]
                    data_samples.append(
                        {
                            "q_text": question_text,
                            "c_text": " ".join(neg_tokens),
                            "label": 0,
                        }
                    )

            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    return pd.DataFrame(data_samples)


def get_ranker_data(split="train", load_cached_data=True):
    """
    Retrieves ranker training/validation data, using cache if available.
    """
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
        cache_path = Config.RANKER_TRAIN_DATA_PATH
        limit = Config.TRAIN_SAMPLE_SIZE
    else:
        meta_path = Config.VAL_METADATA_PATH
        cache_path = Config.RANKER_VAL_DATA_PATH
        limit = Config.VAL_SAMPLE_SIZE

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading ranker {split} data from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing ranker {split} data from scratch...")
    df = _process_ranker_data_logic(meta_path, Config.TRAIN_FILE, limit)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"Saved ranker {split} data to {cache_path}. Size: {len(df)}")
    return df


def _process_reader_data_logic(metadata_path, raw_file_path, sample_size=None):
    """
    Generates (Question, Context, Start, End) tuples for the reader.
    Only uses examples with short answers.
    """
    metadata = pd.read_csv(metadata_path)

    # Filter for has_short_answer
    train_df = metadata[metadata["has_short_answer"] == True].copy()

    if sample_size and len(train_df) > sample_size:
        train_df = train_df.sample(n=sample_size, random_state=Config.SEED)

    data_samples = []

    with open(raw_file_path, "rb") as f:
        for _, row in train_df.iterrows():
            f.seek(row["byte_offset"])
            line = f.readline()
            if not line:
                continue

            try:
                entry = json.loads(line)
                question_text = entry["question_text"]
                doc_text = entry["document_text"]
                doc_tokens = tokenize_text(doc_text)

                annotation = entry["annotations"][0]

                # Get Long Answer Span (Context)
                la_start = annotation["long_answer"]["start_token"]
                la_end = annotation["long_answer"]["end_token"]

                if la_start == -1:
                    continue

                # Get Short Answer Span
                if len(annotation["short_answers"]) > 0:
                    sa_start = annotation["short_answers"][0]["start_token"]
                    sa_end = annotation["short_answers"][0]["end_token"]
                else:
                    # Skip YES/NO for span extraction training
                    continue

                # Check if short answer is inside long answer
                if not (sa_start >= la_start and sa_end <= la_end):
                    continue

                # Extract Context Tokens
                context_tokens = doc_tokens[la_start:la_end]

                # Calculate relative indices
                rel_start = sa_start - la_start
                rel_end = (
                    sa_end - la_start
                )  # end is exclusive in slicing, but often treated as inclusive or exclusive depending on model.
                # For this dataset, end_token is exclusive in the annotation.
                # We will store exclusive end index relative to context.

                # Sanity check
                if rel_end > len(context_tokens):
                    continue

                data_samples.append(
                    {
                        "q_text": question_text,
                        "c_text": " ".join(context_tokens),
                        "start_idx": rel_start,
                        "end_idx": rel_end,
                    }
                )

            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    return pd.DataFrame(data_samples)


def get_reader_data(split="train", load_cached_data=True):
    """
    Retrieves reader training/validation data, using cache if available.
    """
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
        cache_path = Config.READER_TRAIN_DATA_PATH
        limit = Config.TRAIN_SAMPLE_SIZE
    else:
        meta_path = Config.VAL_METADATA_PATH
        cache_path = Config.READER_VAL_DATA_PATH
        limit = Config.VAL_SAMPLE_SIZE

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading reader {split} data from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing reader {split} data from scratch...")
    df = _process_reader_data_logic(meta_path, Config.TRAIN_FILE, limit)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"Saved reader {split} data to {cache_path}. Size: {len(df)}")
    return df


def get_test_candidates(load_cached_data=True):
    """
    Generates candidates for the test set for inference.
    Returns: DataFrame with (example_id, q_text, c_text, c_idx, token_start, token_end)
    """
    cache_path = Config.TEST_FEATURES_PATH
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading test features from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print("Processing test candidates from scratch...")

    test_files = glob.glob(Config.TEST_FILE_PATTERN)
    if not test_files:
        raise FileNotFoundError("No test file found.")
    test_file_path = test_files[0]

    metadata = pd.read_csv(Config.TEST_METADATA_PATH)

    rows = []

    with open(test_file_path, "rb") as f:
        for _, row in metadata.iterrows():
            f.seek(row["byte_offset"])
            line = f.readline()
            if not line:
                continue

            try:
                entry = json.loads(line)
                ex_id = entry["example_id"]
                q_text = entry["question_text"]
                doc_text = entry["document_text"]
                doc_tokens = tokenize_text(doc_text)

                candidates = parse_html_candidates(doc_tokens)

                for idx, (start, end) in enumerate(candidates):
                    c_tokens = doc_tokens[start:end]
                    # Skip extremely short candidates
                    if len(c_tokens) < 5:
                        continue

                    rows.append(
                        {
                            "example_id": ex_id,
                            "q_text": q_text,
                            "c_text": " ".join(c_tokens),
                            "c_idx": idx,
                            "token_start": start,
                            "token_end": end,
                        }
                    )
            except (json.JSONDecodeError, KeyError):
                continue

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"Saved test features to {cache_path}. Size: {len(df)}")
    return df


class NQRankerDataset(Dataset):
    """
    PyTorch Dataset for the Ranker model.
    """

    def __init__(self, data_df, vocab):
        self.data = data_df
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        q_tokens = tokenize_text(row["q_text"])
        c_tokens = tokenize_text(row["c_text"])

        q_ids = self.vocab.convert_tokens_to_ids(q_tokens, Config.MAX_Q_LEN)
        c_ids = self.vocab.convert_tokens_to_ids(c_tokens, Config.MAX_DOC_LEN)

        return {
            "q_ids": torch.tensor(q_ids, dtype=torch.long),
            "c_ids": torch.tensor(c_ids, dtype=torch.long),
            "label": torch.tensor(row["label"], dtype=torch.float),
        }


class NQReaderDataset(Dataset):
    """
    PyTorch Dataset for the Reader model.
    """

    def __init__(self, data_df, vocab):
        self.data = data_df
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        q_tokens = tokenize_text(row["q_text"])
        c_tokens = tokenize_text(row["c_text"])

        q_ids = self.vocab.convert_tokens_to_ids(q_tokens, Config.MAX_Q_LEN)
        c_ids = self.vocab.convert_tokens_to_ids(c_tokens, Config.MAX_DOC_LEN)

        # Ensure indices are within bounds of max_doc_len
        # The end_idx in our data is exclusive (standard python slice),
        # but for classification we usually want the index of the last token.
        # Let's adjust end_idx to be inclusive for the model prediction target.
        start_idx = row["start_idx"]
        end_idx = row["end_idx"] - 1  # Make inclusive

        # Clamp to max length
        start_idx = max(0, min(start_idx, Config.MAX_DOC_LEN - 1))
        end_idx = max(0, min(end_idx, Config.MAX_DOC_LEN - 1))

        return {
            "q_ids": torch.tensor(q_ids, dtype=torch.long),
            "c_ids": torch.tensor(c_ids, dtype=torch.long),
            "start_idx": torch.tensor(start_idx, dtype=torch.long),
            "end_idx": torch.tensor(end_idx, dtype=torch.long),
        }
