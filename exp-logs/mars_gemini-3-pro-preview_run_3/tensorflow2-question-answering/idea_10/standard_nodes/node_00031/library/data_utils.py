import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter
import glob

# Import Config
from library.config import Config


class HTMLSegmenter:
    """
    Parses raw document text into candidate paragraphs based on HTML tags.
    """

    def __init__(self):
        # Tags that typically denote the start of a candidate long answer
        self.block_tags = {
            "<P>",
            "<Table>",
            "<Tr>",
            "<Ul>",
            "<Ol>",
            "<Dl>",
            "<H1>",
            "<H2>",
            "<H3>",
            "<H4>",
            "<H5>",
            "<H6>",
        }

    def segment(self, document_text):
        """
        Splits document text into paragraphs.
        Returns a list of dicts: {'tokens': list, 'start_token': int, 'end_token': int}
        """
        tokens = document_text.split()
        candidates = []

        current_tokens = []
        current_start = 0

        for i, token in enumerate(tokens):
            # If we hit a block tag and have accumulated tokens, verify if it's a valid block start
            if token in self.block_tags and current_tokens:
                candidates.append(
                    {
                        "tokens": current_tokens,
                        "start_token": current_start,
                        "end_token": current_start + len(current_tokens),
                    }
                )
                current_tokens = []
                current_start = i

            current_tokens.append(token)

        # Append last block
        if current_tokens:
            candidates.append(
                {
                    "tokens": current_tokens,
                    "start_token": current_start,
                    "end_token": current_start + len(current_tokens),
                }
            )

        return candidates


class Vocabulary:
    """
    Handles token-to-index mapping.
    """

    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}

    def build(self, texts, vocab_size):
        """
        Builds vocabulary from a list of text strings (space-separated).
        """
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        # Special tokens
        self.token2idx = {Config.PAD_TOKEN: 0, Config.UNK_TOKEN: 1, Config.SEP_TOKEN: 2}

        # Most common tokens
        most_common = counter.most_common(vocab_size - len(self.token2idx))
        for token, _ in most_common:
            self.token2idx[token] = len(self.token2idx)

        self.idx2token = {v: k for k, v in self.token2idx.items()}

    def encode(self, tokens, max_len=None):
        """
        Converts list of tokens to list of indices.
        """
        indices = [
            self.token2idx.get(t, self.token2idx[Config.UNK_TOKEN]) for t in tokens
        ]

        if max_len is not None:
            if len(indices) > max_len:
                indices = indices[:max_len]
            else:
                indices += [self.token2idx[Config.PAD_TOKEN]] * (max_len - len(indices))

        return indices

    def save(self, path):
        # Save as parquet for consistency with requirements
        df = pd.DataFrame(list(self.token2idx.items()), columns=["token", "index"])
        df.to_parquet(path, index=False)

    def load(self, path):
        df = pd.read_parquet(path)
        self.token2idx = dict(zip(df["token"], df["index"]))
        self.idx2token = {v: k for k, v in self.token2idx.items()}


def load_embeddings(vocab, embedding_dim, load_cached_data=True):
    """
    Creates or loads an embedding matrix.
    Since we don't have external GloVe files guaranteed, we initialize randomly.
    """
    if load_cached_data and os.path.exists(Config.EMBEDDING_MATRIX_PATH):
        return np.load(Config.EMBEDDING_MATRIX_PATH)

    vocab_size = len(vocab.token2idx)
    # Random initialization using Xavier/Glorot uniform
    scale = np.sqrt(3.0 / embedding_dim)
    embedding_matrix = np.random.uniform(
        -scale, scale, (vocab_size, embedding_dim)
    ).astype(np.float32)

    # Zero out padding
    if Config.PAD_TOKEN in vocab.token2idx:
        pad_idx = vocab.token2idx[Config.PAD_TOKEN]
        embedding_matrix[pad_idx] = 0.0

    np.save(Config.EMBEDDING_MATRIX_PATH, embedding_matrix)
    return embedding_matrix


class RankerDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_parquet(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Data is stored as lists in parquet, convert to numpy then torch
        input_ids = np.array(row["input_ids"], dtype=np.int64)

        item = {"input_ids": torch.from_numpy(input_ids)}

        if "label" in row:
            item["label"] = torch.tensor(np.float32(row["label"]))

        if "example_id" in row:
            item["example_id"] = row["example_id"]

        if "candidate_idx" in row:
            item["candidate_idx"] = row["candidate_idx"]

        return item


class ReaderDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_parquet(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        q_input_ids = np.array(row["q_input_ids"], dtype=np.int64)
        ctx_input_ids = np.array(row["ctx_input_ids"], dtype=np.int64)

        item = {
            "q_input_ids": torch.from_numpy(q_input_ids),
            "ctx_input_ids": torch.from_numpy(ctx_input_ids),
        }

        if "start_token" in row:
            item["start_idx"] = torch.tensor(np.int64(row["start_token"]))
            item["end_idx"] = torch.tensor(np.int64(row["end_token"]))

        if "example_id" in row:
            item["example_id"] = row["example_id"]

        if "candidate_idx" in row:
            item["candidate_idx"] = row["candidate_idx"]

        return item


class DataProcessor:
    def __init__(self):
        self.segmenter = HTMLSegmenter()
        self.vocab = Vocabulary()

    def _read_jsonl_sample(self, metadata_path, jsonl_file, sample_size=None):
        """Helper to read samples based on metadata."""
        if not os.path.exists(metadata_path):
            print(f"Metadata file not found: {metadata_path}")
            return []

        df = pd.read_csv(metadata_path)
        if sample_size is not None and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=Config.SEED)

        samples = []
        if not os.path.exists(jsonl_file):
            print(f"Data file not found: {jsonl_file}")
            return []

        with open(jsonl_file, "rb") as f:
            for _, row in df.iterrows():
                f.seek(row["byte_offset"])
                line = f.readline()
                if line:
                    try:
                        samples.append(json.loads(line.decode("utf-8")))
                    except json.JSONDecodeError:
                        continue
        return samples

    def build_vocab(self, load_cached_data=True):
        if load_cached_data and os.path.exists(Config.VOCAB_PATH):
            self.vocab.load(Config.VOCAB_PATH)
            return

        print("Building vocabulary...")
        # Load a subset of training data to build vocab
        samples = self._read_jsonl_sample(
            Config.TRAIN_METADATA_PATH, Config.TRAIN_FILE, sample_size=20000
        )

        texts = []
        for s in samples:
            texts.append(s["question_text"])
            texts.append(s["document_text"])

        self.vocab.build(texts, Config.VOCAB_SIZE)
        self.vocab.save(Config.VOCAB_PATH)
        print(f"Vocabulary built with {len(self.vocab.token2idx)} tokens.")

    def _process_ranker_samples(self, samples, is_train=True):
        data_rows = []

        for s in samples:
            example_id = s["example_id"]
            question_tokens = s["question_text"].split()
            candidates = self.segmenter.segment(s["document_text"])

            if not candidates:
                continue

            # Ground truth for training
            positive_indices = set()
            if is_train:
                annotations = s.get("annotations", [])
                for ann in annotations:
                    la = ann["long_answer"]
                    if la["start_token"] != -1:
                        # Find which candidate matches
                        for i, cand in enumerate(candidates):
                            if (
                                cand["start_token"] == la["start_token"]
                                and cand["end_token"] == la["end_token"]
                            ):
                                positive_indices.add(i)

            # Select samples
            selected_indices = []
            labels = []

            if is_train:
                if positive_indices:
                    # Add positives
                    for idx in positive_indices:
                        selected_indices.append(idx)
                        labels.append(1.0)

                    # Negative sampling: use simple token overlap
                    q_set = set(question_tokens)
                    scores = []
                    for i, cand in enumerate(candidates):
                        if i in positive_indices:
                            scores.append(-1)
                            continue
                        overlap = len(q_set.intersection(cand["tokens"]))
                        scores.append(overlap)

                    # Pick top 2 hard negatives (highest overlap that isn't answer)
                    neg_indices = np.argsort(scores)[-2:]
                    for idx in neg_indices:
                        if scores[idx] != -1 and idx < len(candidates):
                            selected_indices.append(idx)
                            labels.append(0.0)
            else:
                # Test mode: take all candidates
                selected_indices = range(len(candidates))
                labels = [-1.0] * len(candidates)

            # Tokenize and Build Features
            for i, cand_idx in enumerate(selected_indices):
                cand = candidates[cand_idx]

                # [Q; SEP; Para]
                q_ids = self.vocab.encode(question_tokens, Config.MAX_QUESTION_LEN)
                p_ids = self.vocab.encode(cand["tokens"], Config.MAX_PARAGRAPH_LEN)
                sep_id = self.vocab.token2idx[Config.SEP_TOKEN]

                combined_ids = q_ids + [sep_id] + p_ids
                # Pad/Truncate to fixed length
                if len(combined_ids) < Config.MAX_RANKER_SEQ_LEN:
                    combined_ids += [self.vocab.token2idx[Config.PAD_TOKEN]] * (
                        Config.MAX_RANKER_SEQ_LEN - len(combined_ids)
                    )
                else:
                    combined_ids = combined_ids[: Config.MAX_RANKER_SEQ_LEN]

                data_rows.append(
                    {
                        "example_id": example_id,
                        "candidate_idx": cand_idx,
                        "input_ids": combined_ids,
                        "label": labels[i],
                    }
                )

        return pd.DataFrame(data_rows)

    def process_ranker_data(self, load_cached_data=True):
        """Generates train/val/test data for Ranker."""
        Config.ensure_directories()

        # Train
        if not (load_cached_data and os.path.exists(Config.RANKER_TRAIN_DATA_PATH)):
            print("Processing Ranker Train Data...")
            samples = self._read_jsonl_sample(
                Config.TRAIN_METADATA_PATH, Config.TRAIN_FILE, Config.SAMPLE_SIZE
            )
            df = self._process_ranker_samples(samples, is_train=True)
            df.to_parquet(Config.RANKER_TRAIN_DATA_PATH)

        # Val
        if not (load_cached_data and os.path.exists(Config.RANKER_VAL_DATA_PATH)):
            print("Processing Ranker Val Data...")
            val_sample = Config.SAMPLE_SIZE // 5 if Config.SAMPLE_SIZE else None
            samples = self._read_jsonl_sample(
                Config.VAL_METADATA_PATH, Config.TRAIN_FILE, val_sample
            )
            df = self._process_ranker_samples(samples, is_train=True)
            df.to_parquet(Config.RANKER_VAL_DATA_PATH)

        # Test features (for inference)
        if not (load_cached_data and os.path.exists(Config.RANKER_TEST_FEATURES_PATH)):
            print("Processing Ranker Test Data...")
            test_files = glob.glob(Config.TEST_FILE_PATTERN)
            if test_files:
                test_file = test_files[0]
                # Process full test set (sample_size=None)
                samples = self._read_jsonl_sample(
                    Config.TEST_METADATA_PATH, test_file, None
                )
                df = self._process_ranker_samples(samples, is_train=False)
                df.to_parquet(Config.RANKER_TEST_FEATURES_PATH)

    def _process_reader_samples(self, samples):
        data_rows = []

        for s in samples:
            example_id = s["example_id"]
            question_tokens = s["question_text"].split()
            candidates = self.segmenter.segment(s["document_text"])

            annotations = s.get("annotations", [])
            for ann in annotations:
                short_answers = ann.get("short_answers", [])
                if not short_answers:
                    continue

                # Use the first short answer span
                sa = short_answers[0]
                s_start = sa["start_token"]
                s_end = sa["end_token"]

                # Find containing paragraph
                target_cand = None
                target_cand_idx = -1

                for i, cand in enumerate(candidates):
                    if cand["start_token"] <= s_start and cand["end_token"] >= s_end:
                        target_cand = cand
                        target_cand_idx = i
                        break

                if target_cand:
                    # Calculate relative indices
                    rel_start = s_start - target_cand["start_token"]
                    rel_end = s_end - target_cand["start_token"]

                    # Ignore if span is too long or outside max paragraph len
                    if rel_end >= Config.MAX_PARAGRAPH_LEN:
                        continue

                    q_ids = self.vocab.encode(question_tokens, Config.MAX_QUESTION_LEN)
                    ctx_ids = self.vocab.encode(
                        target_cand["tokens"], Config.MAX_PARAGRAPH_LEN
                    )

                    data_rows.append(
                        {
                            "example_id": example_id,
                            "candidate_idx": target_cand_idx,
                            "q_input_ids": q_ids,
                            "ctx_input_ids": ctx_ids,
                            "start_token": rel_start,
                            "end_token": rel_end,
                        }
                    )

        return pd.DataFrame(data_rows)

    def process_reader_data(self, load_cached_data=True):
        """Generates train/val data for Reader."""
        Config.ensure_directories()

        if not (load_cached_data and os.path.exists(Config.READER_TRAIN_DATA_PATH)):
            print("Processing Reader Train Data...")
            samples = self._read_jsonl_sample(
                Config.TRAIN_METADATA_PATH, Config.TRAIN_FILE, Config.SAMPLE_SIZE
            )
            df = self._process_reader_samples(samples)
            df.to_parquet(Config.READER_TRAIN_DATA_PATH)

        if not (load_cached_data and os.path.exists(Config.READER_VAL_DATA_PATH)):
            print("Processing Reader Val Data...")
            val_sample = Config.SAMPLE_SIZE // 5 if Config.SAMPLE_SIZE else None
            samples = self._read_jsonl_sample(
                Config.VAL_METADATA_PATH, Config.TRAIN_FILE, val_sample
            )
            df = self._process_reader_samples(samples)
            df.to_parquet(Config.READER_VAL_DATA_PATH)


def prepare_data(load_cached_data=True):
    """
    Main entry point to prepare all data.
    """
    processor = DataProcessor()

    # 1. Build/Load Vocab
    processor.build_vocab(load_cached_data)

    # 2. Embeddings
    load_embeddings(processor.vocab, Config.EMBEDDING_DIM, load_cached_data)

    # 3. Ranker Data
    processor.process_ranker_data(load_cached_data)

    # 4. Reader Data
    processor.process_reader_data(load_cached_data)

    print("Data preparation complete.")


def get_ranker_datasets(load_cached_data=True):
    """Returns train and val datasets for Ranker."""
    prepare_data(load_cached_data)
    return (
        RankerDataset(Config.RANKER_TRAIN_DATA_PATH),
        RankerDataset(Config.RANKER_VAL_DATA_PATH),
    )


def get_reader_datasets(load_cached_data=True):
    """Returns train and val datasets for Reader."""
    prepare_data(load_cached_data)
    return (
        ReaderDataset(Config.READER_TRAIN_DATA_PATH),
        ReaderDataset(Config.READER_VAL_DATA_PATH),
    )


def get_test_dataset(load_cached_data=True):
    """Returns ranker features for test set."""
    prepare_data(load_cached_data)
    return RankerDataset(Config.RANKER_TEST_FEATURES_PATH)
