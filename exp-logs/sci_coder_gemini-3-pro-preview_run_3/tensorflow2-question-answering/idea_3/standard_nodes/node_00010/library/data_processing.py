import os
import json
import re
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from library.config import Config

# Set fixed seeds for reproducibility
random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


class TextProcessor:
    """Handles text cleaning, tokenization, and span alignment."""

    def __init__(self):
        self.html_tag_pattern = re.compile(r"^<.*>$")

    def is_html_tag(self, token):
        return bool(self.html_tag_pattern.match(token))

    def clean_and_map_indices(self, tokens):
        """
        Removes HTML tags from a list of tokens and returns the cleaned tokens
        along with a mapping from old indices to new indices.
        """
        cleaned_tokens = []
        old_to_new_map = {}
        new_idx = 0

        for old_idx, token in enumerate(tokens):
            if not self.is_html_tag(token):
                cleaned_tokens.append(token)
                old_to_new_map[old_idx] = new_idx
                new_idx += 1
            else:
                # Map tag indices to the next valid token index (or len if at end)
                old_to_new_map[old_idx] = new_idx

        return cleaned_tokens, old_to_new_map

    def tokenize(self, text):
        return text.split()


class Vocabulary:
    """Manages token-to-index mapping."""

    def __init__(self):
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.freqs = Counter()
        self.built = False

    def build(self, texts, max_size=Config.VOCAB_SIZE, min_freq=Config.MIN_FREQ):
        print("Building vocabulary...")
        for text in texts:
            self.freqs.update(text)

        # Add special tokens first
        self.token_to_idx = {Config.PAD_TOKEN: 0, Config.UNK_TOKEN: 1}
        self.idx_to_token = {0: Config.PAD_TOKEN, 1: Config.UNK_TOKEN}
        idx = 2

        # Add most common tokens
        for token, freq in self.freqs.most_common(max_size - 2):
            if freq >= min_freq:
                self.token_to_idx[token] = idx
                self.idx_to_token[idx] = token
                idx += 1

        self.built = True
        print(f"Vocabulary built with {len(self.token_to_idx)} tokens.")

    def encode(self, tokens):
        return [
            self.token_to_idx.get(t, self.token_to_idx[Config.UNK_TOKEN])
            for t in tokens
        ]

    def save(self, path):
        # Save as parquet: token, index
        data = [{"token": t, "index": i} for t, i in self.token_to_idx.items()]
        df = pd.DataFrame(data)
        df.to_parquet(path, index=False)
        print(f"Vocabulary saved to {path}")

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocab file not found at {path}")
        df = pd.read_parquet(path)
        self.token_to_idx = dict(zip(df["token"], df["index"]))
        self.idx_to_token = dict(zip(df["index"], df["token"]))
        self.built = True
        print(f"Vocabulary loaded from {path}")

    def __len__(self):
        return len(self.token_to_idx)


class RankerDataset(Dataset):
    def __init__(self, data, vocab):
        """
        data: list of dicts with keys 'q_tokens', 'ctx_tokens', 'label'
        """
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        q_indices = self.vocab.encode(item["q_tokens"])
        ctx_indices = self.vocab.encode(item["ctx_tokens"])

        # Truncation
        q_indices = q_indices[: Config.MAX_Q_LEN]
        ctx_indices = ctx_indices[: Config.MAX_CTX_LEN]

        # Padding
        q_pad_len = Config.MAX_Q_LEN - len(q_indices)
        ctx_pad_len = Config.MAX_CTX_LEN - len(ctx_indices)

        q_indices += [self.vocab.token_to_idx[Config.PAD_TOKEN]] * q_pad_len
        ctx_indices += [self.vocab.token_to_idx[Config.PAD_TOKEN]] * ctx_pad_len

        return {
            "q_input": torch.tensor(q_indices, dtype=torch.long),
            "ctx_input": torch.tensor(ctx_indices, dtype=torch.long),
            "label": torch.tensor(item["label"], dtype=torch.float),
        }


class ReaderDataset(Dataset):
    def __init__(self, data, vocab):
        """
        data: list of dicts with keys 'q_tokens', 'ctx_tokens', 'start_token', 'end_token'
        """
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        q_indices = self.vocab.encode(item["q_tokens"])
        ctx_indices = self.vocab.encode(item["ctx_tokens"])

        # Truncation
        q_indices = q_indices[: Config.MAX_Q_LEN]
        ctx_indices = ctx_indices[: Config.MAX_CTX_LEN]

        # Adjust targets if they fall outside truncated context
        start = item["start_token"]
        end = item["end_token"]

        if start >= Config.MAX_CTX_LEN:
            start = 0
            end = 0
        elif end >= Config.MAX_CTX_LEN:
            end = Config.MAX_CTX_LEN - 1

        # Padding
        q_pad_len = Config.MAX_Q_LEN - len(q_indices)
        ctx_pad_len = Config.MAX_CTX_LEN - len(ctx_indices)

        q_indices += [self.vocab.token_to_idx[Config.PAD_TOKEN]] * q_pad_len
        ctx_indices += [self.vocab.token_to_idx[Config.PAD_TOKEN]] * ctx_pad_len

        return {
            "q_input": torch.tensor(q_indices, dtype=torch.long),
            "ctx_input": torch.tensor(ctx_indices, dtype=torch.long),
            "start_target": torch.tensor(start, dtype=torch.long),
            "end_target": torch.tensor(end, dtype=torch.long),
        }


class DataProcessor:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.vocab = Vocabulary()

    def _read_jsonl_sample(self, file_path, metadata_df, sample_size=None):
        """Reads specific lines from JSONL based on metadata."""
        if sample_size is not None and len(metadata_df) > sample_size:
            metadata_df = metadata_df.sample(n=sample_size, random_state=Config.SEED)

        samples = []
        with open(file_path, "rb") as f:
            for _, row in metadata_df.iterrows():
                f.seek(row["byte_offset"])
                line = f.readline()
                if line:
                    try:
                        samples.append(json.loads(line.decode("utf-8")))
                    except json.JSONDecodeError:
                        continue
        return samples

    def _process_ranker_data(self, raw_data, is_train=True):
        """Generates positive and negative pairs for the ranker."""
        processed_data = []

        for entry in raw_data:
            doc_text = entry["document_text"]
            doc_tokens = doc_text.split()
            question_tokens = self.text_processor.tokenize(entry["question_text"])

            candidates = entry["long_answer_candidates"]
            annotations = entry.get("annotations", [])

            # Identify ground truth long answer index
            correct_candidate_idx = -1
            if is_train:
                for ann in annotations:
                    la = ann["long_answer"]
                    if la["start_token"] != -1:
                        # Find which candidate matches this span
                        for i, cand in enumerate(candidates):
                            if (
                                cand["start_token"] == la["start_token"]
                                and cand["end_token"] == la["end_token"]
                            ):
                                correct_candidate_idx = i
                                break
                        if correct_candidate_idx != -1:
                            break

            # Extract text for candidates
            candidate_texts = []
            for cand in candidates:
                # Extract raw span
                span_tokens = doc_tokens[cand["start_token"] : cand["end_token"]]
                # Clean HTML
                clean_tokens, _ = self.text_processor.clean_and_map_indices(span_tokens)
                if clean_tokens:  # Skip empty
                    candidate_texts.append(clean_tokens)
                else:
                    candidate_texts.append([Config.UNK_TOKEN])

            if is_train:
                if correct_candidate_idx != -1:
                    # Positive Sample
                    processed_data.append(
                        {
                            "q_tokens": question_tokens,
                            "ctx_tokens": candidate_texts[correct_candidate_idx],
                            "label": 1.0,
                        }
                    )

                    # Negative Sample (Randomly select one non-correct candidate)
                    indices = list(range(len(candidates)))
                    indices.remove(correct_candidate_idx)
                    if indices:
                        neg_idx = random.choice(indices)
                        processed_data.append(
                            {
                                "q_tokens": question_tokens,
                                "ctx_tokens": candidate_texts[neg_idx],
                                "label": 0.0,
                            }
                        )
            else:
                # For inference/test, we might process differently, but here we define dataset structure
                # This function is primarily for building training/val datasets for the models
                pass

        return processed_data

    def _process_reader_data(self, raw_data):
        """Generates data for the reader (Question + Paragraph -> Span)."""
        processed_data = []

        for entry in raw_data:
            doc_text = entry["document_text"]
            doc_tokens = doc_text.split()
            question_tokens = self.text_processor.tokenize(entry["question_text"])

            candidates = entry["long_answer_candidates"]
            annotations = entry.get("annotations", [])

            for ann in annotations:
                # We need a valid short answer AND a valid long answer containing it
                la = ann["long_answer"]
                sa_list = ann["short_answers"]

                if la["start_token"] != -1 and len(sa_list) > 0:
                    # Get the containing candidate
                    # (In NQ, the long answer annotation corresponds to one of the candidates)
                    la_start = la["start_token"]
                    la_end = la["end_token"]

                    # Get raw tokens of the paragraph
                    raw_para_tokens = doc_tokens[la_start:la_end]

                    # Clean and get mapping
                    clean_para_tokens, idx_map = (
                        self.text_processor.clean_and_map_indices(raw_para_tokens)
                    )

                    # Process first short answer (simplification)
                    sa = sa_list[0]
                    sa_start_global = sa["start_token"]
                    sa_end_global = sa["end_token"]

                    # Convert global indices to paragraph-relative indices
                    rel_start = sa_start_global - la_start
                    rel_end = (
                        sa_end_global - la_start
                    )  # end token is exclusive in NQ, but we usually predict inclusive or exclusive.
                    # NQ: [start, end). Let's stick to Python slicing semantics or convert to inclusive.
                    # Standard BERT/LSTM span prediction usually predicts Start and End (Inclusive).
                    # Let's treat end as inclusive for the model target.
                    rel_end_inclusive = rel_end - 1

                    # Map to cleaned indices
                    # We need to be careful if the answer boundary is a tag (unlikely but possible)
                    if rel_start in idx_map and rel_end_inclusive in idx_map:
                        mapped_start = idx_map[rel_start]
                        mapped_end = idx_map[rel_end_inclusive]

                        # Sanity check
                        if mapped_start <= mapped_end and mapped_end < len(
                            clean_para_tokens
                        ):
                            processed_data.append(
                                {
                                    "q_tokens": question_tokens,
                                    "ctx_tokens": clean_para_tokens,
                                    "start_token": mapped_start,
                                    "end_token": mapped_end,
                                }
                            )

        return processed_data

    def get_data(self, load_cached_data=True):
        """
        Orchestrates the data loading, processing, and caching.
        Returns ranker_train, ranker_val, reader_train, reader_val datasets (as lists/dicts).
        """

        # 1. Vocabulary
        if load_cached_data and os.path.exists(Config.VOCAB_CACHE):
            self.vocab.load(Config.VOCAB_CACHE)
        else:
            # Build vocab from training sample
            print("Building vocabulary from scratch...")
            train_meta = pd.read_csv(Config.TRAIN_METADATA)
            raw_train = self._read_jsonl_sample(
                Config.TRAIN_FILE, train_meta, Config.TRAIN_SAMPLE_SIZE
            )

            # Collect text for vocab
            texts = []
            for item in raw_train:
                texts.append(self.text_processor.tokenize(item["question_text"]))
                # Sample some doc text
                doc_tokens = item["document_text"].split()
                # Clean a portion of it to get representative words
                clean_doc, _ = self.text_processor.clean_and_map_indices(
                    doc_tokens[:2000]
                )
                texts.append(clean_doc)

            self.vocab.build(texts)
            self.vocab.save(Config.VOCAB_CACHE)

        # 2. Ranker Data
        # Train
        if load_cached_data and os.path.exists(Config.RANKER_TRAIN_CACHE):
            print("Loading cached Ranker Train data...")
            ranker_train_df = pd.read_parquet(Config.RANKER_TRAIN_CACHE)
            ranker_train_data = ranker_train_df.to_dict("records")
        else:
            print("Processing Ranker Train data...")
            train_meta = pd.read_csv(Config.TRAIN_METADATA)
            # Filter for efficiency if needed, but here we use sample size from config
            raw_train = self._read_jsonl_sample(
                Config.TRAIN_FILE, train_meta, Config.TRAIN_SAMPLE_SIZE
            )
            ranker_train_data = self._process_ranker_data(raw_train, is_train=True)
            # Cache
            pd.DataFrame(ranker_train_data).to_parquet(Config.RANKER_TRAIN_CACHE)

        # Val
        if load_cached_data and os.path.exists(Config.RANKER_VAL_CACHE):
            print("Loading cached Ranker Val data...")
            ranker_val_df = pd.read_parquet(Config.RANKER_VAL_CACHE)
            ranker_val_data = ranker_val_df.to_dict("records")
        else:
            print("Processing Ranker Val data...")
            val_meta = pd.read_csv(Config.VAL_METADATA)
            raw_val = self._read_jsonl_sample(
                Config.TRAIN_FILE, val_meta, Config.VAL_SAMPLE_SIZE
            )
            ranker_val_data = self._process_ranker_data(raw_val, is_train=True)
            pd.DataFrame(ranker_val_data).to_parquet(Config.RANKER_VAL_CACHE)

        # 3. Reader Data
        # Train
        if load_cached_data and os.path.exists(Config.READER_TRAIN_CACHE):
            print("Loading cached Reader Train data...")
            reader_train_df = pd.read_parquet(Config.READER_TRAIN_CACHE)
            reader_train_data = reader_train_df.to_dict("records")
        else:
            print("Processing Reader Train data...")
            # Re-use raw_train if available, else reload
            if "raw_train" not in locals():
                train_meta = pd.read_csv(Config.TRAIN_METADATA)
                raw_train = self._read_jsonl_sample(
                    Config.TRAIN_FILE, train_meta, Config.TRAIN_SAMPLE_SIZE
                )
            reader_train_data = self._process_reader_data(raw_train)
            pd.DataFrame(reader_train_data).to_parquet(Config.READER_TRAIN_CACHE)

        # Val
        if load_cached_data and os.path.exists(Config.READER_VAL_CACHE):
            print("Loading cached Reader Val data...")
            reader_val_df = pd.read_parquet(Config.READER_VAL_CACHE)
            reader_val_data = reader_val_df.to_dict("records")
        else:
            print("Processing Reader Val data...")
            if "raw_val" not in locals():
                val_meta = pd.read_csv(Config.VAL_METADATA)
                raw_val = self._read_jsonl_sample(
                    Config.TRAIN_FILE, val_meta, Config.VAL_SAMPLE_SIZE
                )
            reader_val_data = self._process_reader_data(raw_val)
            pd.DataFrame(reader_val_data).to_parquet(Config.READER_VAL_CACHE)

        return ranker_train_data, ranker_val_data, reader_train_data, reader_val_data


def get_data_loaders(load_cached_data=True):
    """
    Factory function to return DataLoaders for training.
    """
    processor = DataProcessor()

    # Ensure working dir exists (redundant with Config but safe)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    r_train, r_val, read_train, read_val = processor.get_data(
        load_cached_data=load_cached_data
    )

    ranker_train_ds = RankerDataset(r_train, processor.vocab)
    ranker_val_ds = RankerDataset(r_val, processor.vocab)

    reader_train_ds = ReaderDataset(read_train, processor.vocab)
    reader_val_ds = ReaderDataset(read_val, processor.vocab)

    loaders = {
        "ranker_train": DataLoader(
            ranker_train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        ),
        "ranker_val": DataLoader(
            ranker_val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        ),
        "reader_train": DataLoader(
            reader_train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        ),
        "reader_val": DataLoader(
            reader_val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        ),
        "vocab": processor.vocab,
    }

    return loaders


def get_test_data_processor():
    """Returns an initialized processor with loaded vocab for inference."""
    processor = DataProcessor()
    if os.path.exists(Config.VOCAB_CACHE):
        processor.vocab.load(Config.VOCAB_CACHE)
    else:
        # Fallback if no vocab exists (should not happen in inference if trained)
        print("Warning: Vocabulary not found for inference. Initializing empty.")
    return processor
