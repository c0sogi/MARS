import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.data_utils import build_vocabularies
from library.features import generate_and_cache_features


class TaggerDataset(Dataset):
    """
    Dataset for the Morphologically-Enhanced Bi-LSTM Tagger.
    Groups tokens by sentence_id to provide context.
    """

    def __init__(self, dataset_type="train", load_cached_data=True, limit=None):
        self.dataset_type = dataset_type
        self.limit = limit

        # Determine paths
        if dataset_type == "train":
            self.data_path = Config.TRAIN_DATA
        elif dataset_type == "val":
            self.data_path = Config.VAL_DATA
        elif dataset_type == "test":
            self.data_path = Config.TEST_DATA
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        self.cache_path = os.path.join(
            Config.WORKING_DIR, f"tagger_processed_{dataset_type}.pt"
        )

        # Load vocabularies
        self.vocab_words, self.vocab_chars, self.vocab_classes = build_vocabularies(
            load_cached_data=load_cached_data
        )

        # Load or Process Data
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading processed {dataset_type} dataset from {self.cache_path}...")
            data = torch.load(self.cache_path)
            self.samples = data["samples"]
            self.word_indices = data["word_indices"]
            self.char_indices_list = data["char_indices_list"]
            self.explicit_features = data["explicit_features"]
            self.class_indices = data["class_indices"]
        else:
            self._process_and_cache(load_cached_data)

        # Apply limit for debugging if requested
        if self.limit:
            self.samples = self.samples[: self.limit]
            print(f"Limiting {dataset_type} dataset to {self.limit} sentences.")

    def _process_and_cache(self, load_cached_data):
        print(f"Processing {self.dataset_type} dataset from scratch...")

        # 1. Load CSV
        df = pd.read_csv(self.data_path, dtype=str, keep_default_na=False)

        # 2. Load Explicit Features
        # These correspond 1:1 with rows in the CSV
        features_array = generate_and_cache_features(
            self.dataset_type, load_cached_data=load_cached_data
        )
        self.explicit_features = torch.tensor(features_array, dtype=torch.float32)

        # 3. Encode Words and Classes
        print("Encoding tokens and classes...")
        # Vectorized lookup is hard with dicts, using list comprehension
        # Words
        unk_idx = self.vocab_words["<UNK>"]
        self.word_indices = torch.tensor(
            [self.vocab_words.stoi.get(w, unk_idx) for w in df["before"]],
            dtype=torch.long,
        )

        # Classes (if available)
        if "class" in df.columns:
            # For train/val
            self.class_indices = torch.tensor(
                [self.vocab_classes[c] for c in df["class"]], dtype=torch.long
            )
        else:
            # For test, fill with 0 (PAD) or dummy
            self.class_indices = torch.zeros(len(df), dtype=torch.long)

        # 4. Encode Characters
        # This results in a list of tensors/arrays because lengths vary
        print("Encoding characters...")
        unk_char = self.vocab_chars["<UNK>"]
        # Pre-fetch stoi for speed
        char_stoi = self.vocab_chars.stoi

        def encode_chars(text):
            return [char_stoi.get(c, unk_char) for c in str(text)]

        # We store as a list of lists (or arrays) to save memory compared to padded tensor
        self.char_indices_list = [encode_chars(t) for t in df["before"]]

        # 5. Group by Sentence
        print("Grouping by sentence_id...")
        # We assume the CSV is sorted or we group it.
        # Using pandas groupby to get indices is robust.
        # We only need the indices into the flat arrays.
        grouped = df.groupby("sentence_id", sort=False).indices
        # grouped is a dict {sentence_id: array_of_indices}
        # We convert values to a list of tensors for the dataset samples
        self.samples = [
            torch.tensor(idxs, dtype=torch.long) for idxs in grouped.values()
        ]

        # 6. Cache
        print(f"Saving processed dataset to {self.cache_path}...")
        torch.save(
            {
                "samples": self.samples,
                "word_indices": self.word_indices,
                "char_indices_list": self.char_indices_list,
                "explicit_features": self.explicit_features,
                "class_indices": self.class_indices,
            },
            self.cache_path,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get row indices for this sentence
        row_idxs = self.samples[idx]

        # Retrieve data using row indices
        word_idx = self.word_indices[row_idxs]
        explicit_feat = self.explicit_features[row_idxs]
        class_idx = self.class_indices[row_idxs]

        # Retrieve and pad char indices for this sentence
        # Each token has a list of chars. We need to pad them to the max word length in this sentence.
        raw_chars = [
            torch.tensor(self.char_indices_list[i], dtype=torch.long) for i in row_idxs
        ]
        if len(raw_chars) > 0:
            char_indices = pad_sequence(
                raw_chars, batch_first=True, padding_value=self.vocab_chars["<PAD>"]
            )
        else:
            # Edge case: empty sentence? Should not happen.
            char_indices = torch.zeros((len(row_idxs), 0), dtype=torch.long)

        return word_idx, char_indices, explicit_feat, class_idx


def collate_fn_tagger(batch):
    """
    Collates a batch of sentences for the Tagger.
    Handles 2D padding for character indices (Batch, Seq, Char).
    """
    word_idxs, char_idxs_list, explicit_feats, class_idxs = zip(*batch)

    # 1. Pad Sentence Sequences (Batch, Seq)
    # word_idxs is list of (Seq,)
    padded_words = pad_sequence(word_idxs, batch_first=True, padding_value=0)  # PAD=0
    padded_classes = pad_sequence(class_idxs, batch_first=True, padding_value=0)
    padded_features = pad_sequence(explicit_feats, batch_first=True, padding_value=0.0)

    # 2. Pad Character Sequences (Batch, Seq, Char)
    # char_idxs_list is list of (Seq, Char_Len), where Char_Len varies per sentence
    batch_size = len(batch)
    max_seq_len = padded_words.size(1)

    # Find max char length in the entire batch
    max_char_len = 0
    for c in char_idxs_list:
        if c.size(1) > max_char_len:
            max_char_len = c.size(1)

    # Create output tensor
    padded_chars = torch.zeros(
        (batch_size, max_seq_len, max_char_len), dtype=torch.long
    )

    for i, chars in enumerate(char_idxs_list):
        seq_len = chars.size(0)
        char_len = chars.size(1)
        # Copy data
        padded_chars[i, :seq_len, :char_len] = chars

    return padded_words, padded_chars, padded_features, padded_classes


class FallbackDataset(Dataset):
    """
    Dataset for the LSTM Seq2Seq Fallback Model.
    Filters for tokens where before != after.
    """

    def __init__(self, dataset_type="train", load_cached_data=True, limit=None):
        self.dataset_type = dataset_type
        self.limit = limit

        if dataset_type == "test":
            # Test set doesn't have targets, so we can't filter by change.
            # Usually Fallback is used during inference on specific tokens.
            # This dataset class is primarily for training/validation.
            # If needed for test, we would load all tokens.
            # However, for this task, we assume this is for training the fallback.
            pass

        if dataset_type == "train":
            self.data_path = Config.TRAIN_DATA
        elif dataset_type == "val":
            self.data_path = Config.VAL_DATA
        else:
            # For test, we generally don't use this dataset class directly in this pipeline
            # as inference is done token-by-token based on Tagger output.
            self.data_path = Config.TEST_DATA

        self.cache_path = os.path.join(
            Config.WORKING_DIR, f"fallback_processed_{dataset_type}.pt"
        )

        self.vocab_words, self.vocab_chars, self.vocab_classes = build_vocabularies(
            load_cached_data=load_cached_data
        )

        if load_cached_data and os.path.exists(self.cache_path):
            print(
                f"Loading processed {dataset_type} fallback data from {self.cache_path}..."
            )
            data = torch.load(self.cache_path)
            self.src_indices = data["src_indices"]
            self.tgt_indices = data["tgt_indices"]
            self.class_indices = data["class_indices"]
        else:
            self._process_and_cache()

        if self.limit:
            self.src_indices = self.src_indices[: self.limit]
            self.tgt_indices = self.tgt_indices[: self.limit]
            self.class_indices = self.class_indices[: self.limit]
            print(f"Limiting {dataset_type} fallback dataset to {self.limit} samples.")

    def _process_and_cache(self):
        print(f"Processing {self.dataset_type} fallback data from scratch...")

        df = pd.read_csv(self.data_path, dtype=str, keep_default_na=False)

        # Filter for changes
        if "after" in df.columns:
            print("Filtering for changed tokens (before != after)...")
            df = df[df["before"] != df["after"]].copy()

        print(f"Found {len(df)} samples for fallback training.")

        # Encode
        char_stoi = self.vocab_chars.stoi
        unk_char = self.vocab_chars["<UNK>"]
        sos_char = self.vocab_chars["<SOS>"]
        eos_char = self.vocab_chars["<EOS>"]

        def encode(text, add_special=False):
            indices = [char_stoi.get(c, unk_char) for c in str(text)]
            if add_special:
                indices = [sos_char] + indices + [eos_char]
            return indices

        # Source: Raw text
        self.src_indices = [encode(t, add_special=False) for t in df["before"]]

        # Target: Normalized text (with SOS/EOS for Seq2Seq)
        if "after" in df.columns:
            self.tgt_indices = [encode(t, add_special=True) for t in df["after"]]
        else:
            self.tgt_indices = [[] for _ in range(len(df))]

        # Class conditioning
        if "class" in df.columns:
            self.class_indices = [self.vocab_classes[c] for c in df["class"]]
        else:
            self.class_indices = [0] * len(df)

        print(f"Saving fallback data to {self.cache_path}...")
        torch.save(
            {
                "src_indices": self.src_indices,
                "tgt_indices": self.tgt_indices,
                "class_indices": self.class_indices,
            },
            self.cache_path,
        )

    def __len__(self):
        return len(self.src_indices)

    def __getitem__(self, idx):
        src = torch.tensor(self.src_indices[idx], dtype=torch.long)
        tgt = torch.tensor(self.tgt_indices[idx], dtype=torch.long)
        cls = torch.tensor(self.class_indices[idx], dtype=torch.long)
        return src, tgt, cls


def collate_fn_fallback(batch):
    """
    Collates batch for Seq2Seq model.
    """
    src_list, tgt_list, cls_list = zip(*batch)

    # Pad sequences
    padded_src = pad_sequence(src_list, batch_first=True, padding_value=0)
    padded_tgt = pad_sequence(tgt_list, batch_first=True, padding_value=0)

    # Stack classes
    stacked_cls = torch.stack(cls_list)

    return padded_src, padded_tgt, stacked_cls
