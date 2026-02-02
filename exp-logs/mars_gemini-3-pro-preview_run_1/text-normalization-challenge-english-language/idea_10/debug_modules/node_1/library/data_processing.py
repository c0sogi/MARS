import os
import torch
import pandas as pd
import numpy as np
import sentencepiece as spm
import json
from torch.utils.data import Dataset
from collections import Counter
from library.config import Config
from library.utils import load_or_create_cache
from library.features import get_cached_features

# =========================================================================
# Vocabulary Management
# =========================================================================


class Vocabulary:
    """
    Handles mapping between tokens (words/chars/classes) and integer IDs.
    """

    def __init__(self, name, special_tokens=None):
        self.name = name
        self.token2id = {}
        self.id2token = {}
        self.special_tokens = special_tokens if special_tokens else []

        # Add special tokens first
        for token in self.special_tokens:
            self.add_token(token)

    def add_token(self, token):
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token

    def get_id(self, token, default=None):
        return self.token2id.get(token, default)

    def get_token(self, idx, default=None):
        return self.id2token.get(idx, default)

    def __len__(self):
        return len(self.token2id)

    def save(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f"vocab_{self.name}.json")
        with open(path, "w") as f:
            json.dump(self.token2id, f)

    def load(self, dir_path):
        path = os.path.join(dir_path, f"vocab_{self.name}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                self.token2id = json.load(f)
            self.id2token = {v: k for k, v in self.token2id.items()}
            return True
        return False

    @classmethod
    def build(cls, tokens, name, max_size=None, special_tokens=None):
        vocab = cls(name, special_tokens)
        counter = Counter(tokens)

        # Sort by frequency then alphabetically
        most_common = counter.most_common()
        if max_size:
            most_common = most_common[:max_size]

        for token, _ in most_common:
            vocab.add_token(token)

        return vocab


# =========================================================================
# BPE Tokenizer Wrapper
# =========================================================================


class BPETokenizer:
    def __init__(self):
        self.config = Config()
        self.model_prefix = self.config.BPE_MODEL_PREFIX
        self.vocab_size = self.config.BPE_VOCAB_SIZE
        self.sp = spm.SentencePieceProcessor()

    def train(self, texts):
        """
        Trains the SentencePiece model.
        Args:
            texts (list or pd.Series): List of sentences or tokens to train on.
        """
        # Create a temp file for training data
        temp_file = os.path.join(self.config.WORKING_DIR, "bpe_train_data.txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(str(text) + "\n")

        # Train model
        cmd = (
            f"--input={temp_file} --model_prefix={self.model_prefix} "
            f"--vocab_size={self.vocab_size} --model_type=bpe "
            f"--character_coverage=0.9995 --pad_id=0 --unk_id=1 --bos_id=-1 --eos_id=-1"
        )
        # SentencePiece prints a lot of info, we can suppress it if needed, but keeping it for logs is fine
        spm.SentencePieceTrainer.train(cmd)

        # Load the trained model
        self.load()

        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)

    def load(self):
        model_path = self.model_prefix + ".model"
        if os.path.exists(model_path):
            self.sp.load(model_path)
        else:
            print(f"Warning: BPE model not found at {model_path}")

    def encode(self, text):
        return self.sp.encode_as_ids(str(text))


# =========================================================================
# Data Preparation Helpers
# =========================================================================


def _group_data_wrapper(df, features=None):
    """
    Groups the dataframe by sentence_id.
    """
    print("Grouping data by sentence_id...")

    # If features are provided, add them to the dataframe temporarily
    if features is not None:
        # Convert numpy array to list of lists to store in pandas
        df["features"] = list(features)

    # Group by sentence_id
    # We aggregate into lists
    agg_dict = {"token_id": list, "before": list, "id": list}

    if "class" in df.columns:
        agg_dict["class"] = list
    if "after" in df.columns:
        agg_dict["after"] = list
    if features is not None:
        agg_dict["features"] = list

    grouped = df.groupby("sentence_id").agg(agg_dict).reset_index()

    # Sort by sentence_id just in case
    grouped["sentence_id"] = grouped["sentence_id"].astype(int)
    grouped = grouped.sort_values("sentence_id").reset_index(drop=True)

    return grouped


def get_grouped_data(df, cache_name, features=None, load_cached_data=True):
    config = Config()
    cache_path = os.path.join(
        config.WORKING_DIR, "cache", f"{cache_name}_grouped.parquet"
    )

    return load_or_create_cache(
        file_path=cache_path,
        compute_func=_group_data_wrapper,
        load_cached_data=load_cached_data,
        df=df,
        features=features,
    )


# =========================================================================
# Datasets
# =========================================================================


class TaggerDataset(Dataset):
    """
    Dataset for the Quad-Hybrid Bi-LSTM Tagger.
    Returns sentence-level batches.
    """

    def __init__(
        self, grouped_df, vocab_words, vocab_chars, vocab_classes, bpe_tokenizer
    ):
        self.df = grouped_df
        self.vocab_words = vocab_words
        self.vocab_chars = vocab_chars
        self.vocab_classes = vocab_classes
        self.bpe = bpe_tokenizer
        self.config = Config()

        self.max_sent_len = self.config.MAX_SENT_LEN
        self.max_token_char_len = self.config.MAX_TOKEN_CHAR_LEN
        self.max_bpe_len = 10  # Heuristic for max subwords per token

        # Pre-fetch special token IDs
        self.pad_word_id = self.vocab_words.get_id("<PAD>")
        self.unk_word_id = self.vocab_words.get_id("<UNK>")
        self.pad_char_id = self.vocab_chars.get_id("<PAD>")
        self.unk_char_id = self.vocab_chars.get_id("<UNK>")
        self.pad_class_id = self.vocab_classes.get_id("PLAIN")  # Default/Padding class

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        tokens = row["before"]
        # Handle features: if loaded from parquet, they are lists. Convert to numpy.
        features = (
            np.array(row["features"], dtype=np.float32)
            if "features" in row
            else np.zeros((len(tokens), self.config.NUM_REGEX_FEATURES))
        )
        classes = row["class"] if "class" in row else ["PLAIN"] * len(tokens)

        # Truncate to max sentence length
        seq_len = min(len(tokens), self.max_sent_len)
        tokens = tokens[:seq_len]
        features = features[:seq_len]
        classes = classes[:seq_len]

        # Initialize tensors
        word_ids = np.full(self.max_sent_len, self.pad_word_id, dtype=np.int64)
        char_ids = np.full(
            (self.max_sent_len, self.max_token_char_len),
            self.pad_char_id,
            dtype=np.int64,
        )
        bpe_ids = np.full(
            (self.max_sent_len, self.max_bpe_len), 0, dtype=np.int64
        )  # 0 is PAD in SentencePiece
        feat_vecs = np.zeros(
            (self.max_sent_len, self.config.NUM_REGEX_FEATURES), dtype=np.float32
        )
        target_ids = np.full(self.max_sent_len, self.pad_class_id, dtype=np.int64)
        mask = np.zeros(self.max_sent_len, dtype=np.float32)

        for i, token in enumerate(tokens):
            token_str = str(token)

            # 1. Word ID
            word_ids[i] = self.vocab_words.get_id(token_str, self.unk_word_id)

            # 2. Char IDs
            chars = [self.vocab_chars.get_id(c, self.unk_char_id) for c in token_str]
            c_len = min(len(chars), self.max_token_char_len)
            char_ids[i, :c_len] = chars[:c_len]

            # 3. BPE IDs
            # bpe.encode returns list of ints. 0 is pad.
            subwords = self.bpe.encode(token_str)
            b_len = min(len(subwords), self.max_bpe_len)
            bpe_ids[i, :b_len] = subwords[:b_len]

            # 4. Explicit Features
            feat_vecs[i] = features[i]

            # 5. Target
            target_ids[i] = self.vocab_classes.get_id(classes[i], self.pad_class_id)

            # 6. Mask
            mask[i] = 1.0

        return {
            "word_ids": torch.tensor(word_ids),
            "char_ids": torch.tensor(char_ids),
            "bpe_ids": torch.tensor(bpe_ids),
            "features": torch.tensor(feat_vecs),
            "targets": torch.tensor(target_ids),
            "mask": torch.tensor(mask),
        }


class Seq2SeqDataset(Dataset):
    """
    Dataset for the Transformer Fallback (Seq2Seq).
    Operates on token pairs (before, after) where change occurred.
    """

    def __init__(self, df, vocab_chars, vocab_classes):
        self.df = df
        self.vocab_chars = vocab_chars
        self.vocab_classes = vocab_classes
        self.config = Config()

        self.max_len = self.config.MAX_TOKEN_CHAR_LEN

        self.pad_id = self.vocab_chars.get_id("<PAD>")
        self.unk_id = self.vocab_chars.get_id("<UNK>")
        self.sos_id = self.vocab_chars.get_id("<SOS>")
        self.eos_id = self.vocab_chars.get_id("<EOS>")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src_token = str(row["before"])
        tgt_token = str(row["after"])
        cls_name = str(row["class"])

        # Source Chars
        src_indices = [self.vocab_chars.get_id(c, self.unk_id) for c in src_token]
        src_indices = src_indices[: self.max_len]

        # Target Chars (with SOS/EOS)
        tgt_indices = [self.vocab_chars.get_id(c, self.unk_id) for c in tgt_token]
        # Truncate to max_len - 2 to fit SOS/EOS
        tgt_indices = tgt_indices[: self.max_len - 2]
        tgt_in = [self.sos_id] + tgt_indices
        tgt_out = tgt_indices + [self.eos_id]

        # Padding
        src_pad_len = self.max_len - len(src_indices)
        src_ids = src_indices + [self.pad_id] * src_pad_len

        tgt_pad_len = self.max_len - len(tgt_in)
        tgt_in_ids = tgt_in + [self.pad_id] * tgt_pad_len
        tgt_out_ids = tgt_out + [self.pad_id] * tgt_pad_len

        # Class ID
        class_id = self.vocab_classes.get_id(cls_name, 0)  # 0 default

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_in": torch.tensor(tgt_in_ids, dtype=torch.long),
            "tgt_out": torch.tensor(tgt_out_ids, dtype=torch.long),
            "class_id": torch.tensor(class_id, dtype=torch.long),
        }


# =========================================================================
# Knowledge Base
# =========================================================================


def _build_kb_wrapper(df):
    """
    Constructs the deterministic knowledge base.
    """
    print("Building Knowledge Base...")

    # Group by (before, class) and find mode of 'after'
    # Using a simple aggregation: take the most frequent 'after'
    # If tie or single, take first.
    def get_mode(x):
        m = pd.Series.mode(x)
        return m[0] if not m.empty else x.iloc[0]

    kb = df.groupby(["before", "class"])["after"].agg(get_mode).reset_index()
    return kb


def build_knowledge_base(df, load_cached_data=True):
    config = Config()
    return load_or_create_cache(
        file_path=config.KB_PATH,
        compute_func=_build_kb_wrapper,
        load_cached_data=load_cached_data,
        df=df,
    )


# =========================================================================
# Main Data Processing Logic
# =========================================================================


def prepare_data(load_cached_data=True):
    """
    Main entry point to prepare all data artifacts.
    """
    config = Config()

    # 1. Load Metadata
    print("Loading metadata...")
    df_train = pd.read_csv(config.TRAIN_FILE, dtype=str, keep_default_na=False)
    df_val = pd.read_csv(config.VAL_FILE, dtype=str, keep_default_na=False)
    df_test = pd.read_csv(config.TEST_FILE, dtype=str, keep_default_na=False)

    if config.DEBUG and config.SAMPLE_SIZE:
        print(f"DEBUG: Sampling {config.SAMPLE_SIZE} rows...")
        df_train = df_train.head(config.SAMPLE_SIZE)
        df_val = df_val.head(config.SAMPLE_SIZE)
        # We generally want to test on full test set even in debug to ensure submission format is correct,
        # but for speed in debug mode we can sample.
        # df_test = df_test.head(config.SAMPLE_SIZE)

    # 2. Build/Load Vocabularies
    print("Building Vocabularies...")

    # Words
    vocab_words = Vocabulary("words", special_tokens=["<PAD>", "<UNK>"])
    if not vocab_words.load(config.VOCAB_DIR) or not load_cached_data:
        vocab_words = Vocabulary.build(
            df_train["before"],
            "words",
            config.WORD_VOCAB_SIZE,
            special_tokens=["<PAD>", "<UNK>"],
        )
        vocab_words.save(config.VOCAB_DIR)

    # Chars
    vocab_chars = Vocabulary(
        "chars", special_tokens=["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
    )
    if not vocab_chars.load(config.VOCAB_DIR) or not load_cached_data:
        # Collect all chars from before and after
        all_chars = set()
        for s in df_train["before"].astype(str):
            all_chars.update(s)
        for s in df_train["after"].astype(str):
            all_chars.update(s)
        # Add basic ascii just in case
        import string

        all_chars.update(string.printable)

        for c in sorted(list(all_chars)):
            vocab_chars.add_token(c)
        vocab_chars.save(config.VOCAB_DIR)

    # Classes
    vocab_classes = Vocabulary("classes", special_tokens=[])
    if not vocab_classes.load(config.VOCAB_DIR) or not load_cached_data:
        vocab_classes = Vocabulary.build(
            df_train["class"], "classes", special_tokens=[]
        )
        vocab_classes.save(config.VOCAB_DIR)

    # 3. Train/Load BPE
    print("Initializing BPE Tokenizer...")
    bpe = BPETokenizer()
    if not os.path.exists(config.BPE_MODEL_PREFIX + ".model") or not load_cached_data:
        print("Training BPE model...")
        bpe.train(df_train["before"].astype(str))
    else:
        bpe.load()

    # 4. Compute Explicit Features
    print("Computing Explicit Features...")
    train_features = get_cached_features(df_train, "train", load_cached_data)
    val_features = get_cached_features(df_val, "val", load_cached_data)
    test_features = get_cached_features(df_test, "test", load_cached_data)

    # 5. Create Grouped Data (for Tagger)
    print("Grouping data for Tagger...")
    train_grouped = get_grouped_data(
        df_train, "train", train_features, load_cached_data
    )
    val_grouped = get_grouped_data(df_val, "val", val_features, load_cached_data)
    test_grouped = get_grouped_data(df_test, "test", test_features, load_cached_data)

    # 6. Create Seq2Seq Data (Filter for changes)
    print("Filtering data for Seq2Seq Fallback...")
    # We only train seq2seq on tokens that actually change
    mask_change = df_train["before"] != df_train["after"]
    df_seq2seq_train = df_train[mask_change].copy()

    mask_change_val = df_val["before"] != df_val["after"]
    df_seq2seq_val = df_val[mask_change_val].copy()

    # 7. Build Knowledge Base
    kb_df = build_knowledge_base(df_train, load_cached_data)

    return {
        "vocab_words": vocab_words,
        "vocab_chars": vocab_chars,
        "vocab_classes": vocab_classes,
        "bpe_tokenizer": bpe,
        "train_grouped": train_grouped,
        "val_grouped": val_grouped,
        "test_grouped": test_grouped,
        "seq2seq_train": df_seq2seq_train,
        "seq2seq_val": df_seq2seq_val,
        "kb_df": kb_df,
    }
