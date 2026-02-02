import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import collections
from library.config import Config
from library.utils import set_seed

# =========================================================================
# 1. VOCABULARY MANAGEMENT
# =========================================================================


class Vocabulary:
    def __init__(self, name, specials=None):
        self.name = name
        self.stoi = {}
        self.itos = {}
        self.specials = specials if specials else ["<pad>", "<unk>"]

        # Initialize with specials
        for i, s in enumerate(self.specials):
            self.stoi[s] = i
            self.itos[i] = s

    def __len__(self):
        return len(self.stoi)

    def add_tokens(self, tokens):
        for token in tokens:
            if token not in self.stoi:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx] = token

    def lookup_indices(self, tokens, unk_token="<unk>"):
        unk_idx = self.stoi.get(unk_token)
        return [self.stoi.get(t, unk_idx) for t in tokens]

    def lookup_token(self, idx):
        return self.itos.get(idx, "<unk>")

    def save(self, path):
        # Save as parquet: token, index
        data = [{"token": k, "index": v} for k, v in self.stoi.items()]
        df = pd.DataFrame(data)
        df.to_parquet(path, index=False)

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")
        df = pd.read_parquet(path)
        self.stoi = dict(zip(df["token"], df["index"]))
        self.itos = dict(zip(df["index"], df["token"]))


# =========================================================================
# 2. KNOWLEDGE BASE
# =========================================================================


class KnowledgeBase:
    def __init__(self):
        # Map (token, class) -> normalized_text
        self.lookup = {}

    def build(self, df):
        """
        Builds the KB from a dataframe containing 'before', 'class', 'after'.
        We prioritize the most frequent normalization if there are conflicts,
        though usually (token, class) is deterministic.
        """
        # Filter for valid columns
        if "class" not in df.columns or "after" not in df.columns:
            return

        # Group by token and class, take the most frequent 'after'
        # This resolves rare inconsistencies in labeling
        grouped = (
            df.groupby(["before", "class"])["after"]
            .agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
            .reset_index()
        )

        for _, row in grouped.iterrows():
            key = (str(row["before"]), str(row["class"]))
            self.lookup[key] = str(row["after"])

    def get(self, token, class_name):
        return self.lookup.get((str(token), str(class_name)), None)

    def save(self, path):
        # Save as parquet
        data = []
        for (token, cls), after in self.lookup.items():
            data.append({"before": token, "class": cls, "after": after})
        df = pd.DataFrame(data)
        df.to_parquet(path, index=False)

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"KB file not found at {path}")
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            self.lookup[(str(row["before"]), str(row["class"]))] = str(row["after"])


# =========================================================================
# 3. DATA PROCESSING & CACHING
# =========================================================================


def load_or_create_grouped_data(
    csv_path, cache_path, load_cached_data=True, is_test=False
):
    """
    Loads raw CSV, groups by sentence_id, and caches as Parquet.
    """
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached grouped data from {cache_path}")
        df_grouped = pd.read_parquet(cache_path)

        # Handle Debug Mode on cached data
        if Config.DEBUG:
            df_grouped = df_grouped.head(Config.MAX_DEBUG_SAMPLES)
        return df_grouped

    print(f"Processing raw data from {csv_path}...")
    # Read CSV
    # Using keep_default_na=False to handle "null" or "NaN" as literal text if present
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    # Ensure ID columns are int for sorting
    df["sentence_id"] = pd.to_numeric(df["sentence_id"])
    df["token_id"] = pd.to_numeric(df["token_id"])

    # Sort just in case
    df = df.sort_values(["sentence_id", "token_id"])

    # Group by sentence_id
    # We aggregate columns into lists
    agg_dict = {"before": list, "id": list}
    if not is_test:
        agg_dict["class"] = list
        agg_dict["after"] = list

    df_grouped = df.groupby("sentence_id").agg(agg_dict).reset_index()

    # Save to cache
    print(f"Saving grouped data to {cache_path}...")
    df_grouped.to_parquet(cache_path, index=False)

    # Handle Debug Mode
    if Config.DEBUG:
        df_grouped = df_grouped.head(Config.MAX_DEBUG_SAMPLES)

    return df_grouped


def load_or_create_vocabs(train_grouped_df, load_cached_data=True):
    """
    Creates or loads vocabularies for Tokens, Characters, and Classes.
    """
    # Paths
    token_path = Config.VOCAB_TOKENS_PATH
    char_path = Config.VOCAB_CHARS_PATH
    class_path = Config.VOCAB_CLASSES_PATH

    vocab_tokens = Vocabulary("tokens", specials=["<pad>", "<unk>"])
    vocab_chars = Vocabulary("chars", specials=["<pad>", "<unk>", "<sos>", "<eos>"])
    vocab_classes = Vocabulary(
        "classes", specials=[]
    )  # Classes don't need padding usually, but we handle indices carefully

    if (
        load_cached_data
        and os.path.exists(token_path)
        and os.path.exists(char_path)
        and os.path.exists(class_path)
    ):
        print("Loading vocabularies from cache...")
        vocab_tokens.load(token_path)
        vocab_chars.load(char_path)
        vocab_classes.load(class_path)
        return vocab_tokens, vocab_chars, vocab_classes

    print("Building vocabularies from training data...")

    # Flatten lists
    all_tokens = [t for seq in train_grouped_df["before"] for t in seq]
    all_classes = [c for seq in train_grouped_df["class"] for c in seq]

    # 1. Token Vocab (Frequency filtering)
    counter = collections.Counter(all_tokens)
    common_tokens = [
        t for t, c in counter.most_common(Config.MAX_VOCAB_SIZE) if c >= Config.MIN_FREQ
    ]
    vocab_tokens.add_tokens(common_tokens)

    # 2. Char Vocab
    # Include chars from 'before' and 'after' (for seq2seq)
    # Since we only have 'before' easily accessible here as a flat list, let's iterate
    # Note: Ideally we scan 'after' too. Let's do that.
    all_after = [t for seq in train_grouped_df["after"] for t in seq]
    unique_chars = set()
    for t in all_tokens:
        unique_chars.update(str(t))
    for t in all_after:
        unique_chars.update(str(t))

    vocab_chars.add_tokens(sorted(list(unique_chars)))

    # 3. Class Vocab
    unique_classes = sorted(list(set(all_classes)))
    vocab_classes.add_tokens(unique_classes)

    # Save
    print("Saving vocabularies...")
    vocab_tokens.save(token_path)
    vocab_chars.save(char_path)
    vocab_classes.save(class_path)

    return vocab_tokens, vocab_chars, vocab_classes


def load_or_create_kb(train_grouped_df, load_cached_data=True):
    """
    Creates or loads the Knowledge Base.
    """
    kb_path = Config.KNOWLEDGE_BASE_PATH
    kb = KnowledgeBase()

    # Verify cache schema before loading (Cite debug_lesson_4)
    cache_valid = False
    if load_cached_data and os.path.exists(kb_path):
        try:
            df_check = pd.read_parquet(kb_path)
            if {"before", "class", "after"}.issubset(df_check.columns):
                cache_valid = True
        except Exception:
            pass

    if cache_valid:
        print("Loading Knowledge Base from cache...")
        kb.load(kb_path)
        return kb

    print("Building Knowledge Base...")
    # We need a flat dataframe for KB construction
    # Explode the grouped dataframe
    flat_data = {
        "before": [t for seq in train_grouped_df["before"] for t in seq],
        "class": [c for seq in train_grouped_df["class"] for c in seq],
        "after": [a for seq in train_grouped_df["after"] for a in seq],
    }
    df_flat = pd.DataFrame(flat_data)

    kb.build(df_flat)

    print("Saving Knowledge Base...")
    kb.save(kb_path)
    return kb


# =========================================================================
# 4. DATASET CLASSES
# =========================================================================


class TaggerDataset(Dataset):
    """
    Dataset for the Bi-LSTM Tagger.
    Returns:
        - word_ids: (seq_len)
        - char_ids: (seq_len, max_char_len)
        - class_ids: (seq_len)
    """

    def __init__(
        self, df_grouped, vocab_tokens, vocab_chars, vocab_classes, is_test=False
    ):
        self.df = df_grouped
        self.vocab_tokens = vocab_tokens
        self.vocab_chars = vocab_chars
        self.vocab_classes = vocab_classes
        self.is_test = is_test
        self.max_char_len = Config.MAX_CHAR_LEN

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = row["before"]

        # Word IDs
        word_ids = self.vocab_tokens.lookup_indices(tokens)

        # Char IDs (2D: seq_len x max_char_len)
        char_ids_list = []
        for t in tokens:
            chars = list(str(t))
            c_ids = self.vocab_chars.lookup_indices(chars)
            # Truncate
            c_ids = c_ids[: self.max_char_len]
            # Pad is handled in collate or here?
            # Better to return list here and pad in collate for batch efficiency,
            # but for CNN fixed width is often expected. Let's pad to fixed width here for simplicity.
            pad_len = self.max_char_len - len(c_ids)
            if pad_len > 0:
                c_ids = c_ids + [self.vocab_chars.stoi["<pad>"]] * pad_len
            char_ids_list.append(c_ids)

        if not self.is_test:
            classes = row["class"]
            class_ids = self.vocab_classes.lookup_indices(
                classes, unk_token=None
            )  # Classes shouldn't be unk usually
            # Handle potential unk if class not found (shouldn't happen with proper vocab)
            class_ids = [c if c is not None else 0 for c in class_ids]
            return (
                torch.tensor(word_ids, dtype=torch.long),
                torch.tensor(char_ids_list, dtype=torch.long),
                torch.tensor(class_ids, dtype=torch.long),
            )
        else:
            return (
                torch.tensor(word_ids, dtype=torch.long),
                torch.tensor(char_ids_list, dtype=torch.long),
                row["id"],
            )  # Return IDs for submission mapping


def tagger_collate_fn(batch):
    """
    Collate function for TaggerDataset.
    Pads sentences to the length of the longest sentence in the batch.
    """
    # Check if test or train
    is_test = len(batch[0]) == 3 and isinstance(
        batch[0][2], list
    )  # id is list of strings

    if not is_test:
        word_ids, char_ids, class_ids = zip(*batch)

        # Pad sequences (batch_first=True)
        word_ids_padded = pad_sequence(
            word_ids, batch_first=True, padding_value=0
        )  # 0 is <pad>
        class_ids_padded = pad_sequence(
            class_ids, batch_first=True, padding_value=-1
        )  # -1 for ignore_index in loss

        # Pad char_ids. Input is (seq_len, char_len). Output: (batch, max_seq_len, char_len)
        # pad_sequence works on the first dim (seq_len).
        char_ids_padded = pad_sequence(char_ids, batch_first=True, padding_value=0)

        return word_ids_padded, char_ids_padded, class_ids_padded
    else:
        word_ids, char_ids, row_ids = zip(*batch)
        word_ids_padded = pad_sequence(word_ids, batch_first=True, padding_value=0)
        char_ids_padded = pad_sequence(char_ids, batch_first=True, padding_value=0)
        return word_ids_padded, char_ids_padded, row_ids


class Seq2SeqDataset(Dataset):
    """
    Dataset for the Fallback Model.
    Filters for tokens where before != after.
    Returns:
        - src_chars: (src_len)
        - tgt_chars: (tgt_len)
    """

    def __init__(self, df_grouped, vocab_chars):
        self.vocab_chars = vocab_chars
        self.data = []

        # Pre-process: flatten and filter
        # This might be memory intensive, but necessary for efficient training
        # We iterate grouped df
        for _, row in df_grouped.iterrows():
            befores = row["before"]
            afters = row["after"]
            classes = row["class"]

            for b, a, c in zip(befores, afters, classes):
                # Filter logic: Only train on changed tokens
                # Also skip PLAIN and PUNCT explicitly as they are handled by identity rule
                if b != a and c not in ["PLAIN", "PUNCT"]:
                    self.data.append((b, a))

        if Config.DEBUG:
            self.data = self.data[: Config.MAX_DEBUG_SAMPLES]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_txt, tgt_txt = self.data[idx]

        # Add <sos> and <eos>
        # Src: chars
        src_ids = self.vocab_chars.lookup_indices(list(src_txt))

        # Tgt: <sos> + chars + <eos>
        tgt_ids = (
            [self.vocab_chars.stoi["<sos>"]]
            + self.vocab_chars.lookup_indices(list(tgt_txt))
            + [self.vocab_chars.stoi["<eos>"]]
        )

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(
            tgt_ids, dtype=torch.long
        )


def seq2seq_collate_fn(batch):
    src_ids, tgt_ids = zip(*batch)

    # Pad
    src_padded = pad_sequence(src_ids, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_ids, batch_first=True, padding_value=0)

    return src_padded, tgt_padded


# =========================================================================
# 5. MAIN INTERFACE
# =========================================================================


def get_tagger_dataloaders(load_cached_data=True):
    """
    Main function to get dataloaders for the Tagger.
    """
    # 1. Load Data
    train_df = load_or_create_grouped_data(
        Config.TRAIN_DATA_PATH, Config.TRAIN_GROUPED_PATH, load_cached_data
    )
    val_df = load_or_create_grouped_data(
        Config.VAL_DATA_PATH, Config.VAL_GROUPED_PATH, load_cached_data
    )

    # 2. Vocabs
    vocab_tokens, vocab_chars, vocab_classes = load_or_create_vocabs(
        train_df, load_cached_data
    )

    # 3. Datasets
    train_ds = TaggerDataset(train_df, vocab_tokens, vocab_chars, vocab_classes)
    val_ds = TaggerDataset(val_df, vocab_tokens, vocab_chars, vocab_classes)

    # 4. Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=tagger_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=tagger_collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, vocab_tokens, vocab_chars, vocab_classes


def get_seq2seq_dataloaders(load_cached_data=True):
    """
    Main function to get dataloaders for the Seq2Seq Fallback.
    """
    # 1. Load Data (Need train for training, val for validation)
    train_df = load_or_create_grouped_data(
        Config.TRAIN_DATA_PATH, Config.TRAIN_GROUPED_PATH, load_cached_data
    )
    val_df = load_or_create_grouped_data(
        Config.VAL_DATA_PATH, Config.VAL_GROUPED_PATH, load_cached_data
    )

    # 2. Vocabs (Reuse existing if possible)
    vocab_tokens, vocab_chars, vocab_classes = load_or_create_vocabs(
        train_df, load_cached_data
    )

    # 3. Datasets
    train_ds = Seq2SeqDataset(train_df, vocab_chars)
    val_ds = Seq2SeqDataset(val_df, vocab_chars)

    # 4. Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=seq2seq_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=seq2seq_collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, vocab_chars


def get_test_dataloader(load_cached_data=True):
    """
    Gets dataloader for inference.
    """
    test_df = load_or_create_grouped_data(
        Config.TEST_DATA_PATH, Config.TEST_GROUPED_PATH, load_cached_data, is_test=True
    )

    # Load vocabs (must exist)
    vocab_tokens = Vocabulary("tokens")
    vocab_chars = Vocabulary("chars")
    vocab_classes = Vocabulary("classes")

    vocab_tokens.load(Config.VOCAB_TOKENS_PATH)
    vocab_chars.load(Config.VOCAB_CHARS_PATH)
    vocab_classes.load(Config.VOCAB_CLASSES_PATH)

    test_ds = TaggerDataset(
        test_df, vocab_tokens, vocab_chars, vocab_classes, is_test=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=tagger_collate_fn,
        pin_memory=True,
    )

    return test_loader, test_df, vocab_tokens, vocab_chars, vocab_classes


def get_knowledge_base(load_cached_data=True):
    # Check cache validity (Cite debug_lesson_4: Verify schema at interface)
    cache_path = Config.KNOWLEDGE_BASE_PATH
    is_cache_valid = False
    if load_cached_data and os.path.exists(cache_path):
        try:
            df_check = pd.read_parquet(cache_path)
            if {"before", "class", "after"}.issubset(df_check.columns):
                is_cache_valid = True
        except Exception:
            pass

    # Ensure train data is processed to build KB if not cached or invalid
    if not is_cache_valid:
        train_df = load_or_create_grouped_data(
            Config.TRAIN_DATA_PATH, Config.TRAIN_GROUPED_PATH, load_cached_data
        )
        # Force rebuild by passing load_cached_data=False
        return load_or_create_kb(train_df, load_cached_data=False)
    else:
        # Load directly
        kb = KnowledgeBase()
        kb.load(cache_path)
        return kb
