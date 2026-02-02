import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from library.config import Config

# =========================================================================
# VOCABULARY MANAGEMENT
# =========================================================================


class Vocabulary:
    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        self.char2id = {}
        self.id2char = {}
        self.class2id = {}
        self.id2class = {}

    def build(self, df_train):
        """
        Builds vocabularies from the training dataframe.
        """
        print("Building vocabularies...")

        # 1. Class Vocabulary
        classes = sorted(df_train["class"].unique().tolist())
        self.class2id = {c: i for i, c in enumerate(classes)}
        self.id2class = {i: c for i, c in enumerate(classes)}

        # 2. Character Vocabulary (from both before and after to cover all generated text)
        # We use a set for speed
        chars = set()
        # Sample a subset for speed if dataset is massive, but here we iterate all
        # Using vectorized operations for speed
        all_text = pd.concat([df_train["before"], df_train["after"]]).astype(str)
        unique_chars = set("".join(all_text.unique()))

        special_tokens = [
            Config.PAD_TOKEN,
            Config.UNK_TOKEN,
            Config.SOS_TOKEN,
            Config.EOS_TOKEN,
        ]

        self.char2id = {t: i for i, t in enumerate(special_tokens)}
        current_id = len(special_tokens)
        for c in sorted(list(unique_chars)):
            if c not in self.char2id:
                self.char2id[c] = current_id
                current_id += 1
        self.id2char = {v: k for k, v in self.char2id.items()}

        # 3. Token Vocabulary (Input words)
        # Count frequencies
        counter = Counter(df_train["before"].astype(str))

        self.token2id = {Config.PAD_TOKEN: 0, Config.UNK_TOKEN: 1}
        current_id = 2

        # Sort by frequency then alphabetically
        most_common = counter.most_common(Config.MAX_VOCAB_SIZE)

        for token, freq in most_common:
            if freq >= Config.MIN_FREQ:
                self.token2id[token] = current_id
                current_id += 1

        self.id2token = {v: k for k, v in self.token2id.items()}

        print(
            f"Vocab sizes - Tokens: {len(self.token2id)}, Chars: {len(self.char2id)}, Classes: {len(self.class2id)}"
        )

    def save(self):
        """Saves vocabularies to Parquet files."""
        # Tokens
        df_tokens = pd.DataFrame(list(self.token2id.items()), columns=["token", "id"])
        df_tokens.to_parquet(Config.VOCAB_TOKENS_PATH, index=False)

        # Chars
        df_chars = pd.DataFrame(list(self.char2id.items()), columns=["char", "id"])
        df_chars.to_parquet(Config.VOCAB_CHARS_PATH, index=False)

        # Classes
        df_classes = pd.DataFrame(list(self.class2id.items()), columns=["class", "id"])
        df_classes.to_parquet(Config.VOCAB_CLASSES_PATH, index=False)
        print("Vocabularies saved.")

    def load(self):
        """Loads vocabularies from Parquet files."""
        if not (
            os.path.exists(Config.VOCAB_TOKENS_PATH)
            and os.path.exists(Config.VOCAB_CHARS_PATH)
            and os.path.exists(Config.VOCAB_CLASSES_PATH)
        ):
            return False

        print("Loading vocabularies from cache...")
        df_tokens = pd.read_parquet(Config.VOCAB_TOKENS_PATH)
        self.token2id = dict(zip(df_tokens["token"], df_tokens["id"]))
        self.id2token = dict(zip(df_tokens["id"], df_tokens["token"]))

        df_chars = pd.read_parquet(Config.VOCAB_CHARS_PATH)
        self.char2id = dict(zip(df_chars["char"], df_chars["id"]))
        self.id2char = dict(zip(df_chars["id"], df_chars["char"]))

        df_classes = pd.read_parquet(Config.VOCAB_CLASSES_PATH)
        self.class2id = dict(zip(df_classes["class"], df_classes["id"]))
        self.id2class = dict(zip(df_classes["id"], df_classes["class"]))
        return True

    def encode_token(self, token):
        return self.token2id.get(str(token), self.token2id[Config.UNK_TOKEN])

    def encode_char(self, char):
        return self.char2id.get(char, self.char2id[Config.UNK_TOKEN])

    def encode_class(self, class_name):
        return self.class2id.get(
            class_name, 0
        )  # Default to 0 if unknown (shouldn't happen)

    def decode_class(self, class_id):
        return self.id2class.get(class_id, "PLAIN")


# =========================================================================
# KNOWLEDGE BASE
# =========================================================================


class KnowledgeBase:
    def __init__(self):
        self.lookup = {}

    def build(self, df_train):
        """
        Builds the deterministic lookup table (Token, Class) -> Normalized.
        """
        print("Building Knowledge Base...")
        # We prioritize the most frequent mapping if there are conflicts,
        # but usually (token, class) -> normalized is consistent.
        # If duplicates exist, the last one seen overrides.
        # To be robust, we could use mode, but simple assignment is fast and effective for this task.

        # Create a key column
        temp_df = df_train[["before", "class", "after"]].copy()
        temp_df["before"] = temp_df["before"].astype(str)
        temp_df["class"] = temp_df["class"].astype(str)
        temp_df["after"] = temp_df["after"].astype(str)

        # Drop duplicates to keep unique mappings
        temp_df = temp_df.drop_duplicates(subset=["before", "class"])

        # Convert to dictionary
        # Key: (before, class), Value: after
        self.lookup = dict(
            zip(zip(temp_df["before"], temp_df["class"]), temp_df["after"])
        )
        print(f"Knowledge Base built with {len(self.lookup)} entries.")

    def save(self):
        # Save as DataFrame
        data = [
            {"before": k[0], "class": k[1], "after": v} for k, v in self.lookup.items()
        ]
        df = pd.DataFrame(data)
        df.to_parquet(Config.KNOWLEDGE_BASE_PATH, index=False)
        print("Knowledge Base saved.")

    def load(self):
        if not os.path.exists(Config.KNOWLEDGE_BASE_PATH):
            return False
        print("Loading Knowledge Base from cache...")
        df = pd.read_parquet(Config.KNOWLEDGE_BASE_PATH)
        self.lookup = dict(zip(zip(df["before"], df["class"]), df["after"]))
        return True

    def get(self, token, class_name):
        return self.lookup.get((str(token), str(class_name)))


# =========================================================================
# DATASETS
# =========================================================================


class TaggerDataset(Dataset):
    """
    Dataset for the Sentence-Level Tagger (Bi-LSTM).
    Groups tokens by sentence_id.
    """

    def __init__(self, df_grouped, vocab):
        """
        Args:
            df_grouped: DataFrame where each row is a sentence.
                        Columns: [token_ids, char_ids_list, class_ids]
            vocab: Vocabulary object
        """
        self.data = df_grouped
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        token_ids = torch.tensor(row["token_ids"], dtype=torch.long)
        class_ids = torch.tensor(row["class_ids"], dtype=torch.long)

        # char_ids_list is a List[List[int]] (one list per token)
        # We return it as is; collate_fn will handle padding
        char_ids_list = row["char_ids_list"]

        return token_ids, char_ids_list, class_ids


class Seq2SeqDataset(Dataset):
    """
    Dataset for the Character-Level Transformer (Fallback).
    Contains only tokens where before != after.
    """

    def __init__(self, df_filtered, vocab):
        self.data = df_filtered
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        src = torch.tensor(row["src_ids"], dtype=torch.long)
        tgt = torch.tensor(row["tgt_ids"], dtype=torch.long)
        class_id = torch.tensor(row["class_id"], dtype=torch.long)

        return src, tgt, class_id


class SubmissionDataset(Dataset):
    """
    Dataset for Inference (Test Set).
    """

    def __init__(self, df_grouped, vocab):
        self.data = df_grouped
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        token_ids = torch.tensor(row["token_ids"], dtype=torch.long)
        char_ids_list = row["char_ids_list"]
        # We also need original tokens and IDs to reconstruct submission
        raw_tokens = row["raw_tokens"]
        row_ids = row["row_ids"]  # e.g. "0_1", "0_2"

        return token_ids, char_ids_list, raw_tokens, row_ids


# =========================================================================
# COLLATE FUNCTIONS
# =========================================================================


def collate_tagger(batch):
    """
    Pads sentences to the max length in the batch.
    Handles nested character padding.
    """
    # batch is list of (token_ids, char_ids_list, class_ids)

    token_ids_list = [item[0] for item in batch]
    char_ids_nested = [item[1] for item in batch]  # List of Lists of Lists
    class_ids_list = [item[2] for item in batch]

    # Pad Token and Class sequences (Batch, Seq_Len)
    padded_tokens = pad_sequence(token_ids_list, batch_first=True, padding_value=0)
    padded_classes = pad_sequence(
        class_ids_list, batch_first=True, padding_value=-100
    )  # -100 for ignore_index

    # Handle Character Padding
    # We need a tensor of shape (Batch, Seq_Len, Max_Word_Len)
    batch_size = len(batch)
    max_seq_len = padded_tokens.size(1)

    # Find max word length in this batch
    max_word_len = 0
    for sent_chars in char_ids_nested:
        for word_chars in sent_chars:
            max_word_len = max(max_word_len, len(word_chars))

    # Limit max word len to avoid OOM if some outlier exists
    max_word_len = min(max_word_len, 50)

    padded_chars = torch.zeros(
        (batch_size, max_seq_len, max_word_len), dtype=torch.long
    )

    for i, sent_chars in enumerate(char_ids_nested):
        # sent_chars is List[List[int]]
        seq_len = min(len(sent_chars), max_seq_len)
        for j in range(seq_len):
            word_chars = sent_chars[j]
            w_len = min(len(word_chars), max_word_len)
            if w_len > 0:
                padded_chars[i, j, :w_len] = torch.tensor(
                    word_chars[:w_len], dtype=torch.long
                )

    # Create mask (Batch, Seq_Len)
    mask = (padded_tokens != 0).long()

    return padded_tokens, padded_chars, padded_classes, mask


def collate_submission(batch):
    # batch is list of (token_ids, char_ids_list, raw_tokens, row_ids)
    token_ids_list = [item[0] for item in batch]
    char_ids_nested = [item[1] for item in batch]
    raw_tokens_list = [item[2] for item in batch]
    row_ids_list = [item[3] for item in batch]

    padded_tokens = pad_sequence(token_ids_list, batch_first=True, padding_value=0)

    batch_size = len(batch)
    max_seq_len = padded_tokens.size(1)

    max_word_len = 0
    for sent_chars in char_ids_nested:
        for word_chars in sent_chars:
            max_word_len = max(max_word_len, len(word_chars))
    max_word_len = min(max_word_len, 50)

    padded_chars = torch.zeros(
        (batch_size, max_seq_len, max_word_len), dtype=torch.long
    )

    for i, sent_chars in enumerate(char_ids_nested):
        seq_len = min(len(sent_chars), max_seq_len)
        for j in range(seq_len):
            word_chars = sent_chars[j]
            w_len = min(len(word_chars), max_word_len)
            if w_len > 0:
                padded_chars[i, j, :w_len] = torch.tensor(
                    word_chars[:w_len], dtype=torch.long
                )

    mask = (padded_tokens != 0).long()

    return padded_tokens, padded_chars, mask, raw_tokens_list, row_ids_list


def collate_seq2seq(batch):
    """
    Pads source and target character sequences.
    """
    # batch is list of (src, tgt, class_id)
    src_list = [item[0] for item in batch]
    tgt_list = [item[1] for item in batch]
    class_list = [item[2] for item in batch]

    # Pad sequences
    padded_src = pad_sequence(src_list, batch_first=True, padding_value=0)
    padded_tgt = pad_sequence(tgt_list, batch_first=True, padding_value=0)

    class_ids = torch.stack(class_list)

    # Create masks (True where padded)
    src_key_padding_mask = padded_src == 0
    tgt_key_padding_mask = padded_tgt == 0

    return padded_src, padded_tgt, class_ids, src_key_padding_mask, tgt_key_padding_mask


# =========================================================================
# DATA PREPARATION PIPELINE
# =========================================================================


def process_tagger_data(df, vocab):
    """
    Groups data by sentence_id and converts to IDs.
    Returns DataFrame suitable for TaggerDataset.
    """
    print("Processing Tagger Data (Grouping by Sentence)...")

    # Ensure strings
    df["before"] = df["before"].astype(str)
    if "class" in df.columns:
        df["class"] = df["class"].astype(str)

    # Helper to process a group
    # We use a more vectorized approach where possible or efficient apply

    # 1. Convert tokens to IDs
    # Using map is faster than apply
    # Handle UNK
    unk_id = vocab.token2id[Config.UNK_TOKEN]
    df["token_id_val"] = df["before"].map(vocab.token2id).fillna(unk_id).astype(int)

    # 2. Convert class to IDs (if exists)
    if "class" in df.columns:
        df["class_id_val"] = df["class"].map(vocab.class2id).fillna(0).astype(int)
    else:
        df["class_id_val"] = 0  # Dummy for test

    # 3. Convert chars to IDs (List of Lists)
    # This is the slow part.
    def encode_chars(text):
        return [vocab.char2id.get(c, vocab.char2id[Config.UNK_TOKEN]) for c in text]

    df["char_ids_val"] = df["before"].apply(encode_chars)

    # 4. Group by sentence_id
    # We aggregate into lists
    agg_dict = {
        "token_id_val": list,
        "char_ids_val": list,
        "before": list,  # Keep raw tokens for submission/debugging
        "id": list,  # Keep row IDs
    }
    if "class" in df.columns:
        agg_dict["class_id_val"] = list

    grouped = df.groupby("sentence_id").agg(agg_dict).reset_index()

    # Rename columns to match Dataset expectation
    rename_map = {
        "token_id_val": "token_ids",
        "char_ids_val": "char_ids_list",
        "class_id_val": "class_ids",
        "before": "raw_tokens",
        "id": "row_ids",
    }
    grouped = grouped.rename(columns=rename_map)

    return grouped


def process_seq2seq_data(df, vocab):
    """
    Filters for changed tokens and converts to character IDs.
    """
    print("Processing Seq2Seq Data (Filtering changed tokens)...")
    df["before"] = df["before"].astype(str)
    df["after"] = df["after"].astype(str)
    df["class"] = df["class"].astype(str)

    # Filter
    changed = df[df["before"] != df["after"]].copy()

    # Encode
    def encode_seq(text, add_special=True):
        ids = [vocab.char2id.get(c, vocab.char2id[Config.UNK_TOKEN]) for c in text]
        if add_special:
            ids = (
                [vocab.char2id[Config.SOS_TOKEN]]
                + ids
                + [vocab.char2id[Config.EOS_TOKEN]]
            )
        return ids

    # Src: No SOS/EOS needed for Encoder usually, but consistent embedding is fine.
    # Actually standard Transformer Encoder doesn't strictly need SOS/EOS, but Decoder needs them.
    # We will add SOS/EOS to Target. Source can just be chars.

    changed["src_ids"] = changed["before"].apply(
        lambda x: encode_seq(x, add_special=False)
    )
    changed["tgt_ids"] = changed["after"].apply(
        lambda x: encode_seq(x, add_special=True)
    )
    changed["class_id"] = changed["class"].map(vocab.class2id).fillna(0).astype(int)

    return changed[["src_ids", "tgt_ids", "class_id"]]


def prepare_data(load_cached_data=True):
    """
    Main entry point. Loads or creates all data resources.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Vocabulary & KB
    vocab = Vocabulary()
    kb = KnowledgeBase()

    if load_cached_data and vocab.load() and kb.load():
        print("Loaded Vocab and KB from cache.")
    else:
        # Load Raw Train Data
        print(f"Reading {Config.TRAIN_FILE}...")
        df_train = pd.read_csv(Config.TRAIN_FILE, dtype=str, keep_default_na=False)

        vocab.build(df_train)
        vocab.save()

        kb.build(df_train)
        kb.save()

        # Free memory
        del df_train

    # 2. Process Datasets
    # We process Train/Val/Test separately

    # --- Train ---
    if load_cached_data and os.path.exists(Config.TRAIN_GROUPED_PATH):
        print("Loading Train Grouped Data from cache...")
        # engine='pyarrow' is crucial for list columns
        df_train_grouped = pd.read_parquet(Config.TRAIN_GROUPED_PATH, engine="pyarrow")
        # Also need filtered seq2seq data?
        # For simplicity, we re-filter or we could save it separately.
        # Let's assume we re-process Seq2Seq from raw train since it's smaller?
        # No, better to cache it. But for now, let's just cache the grouped tagger data.
        # To strictly follow instructions, we should cache everything.
        # Let's assume we re-load raw train if we need to rebuild seq2seq.
    else:
        print("Processing Train Data...")
        df_train = pd.read_csv(Config.TRAIN_FILE, dtype=str, keep_default_na=False)
        df_train_grouped = process_tagger_data(df_train, vocab)
        df_train_grouped.to_parquet(
            Config.TRAIN_GROUPED_PATH, index=False, engine="pyarrow"
        )

        # Seq2Seq Train
        df_train_seq2seq = process_seq2seq_data(df_train, vocab)
        # Save as pickle or parquet? Parquet with lists works.
        df_train_seq2seq.to_parquet(
            Config.WORKING_DIR + "/train_seq2seq.parquet", index=False, engine="pyarrow"
        )
        del df_train

    # Load Seq2Seq Train if cached
    if load_cached_data and os.path.exists(
        Config.WORKING_DIR + "/train_seq2seq.parquet"
    ):
        df_train_seq2seq = pd.read_parquet(
            Config.WORKING_DIR + "/train_seq2seq.parquet", engine="pyarrow"
        )

    # --- Val ---
    if load_cached_data and os.path.exists(Config.VAL_GROUPED_PATH):
        print("Loading Val Grouped Data from cache...")
        df_val_grouped = pd.read_parquet(Config.VAL_GROUPED_PATH, engine="pyarrow")
        if os.path.exists(Config.WORKING_DIR + "/val_seq2seq.parquet"):
            df_val_seq2seq = pd.read_parquet(
                Config.WORKING_DIR + "/val_seq2seq.parquet", engine="pyarrow"
            )
        else:
            # Fallback if partial cache
            df_val = pd.read_csv(Config.VAL_FILE, dtype=str, keep_default_na=False)
            df_val_seq2seq = process_seq2seq_data(df_val, vocab)
            del df_val
    else:
        print("Processing Val Data...")
        df_val = pd.read_csv(Config.VAL_FILE, dtype=str, keep_default_na=False)
        df_val_grouped = process_tagger_data(df_val, vocab)
        df_val_grouped.to_parquet(
            Config.VAL_GROUPED_PATH, index=False, engine="pyarrow"
        )

        df_val_seq2seq = process_seq2seq_data(df_val, vocab)
        df_val_seq2seq.to_parquet(
            Config.WORKING_DIR + "/val_seq2seq.parquet", index=False, engine="pyarrow"
        )
        del df_val

    # --- Test ---
    if load_cached_data and os.path.exists(Config.TEST_GROUPED_PATH):
        print("Loading Test Grouped Data from cache...")
        df_test_grouped = pd.read_parquet(Config.TEST_GROUPED_PATH, engine="pyarrow")
    else:
        print("Processing Test Data...")
        df_test = pd.read_csv(Config.TEST_FILE, dtype=str, keep_default_na=False)
        # Test data doesn't have 'class', process_tagger_data handles this
        df_test_grouped = process_tagger_data(df_test, vocab)
        df_test_grouped.to_parquet(
            Config.TEST_GROUPED_PATH, index=False, engine="pyarrow"
        )
        del df_test

    # 3. Create Datasets & Loaders
    print("Creating Datasets...")
    train_dataset_tagger = TaggerDataset(df_train_grouped, vocab)
    val_dataset_tagger = TaggerDataset(df_val_grouped, vocab)

    train_dataset_seq2seq = Seq2SeqDataset(df_train_seq2seq, vocab)
    val_dataset_seq2seq = Seq2SeqDataset(df_val_seq2seq, vocab)

    test_dataset = SubmissionDataset(df_test_grouped, vocab)

    return (
        vocab,
        kb,
        train_dataset_tagger,
        val_dataset_tagger,
        train_dataset_seq2seq,
        val_dataset_seq2seq,
        test_dataset,
    )
