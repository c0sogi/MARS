import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from library.config import Config
from library.preprocessing import (
    build_vocabularies,
    process_tagger_data,
    process_seq2seq_data,
)

# ==========================================
# Dataset Classes
# ==========================================


class TaggerDataset(Dataset):
    """
    Dataset for the Multi-Granularity Bi-LSTM Tagger.
    Input: Grouped sentences with Word, Char, and BPE features.
    """

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Convert lists to tensors
        word_ids = torch.tensor(row["word_id"], dtype=torch.long)
        label_ids = torch.tensor(row["label_id"], dtype=torch.long)

        # Char IDs is a list of lists. Convert to list of tensors for collation
        char_ids = [torch.tensor(x, dtype=torch.long) for x in row["char_ids"]]

        # BPE IDs is a list of lists.
        bpe_ids = [torch.tensor(x, dtype=torch.long) for x in row["bpe_ids"]]

        # ID is a list of strings
        ids = row["id"]

        return {
            "word_ids": word_ids,
            "char_ids": char_ids,
            "bpe_ids": bpe_ids,
            "label_ids": label_ids,
            "ids": ids,
        }


class Seq2SeqDataset(Dataset):
    """
    Dataset for the Transformer Seq2Seq Fallback.
    Input: Source Chars, Class ID.
    Target: Target Chars.
    """

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        src_ids = torch.tensor(row["src_ids"], dtype=torch.long)
        tgt_ids = torch.tensor(row["tgt_ids"], dtype=torch.long)
        class_id = torch.tensor(row["class_id"], dtype=torch.long)

        return {"src_ids": src_ids, "tgt_ids": tgt_ids, "class_id": class_id}


# ==========================================
# Collate Functions
# ==========================================


class TaggerCollate:
    """
    Custom collate function for Tagger.
    Handles 2D padding for words/labels and 3D padding for chars/BPE.
    """

    def __init__(self, pad_idx=0):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # batch is a list of dicts

        # 1. Word IDs & Label IDs (2D Padding: Batch x Seq)
        word_seqs = [item["word_ids"] for item in batch]
        label_seqs = [item["label_ids"] for item in batch]

        word_ids_padded = pad_sequence(
            word_seqs, batch_first=True, padding_value=self.pad_idx
        )
        label_ids_padded = pad_sequence(
            label_seqs, batch_first=True, padding_value=self.pad_idx
        )

        # Create Mask (Batch x Seq)
        mask = word_ids_padded != self.pad_idx

        batch_size = len(batch)
        max_seq_len = word_ids_padded.size(1)

        # 2. Char IDs (3D Padding: Batch x Seq x Char)
        # Determine max char length in this batch
        max_char_len = 0
        for item in batch:
            for token_chars in item["char_ids"]:
                if len(token_chars) > max_char_len:
                    max_char_len = len(token_chars)

        # Initialize 3D tensor
        char_ids_padded = torch.full(
            (batch_size, max_seq_len, max_char_len), self.pad_idx, dtype=torch.long
        )

        for i, item in enumerate(batch):
            for j, token_chars in enumerate(item["char_ids"]):
                length = len(token_chars)
                if length > 0:
                    char_ids_padded[i, j, :length] = token_chars

        # 3. BPE IDs (3D Padding: Batch x Seq x BPE)
        max_bpe_len = 0
        for item in batch:
            for token_bpe in item["bpe_ids"]:
                if len(token_bpe) > max_bpe_len:
                    max_bpe_len = len(token_bpe)

        bpe_ids_padded = torch.full(
            (batch_size, max_seq_len, max_bpe_len), self.pad_idx, dtype=torch.long
        )

        for i, item in enumerate(batch):
            for j, token_bpe in enumerate(item["bpe_ids"]):
                length = len(token_bpe)
                if length > 0:
                    bpe_ids_padded[i, j, :length] = token_bpe

        # 4. IDs (List of Lists, no padding needed, just aggregation)
        ids = [item["ids"] for item in batch]

        return {
            "word_ids": word_ids_padded,
            "char_ids": char_ids_padded,
            "bpe_ids": bpe_ids_padded,
            "label_ids": label_ids_padded,
            "mask": mask,
            "ids": ids,
        }


class Seq2SeqCollate:
    """
    Custom collate for Seq2Seq.
    Handles 2D padding for source and target sequences.
    """

    def __init__(self, pad_idx=0):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        src_seqs = [item["src_ids"] for item in batch]
        tgt_seqs = [item["tgt_ids"] for item in batch]
        class_ids = [item["class_id"] for item in batch]

        src_padded = pad_sequence(
            src_seqs, batch_first=True, padding_value=self.pad_idx
        )
        tgt_padded = pad_sequence(
            tgt_seqs, batch_first=True, padding_value=self.pad_idx
        )
        class_ids_tensor = torch.stack(class_ids)

        return {
            "src_ids": src_padded,
            "tgt_ids": tgt_padded,
            "class_id": class_ids_tensor,
        }


# ==========================================
# Loader Factory Functions
# ==========================================


def get_tagger_loaders(debug=Config.DEBUG, load_cached=True):
    """
    Prepares DataLoaders for the Tagger model.

    Returns:
        train_loader, val_loader, test_loader,
        word_vocab, char_vocab, class_vocab, bpe_tokenizer
    """
    # 1. Check Vocab Cache
    vocab_cached = (
        os.path.exists(Config.VOCAB_WORDS_PATH)
        and os.path.exists(Config.VOCAB_CHARS_PATH)
        and os.path.exists(Config.VOCAB_CLASSES_PATH)
        and os.path.exists(f"{Config.VOCAB_BPE_MODEL_PATH}.model")
    )

    # 2. Load Raw Data if needed (for vocab build or data processing)
    # We only load raw train data if vocab is missing OR train data cache is missing
    need_raw_train = not vocab_cached or (
        not os.path.exists(Config.TRAIN_TAGGER_DATA_PATH)
    )

    df_train = None
    if need_raw_train and not load_cached:
        # If forcing no cache, we must load
        df_train = pd.read_csv(Config.TRAIN_FILE)
    elif need_raw_train and load_cached:
        # Check if we really need to load raw
        # If vocab is missing -> need raw
        # If vocab exists but data cache missing -> need raw
        if not vocab_cached or not os.path.exists(Config.TRAIN_TAGGER_DATA_PATH):
            print("Loading raw training data for initialization...")
            df_train = pd.read_csv(Config.TRAIN_FILE)

    # 3. Build/Load Vocabs
    word_vocab, char_vocab, class_vocab, bpe_tokenizer = build_vocabularies(
        df_train, load_cached=load_cached
    )

    # 4. Process Data
    # Train
    train_grouped = process_tagger_data(
        df_train,
        word_vocab,
        char_vocab,
        class_vocab,
        bpe_tokenizer,
        mode="train",
        load_cached=load_cached,
    )

    # Val
    # Load raw val only if cache missing
    if not os.path.exists(Config.VAL_TAGGER_DATA_PATH) or not load_cached:
        df_val = pd.read_csv(Config.VAL_FILE)
    else:
        df_val = None  # Process function will load from cache

    val_grouped = process_tagger_data(
        df_val,
        word_vocab,
        char_vocab,
        class_vocab,
        bpe_tokenizer,
        mode="val",
        load_cached=load_cached,
    )

    # Test
    # Load raw test only if cache missing
    if not os.path.exists(Config.TEST_TAGGER_DATA_PATH) or not load_cached:
        df_test = pd.read_csv(Config.TEST_FILE)
    else:
        df_test = None

    test_grouped = process_tagger_data(
        df_test,
        word_vocab,
        char_vocab,
        class_vocab,
        bpe_tokenizer,
        mode="test",
        load_cached=load_cached,
    )

    # 5. Debug Subsampling
    if debug:
        print(f"DEBUG Mode: Subsampling to {Config.MAX_DEBUG_SAMPLES} samples.")
        train_grouped = train_grouped.head(Config.MAX_DEBUG_SAMPLES)
        val_grouped = val_grouped.head(Config.MAX_DEBUG_SAMPLES // 5)
        test_grouped = test_grouped.head(Config.MAX_DEBUG_SAMPLES // 5)

    # 6. Create Datasets
    train_dataset = TaggerDataset(train_grouped)
    val_dataset = TaggerDataset(val_grouped)
    test_dataset = TaggerDataset(test_grouped)

    # 7. Create Loaders
    collate_fn = TaggerCollate(pad_idx=Config.PAD_IDX)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        word_vocab,
        char_vocab,
        class_vocab,
        bpe_tokenizer,
    )


def get_seq2seq_loaders(debug=Config.DEBUG, load_cached=True):
    """
    Prepares DataLoaders for the Seq2Seq model.

    Returns:
        train_loader, val_loader, char_vocab, class_vocab
    """
    # 1. Vocabs (Must exist or be built via tagger flow usually, but we handle here)
    # We assume vocabs are built. If not, we might fail or need to load train.
    # To be safe, we try to load cached. If fail, we load raw train.

    vocab_cached = os.path.exists(Config.VOCAB_CHARS_PATH) and os.path.exists(
        Config.VOCAB_CLASSES_PATH
    )

    df_train = None
    if not vocab_cached:
        print("Vocabs missing for Seq2Seq. Loading raw train to build...")
        df_train = pd.read_csv(Config.TRAIN_FILE)

    # We pass df_train to build_vocabularies, it handles the rest
    _, char_vocab, class_vocab, _ = build_vocabularies(df_train, load_cached=True)

    # 2. Process Data
    # Check cache for seq2seq data
    if not os.path.exists(Config.TRAIN_SEQ2SEQ_DATA_PATH) or not load_cached:
        if df_train is None:
            df_train = pd.read_csv(Config.TRAIN_FILE)
    else:
        df_train = None  # Will load from cache

    train_df = process_seq2seq_data(
        df_train, char_vocab, class_vocab, mode="train", load_cached=load_cached
    )

    if not os.path.exists(Config.VAL_SEQ2SEQ_DATA_PATH) or not load_cached:
        df_val = pd.read_csv(Config.VAL_FILE)
    else:
        df_val = None

    val_df = process_seq2seq_data(
        df_val, char_vocab, class_vocab, mode="val", load_cached=load_cached
    )

    # 3. Debug Subsampling
    if debug:
        if train_df is not None and len(train_df) > 0:
            train_df = train_df.head(Config.MAX_DEBUG_SAMPLES)
        if val_df is not None and len(val_df) > 0:
            val_df = val_df.head(Config.MAX_DEBUG_SAMPLES // 5)

    # 4. Create Datasets
    # Handle empty dataframes (e.g. if no changes found)
    if train_df is None or len(train_df) == 0:
        print("Warning: Train Seq2Seq dataframe is empty.")
        train_dataset = Seq2SeqDataset(
            pd.DataFrame(columns=["src_ids", "tgt_ids", "class_id"])
        )
    else:
        train_dataset = Seq2SeqDataset(train_df)

    if val_df is None or len(val_df) == 0:
        val_dataset = Seq2SeqDataset(
            pd.DataFrame(columns=["src_ids", "tgt_ids", "class_id"])
        )
    else:
        val_dataset = Seq2SeqDataset(val_df)

    # 5. Create Loaders
    collate_fn = Seq2SeqCollate(pad_idx=Config.PAD_IDX)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, char_vocab, class_vocab
