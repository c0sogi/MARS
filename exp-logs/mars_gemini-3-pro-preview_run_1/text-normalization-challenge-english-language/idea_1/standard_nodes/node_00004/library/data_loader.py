import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from collections import Counter
from library.config import Config
from library.utils import get_logger, timer

logger = get_logger("data_loader")


class Vocabulary:
    def __init__(self):
        self.token2id = {
            Config.PAD_TOKEN: Config.PAD_TOKEN_ID,
            Config.UNK_TOKEN: Config.UNK_TOKEN_ID,
        }
        self.id2token = {
            Config.PAD_TOKEN_ID: Config.PAD_TOKEN,
            Config.UNK_TOKEN_ID: Config.UNK_TOKEN,
        }
        self.class2id = {Config.PAD_TOKEN: Config.PAD_TOKEN_ID}
        self.id2class = {Config.PAD_TOKEN_ID: Config.PAD_TOKEN}
        self.token_counts = Counter()
        self.class_counts = Counter()

    def fit(self, token_lists, class_lists):
        """
        Builds vocabulary from lists of tokens and classes.
        """
        logger.info("Building vocabulary from training data...")

        # Count tokens
        for tokens in token_lists:
            self.token_counts.update(tokens)

        # Count classes
        for classes in class_lists:
            self.class_counts.update(classes)

        # Create token mappings (Top N frequent tokens)
        # Start index after special tokens
        current_idx = 2
        most_common_tokens = self.token_counts.most_common(Config.VOCAB_SIZE - 2)
        for token, _ in most_common_tokens:
            if token not in self.token2id:
                self.token2id[token] = current_idx
                self.id2token[current_idx] = token
                current_idx += 1

        # Create class mappings (All classes)
        # Start index after PAD (0)
        current_class_idx = 1
        for cls_name in sorted(self.class_counts.keys()):
            if cls_name not in self.class2id:
                self.class2id[cls_name] = current_class_idx
                self.id2class[current_class_idx] = cls_name
                current_class_idx += 1

        logger.info(
            f"Vocabulary built. Tokens: {len(self.token2id)}, Classes: {len(self.class2id)}"
        )

    def save(self, directory):
        """
        Saves vocabulary to parquet files to avoid pickle.
        """
        os.makedirs(directory, exist_ok=True)

        # Save Tokens
        tokens_df = pd.DataFrame(
            [{"id": k, "token": v} for k, v in self.id2token.items()]
        )
        tokens_df.to_parquet(
            os.path.join(directory, "vocab_tokens.parquet"), index=False
        )

        # Save Classes
        classes_df = pd.DataFrame(
            [{"id": k, "class": v} for k, v in self.id2class.items()]
        )
        classes_df.to_parquet(
            os.path.join(directory, "vocab_classes.parquet"), index=False
        )
        logger.info(f"Vocabulary saved to {directory}")

    def load(self, directory):
        """
        Loads vocabulary from parquet files.
        """
        token_path = os.path.join(directory, "vocab_tokens.parquet")
        class_path = os.path.join(directory, "vocab_classes.parquet")

        if not os.path.exists(token_path) or not os.path.exists(class_path):
            raise FileNotFoundError("Vocabulary files not found.")

        tokens_df = pd.read_parquet(token_path)
        self.id2token = dict(zip(tokens_df["id"], tokens_df["token"]))
        self.token2id = {v: k for k, v in self.id2token.items()}

        classes_df = pd.read_parquet(class_path)
        self.id2class = dict(zip(classes_df["id"], classes_df["class"]))
        self.class2id = {v: k for k, v in self.id2class.items()}

        logger.info(
            f"Vocabulary loaded. Tokens: {len(self.token2id)}, Classes: {len(self.class2id)}"
        )

    def encode_tokens(self, tokens):
        return [self.token2id.get(t, Config.UNK_TOKEN_ID) for t in tokens]

    def encode_classes(self, classes):
        # Default to 0 (PAD) if class unknown, though unlikely in train
        return [self.class2id.get(c, 0) for c in classes]


def preprocess_grouped_data(source_path, save_path, load_cached=True, is_test=False):
    """
    Groups raw token-level CSV data into sentence-level sequences.
    Caches the result as a Parquet file.
    """
    if load_cached and os.path.exists(save_path):
        logger.info(f"Loading cached grouped data from {save_path}")
        return pd.read_parquet(save_path)

    logger.info(f"Processing raw data from {source_path}...")

    # Load raw csv
    # keep_default_na=False is crucial for text data to preserve "null", "NaN" strings
    df = pd.read_csv(source_path, dtype=str, keep_default_na=False)

    # Ensure columns are strings
    df["before"] = df["before"].astype(str)
    if "class" in df.columns:
        df["class"] = df["class"].astype(str)
    if "after" in df.columns:
        df["after"] = df["after"].astype(str)

    # Group by sentence_id
    # We aggregate into lists.
    agg_dict = {"before": list, "id": list}  # We need this for submission mapping

    if not is_test:
        agg_dict["class"] = list
        agg_dict["after"] = list

    # Grouping
    # Using sort_values + groupby is often efficient for large data
    df = df.sort_values(["sentence_id", "token_id"])
    grouped_df = df.groupby("sentence_id", as_index=False).agg(agg_dict)

    # Save to cache
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    grouped_df.to_parquet(save_path, index=False)
    logger.info(f"Grouped data saved to {save_path}")

    return grouped_df


def build_knowledge_base(df_train, save_path, load_cached=True):
    """
    Constructs a deterministic lookup table: (raw_token, class) -> normalized_text.
    Selects the most frequent normalization for conflicts.
    """
    if load_cached and os.path.exists(save_path):
        logger.info(f"Loading cached Knowledge Base from {save_path}")
        kb_df = pd.read_parquet(save_path)
        # Convert back to dict for O(1) lookup
        # Key: "token|class", Value: "after"
        kb = {}
        for _, row in kb_df.iterrows():
            kb[(row["token"], row["class"])] = row["after"]
        return kb

    logger.info("Building Knowledge Base from training data...")

    # We need to iterate over the grouped dataframe or use the raw one.
    # Since we have the grouped one loaded in memory usually, we can iterate it,
    # but it's flatter to use raw. However, to avoid reloading raw, we iterate the lists.
    # Actually, constructing from the raw CSV logic is simpler if we had it.
    # But here we likely have the grouped DF. Let's iterate the grouped DF.

    counts = {}  # (token, class, after) -> count

    # Iterate efficiently
    # Flattening the lists is one way, but might duplicate memory.
    # Let's just loop.
    for _, row in df_train.iterrows():
        befores = row["before"]
        classes = row["class"]
        afters = row["after"]

        for b, c, a in zip(befores, classes, afters):
            key = (b, c, a)
            counts[key] = counts.get(key, 0) + 1

    # Resolve conflicts by picking max count
    # Structure: (token, class) -> {after: count, after2: count}
    conflict_resolver = {}
    for (b, c, a), count in counts.items():
        if (b, c) not in conflict_resolver:
            conflict_resolver[(b, c)] = (a, count)
        else:
            current_best_a, current_best_count = conflict_resolver[(b, c)]
            if count > current_best_count:
                conflict_resolver[(b, c)] = (a, count)

    # Final KB
    kb = {k: v[0] for k, v in conflict_resolver.items()}

    # Save
    # We save as a DataFrame with columns: token, class, after
    kb_list = [{"token": k[0], "class": k[1], "after": v} for k, v in kb.items()]
    kb_df = pd.DataFrame(kb_list)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    kb_df.to_parquet(save_path, index=False)
    logger.info(f"Knowledge Base built with {len(kb)} entries and saved to {save_path}")

    return kb


class TextNormalizationDataset(Dataset):
    def __init__(self, df, vocab, max_len=128, is_test=False):
        self.df = df
        self.vocab = vocab
        self.max_len = max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = row["before"]

        # Truncate if necessary
        if len(tokens) > self.max_len:
            tokens = tokens[: self.max_len]

        # Encode tokens
        input_ids = self.vocab.encode_tokens(tokens)
        seq_len = len(input_ids)

        # Padding
        pad_len = self.max_len - seq_len
        input_ids = input_ids + [Config.PAD_TOKEN_ID] * pad_len
        attention_mask = [1] * seq_len + [0] * pad_len

        # Prepare output tensors
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

        # Handle Labels (Classes)
        if not self.is_test:
            classes = row["class"]
            if len(classes) > self.max_len:
                classes = classes[: self.max_len]

            class_ids = self.vocab.encode_classes(classes)
            # Pad labels with PAD_TOKEN_ID (usually 0) or -100 if using CrossEntropy ignore_index
            # Here we use PAD_TOKEN_ID (0) as defined in config
            class_ids = class_ids + [Config.PAD_TOKEN_ID] * pad_len
            item["labels"] = torch.tensor(class_ids, dtype=torch.long)

        # Handle IDs for submission reconstruction
        # We pass the sentence_id and let the loop handle token indices,
        # or we pass the padded list of ID strings?
        # Passing strings in DataLoader collate is messy.
        # We will pass the sentence_id (int) and length.
        item["sentence_id"] = int(row["sentence_id"])
        item["seq_len"] = seq_len

        return item


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Main function to prepare data and return dataloaders.
    """
    with timer("Data Preparation", logger):
        # 1. Define Paths for Cache
        train_grouped_path = os.path.join(Config.WORKING_DIR, "train_grouped.parquet")
        val_grouped_path = os.path.join(Config.WORKING_DIR, "val_grouped.parquet")
        test_grouped_path = os.path.join(Config.WORKING_DIR, "test_grouped.parquet")
        kb_path = os.path.join(Config.WORKING_DIR, "knowledge_base.parquet")

        # 2. Preprocess/Load Grouped Data
        df_train = preprocess_grouped_data(
            Config.TRAIN_DATA_PATH, train_grouped_path, load_cached_data, is_test=False
        )
        df_val = preprocess_grouped_data(
            Config.VAL_DATA_PATH, val_grouped_path, load_cached_data, is_test=False
        )
        df_test = preprocess_grouped_data(
            Config.TEST_DATA_PATH, test_grouped_path, load_cached_data, is_test=True
        )

        # Debugging: Subsample
        if debug or Config.MAX_TRAIN_SAMPLES:
            limit = Config.MAX_TRAIN_SAMPLES if Config.MAX_TRAIN_SAMPLES else 1000
            logger.info(f"Debug mode: Limiting train data to {limit} sentences.")
            df_train = df_train.head(limit)
            df_val = df_val.head(limit // 5)  # Keep val proportional roughly

        # 3. Build/Load Vocabulary
        vocab = Vocabulary()
        # Check if vocab files exist
        vocab_exists = os.path.exists(
            os.path.join(Config.WORKING_DIR, "vocab_tokens.parquet")
        ) and os.path.exists(os.path.join(Config.WORKING_DIR, "vocab_classes.parquet"))

        if load_cached_data and vocab_exists:
            vocab.load(Config.WORKING_DIR)
        else:
            vocab.fit(df_train["before"].tolist(), df_train["class"].tolist())
            vocab.save(Config.WORKING_DIR)

        # 4. Build/Load Knowledge Base
        kb = build_knowledge_base(df_train, kb_path, load_cached_data)

        # 5. Create Datasets
        train_dataset = TextNormalizationDataset(
            df_train, vocab, Config.MAX_LEN, is_test=False
        )
        val_dataset = TextNormalizationDataset(
            df_val, vocab, Config.MAX_LEN, is_test=False
        )
        test_dataset = TextNormalizationDataset(
            df_test, vocab, Config.MAX_LEN, is_test=True
        )

        # 6. Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        logger.info(
            f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
        )

        return train_loader, val_loader, test_loader, vocab, kb
