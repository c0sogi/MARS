import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class CharTokenizer:
    """
    Character-level tokenizer for Seq2Seq models.
    Handles encoding of source and target strings into integer sequences.
    """

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "<SEP>"]
        self.pad_token_id = 0
        self.sos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.sep_token_id = 4

        # Initialize vocab with special tokens
        for idx, token in enumerate(self.special_tokens):
            self.char2idx[token] = idx
            self.idx2char[idx] = token

    def fit_on_texts(self, texts):
        """
        Builds vocabulary from a list of strings.
        """
        unique_chars = set()
        for text in texts:
            unique_chars.update(str(text))

        # Sort for determinism
        sorted_chars = sorted(list(unique_chars))

        start_idx = len(self.special_tokens)
        for i, char in enumerate(sorted_chars):
            idx = start_idx + i
            self.char2idx[char] = idx
            self.idx2char[idx] = char

    def encode(self, text, add_special_tokens=False):
        """
        Converts a string to a list of token IDs.
        """
        text = str(text)
        ids = [self.char2idx.get(c, self.unk_token_id) for c in text]
        if add_special_tokens:
            ids = [self.sos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        """
        Converts a list of token IDs back to a string.
        """
        chars = []
        for i in ids:
            if skip_special_tokens and i < len(self.special_tokens):
                continue
            chars.append(self.idx2char.get(i, ""))
        return "".join(chars)

    def save_vocab(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char2idx, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.char2idx = json.load(f)
        self.idx2char = {int(v): k for k, v in self.char2idx.items()}
        # Update special token IDs based on loaded vocab (though they should be static)
        self.pad_token_id = self.char2idx.get("<PAD>", 0)
        self.sos_token_id = self.char2idx.get("<SOS>", 1)
        self.eos_token_id = self.char2idx.get("<EOS>", 2)
        self.unk_token_id = self.char2idx.get("<UNK>", 3)
        self.sep_token_id = self.char2idx.get("<SEP>", 4)

    def __len__(self):
        return len(self.char2idx)


class NormalizationDataset(Dataset):
    """
    PyTorch Dataset for Text Normalization.
    Input: [Prev] <SEP> [Current] <SEP> [Next]
    Target: [Normalized]
    """

    def __init__(self, df, tokenizer, max_len=128, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        # Pre-extract columns to numpy arrays for faster access
        self.before = df["before"].astype(str).values
        self.prev = df["prev"].astype(str).values
        self.next = df["next"].astype(str).values

        if not self.is_test:
            self.after = df["after"].astype(str).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct Source Sequence: prev <SEP> current <SEP> next
        prev_tok = self.prev[idx]
        curr_tok = self.before[idx]
        next_tok = self.next[idx]

        # Encode parts
        prev_ids = self.tokenizer.encode(prev_tok)
        curr_ids = self.tokenizer.encode(curr_tok)
        next_ids = self.tokenizer.encode(next_tok)
        sep = [self.tokenizer.sep_token_id]

        # Combine: prev + SEP + curr + SEP + next
        # We truncate prev/next if too long, but usually they are single words
        src_ids = prev_ids + sep + curr_ids + sep + next_ids

        # Truncate to max_len
        if len(src_ids) > self.max_len:
            src_ids = src_ids[: self.max_len]

        # Pad Source
        pad_len = self.max_len - len(src_ids)
        src_ids = src_ids + [self.tokenizer.pad_token_id] * pad_len

        src_tensor = torch.tensor(src_ids, dtype=torch.long)

        if self.is_test:
            return src_tensor

        # Construct Target Sequence
        target_text = self.after[idx]
        tgt_ids = self.tokenizer.encode(target_text, add_special_tokens=True)

        # Truncate Target
        if len(tgt_ids) > self.max_len:
            tgt_ids = tgt_ids[: self.max_len - 1] + [self.tokenizer.eos_token_id]

        # Pad Target
        tgt_pad_len = self.max_len - len(tgt_ids)
        tgt_ids_padded = tgt_ids + [self.tokenizer.pad_token_id] * tgt_pad_len

        tgt_tensor = torch.tensor(tgt_ids_padded, dtype=torch.long)

        # Decoder Input (Shifted Right): <SOS> ...
        # Decoder Target: ... <EOS>
        # However, standard seq2seq usually takes:
        # Encoder Input: src
        # Decoder Input: tgt[:-1]
        # Target: tgt[1:]

        # Here we return the full padded target tensor.
        # The training loop is responsible for slicing input/target.

        return src_tensor, tgt_tensor


def _generate_context(df):
    """
    Helper to generate prev/next columns respecting sentence boundaries.
    """
    df["before"] = df["before"].fillna("").astype(str)

    s_ids = df["sentence_id"].values
    tokens = df["before"].values

    # Shift tokens
    prev_tokens = np.roll(tokens, 1)
    next_tokens = np.roll(tokens, -1)

    # Shift sentence IDs
    prev_s_ids = np.roll(s_ids, 1)
    next_s_ids = np.roll(s_ids, -1)

    # Handle boundaries
    prev_tokens[0] = "<START>"
    next_tokens[-1] = "<END>"

    # If sentence ID changed, prev is start
    start_mask = s_ids != prev_s_ids
    prev_tokens[start_mask] = "<START>"

    # If sentence ID changed, next is end
    end_mask = s_ids != next_s_ids
    next_tokens[end_mask] = "<END>"

    df["prev"] = prev_tokens
    df["next"] = next_tokens

    return df


def prepare_dataloaders(load_cached_data=True, debug=False):
    """
    Prepares DataLoaders for the Transformer model.
    Handles caching, filtering, and upsampling.
    """
    # Define cache paths
    os.makedirs(Config.TRANSFORMER_CACHE_DIR, exist_ok=True)

    train_cache_path = os.path.join(
        Config.TRANSFORMER_CACHE_DIR, "processed_train.parquet"
    )
    val_cache_path = os.path.join(Config.TRANSFORMER_CACHE_DIR, "processed_val.parquet")
    vocab_path = Config.VOCAB_PATH

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_cache_path)
        and os.path.exists(val_cache_path)
        and os.path.exists(vocab_path)
    )

    tokenizer = CharTokenizer()

    if load_cached_data and cache_exists and not debug:
        print("Loading Transformer data from cache...")
        df_train = pd.read_parquet(train_cache_path)
        df_val = pd.read_parquet(val_cache_path)
        tokenizer.load_vocab(vocab_path)
    else:
        print("Processing Transformer data from scratch...")

        # Load Metadata
        df_train_full = pd.read_csv(Config.TRAIN_DATA)
        df_val_full = pd.read_csv(Config.VAL_DATA)

        if debug:
            print(f"DEBUG mode: reducing dataset size to {Config.DEBUG_SIZE}")
            # Filter by sentence_id to keep context intact
            unique_sents = df_train_full["sentence_id"].unique()[
                : Config.DEBUG_SIZE // 10
            ]
            df_train_full = df_train_full[
                df_train_full["sentence_id"].isin(unique_sents)
            ].copy()

            unique_sents_val = df_val_full["sentence_id"].unique()[
                : Config.DEBUG_SIZE // 50
            ]
            df_val_full = df_val_full[
                df_val_full["sentence_id"].isin(unique_sents_val)
            ].copy()

        # 1. Generate Context (Prev/Next) BEFORE filtering
        print("Generating context...")
        df_train_full = _generate_context(df_train_full)
        df_val_full = _generate_context(df_val_full)

        # 2. Filter (Remove PLAIN and PUNCT)
        print("Filtering dataset (removing PLAIN/PUNCT)...")
        # Keep only semiotic classes
        exclude_classes = ["PLAIN", "PUNCT"]
        df_train = df_train_full[~df_train_full["class"].isin(exclude_classes)].copy()
        df_val = df_val_full[~df_val_full["class"].isin(exclude_classes)].copy()

        print(f"Filtered Train Size: {len(df_train)}")
        print(f"Filtered Val Size: {len(df_val)}")

        # 3. Upsampling (Train only)
        if not debug:
            print("Upsampling rare classes...")
            class_counts = df_train["class"].value_counts()

            # Determine target count from reference class
            ref_class = Config.REFERENCE_CLASS_FOR_UPSAMPLING
            if ref_class in class_counts:
                target_count = class_counts[ref_class]
            else:
                target_count = class_counts.max()  # Fallback

            dfs_to_concat = [df_train]

            for cls in Config.UPSAMPLE_CLASSES:
                if cls in class_counts:
                    count = class_counts[cls]
                    if count < target_count:
                        # Calculate how many samples to add
                        n_samples = target_count - count
                        subset = df_train[df_train["class"] == cls]
                        if len(subset) > 0:
                            upsampled_subset = subset.sample(
                                n=n_samples, replace=True, random_state=Config.SEED
                            )
                            dfs_to_concat.append(upsampled_subset)
                            # print(f"  Upsampled {cls}: +{n_samples} samples")

            df_train = pd.concat(dfs_to_concat, ignore_index=True)
            # Shuffle after concatenation
            df_train = df_train.sample(frac=1, random_state=Config.SEED).reset_index(
                drop=True
            )
            print(f"Train Size after Upsampling: {len(df_train)}")

        # 4. Fit Tokenizer
        print("Fitting tokenizer...")
        # Fit on both 'before' and 'after' text from the filtered training set
        all_text = pd.concat(
            [df_train["before"], df_train["after"], df_train["prev"], df_train["next"]]
        )
        tokenizer.fit_on_texts(all_text.astype(str).tolist())

        # 5. Cache
        if not debug:
            print("Saving processed data to cache...")
            df_train.to_parquet(train_cache_path)
            df_val.to_parquet(val_cache_path)
            tokenizer.save_vocab(vocab_path)

    # Create Datasets
    train_dataset = NormalizationDataset(
        df_train, tokenizer, max_len=Config.MAX_SEQ_LEN
    )
    val_dataset = NormalizationDataset(df_val, tokenizer, max_len=Config.MAX_SEQ_LEN)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, tokenizer
