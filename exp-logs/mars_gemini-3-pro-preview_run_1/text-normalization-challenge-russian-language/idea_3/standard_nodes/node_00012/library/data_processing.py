import os
import re
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict, Counter

from library.config import Config
from library.utils import load_metadata, save_npy, load_npy, get_artifact_path

# --- Tokenizer ---


class CharTokenizer:
    """
    Character-level tokenizer for the neural model.
    Handles mapping between characters and integer IDs.
    """

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.special_tokens = ["<pad>", "<sos>", "<eos>", "<sep>", "<unk>"]

        # Initialize with special tokens
        for i, token in enumerate(self.special_tokens):
            self.char_to_id[token] = i
            self.id_to_char[i] = token

    def fit(self, texts, vocab_size=None):
        """
        Builds vocabulary from a list of texts.
        """
        counter = Counter()
        for text in texts:
            counter.update(str(text))

        # Sort by frequency
        most_common = counter.most_common()

        # Add characters to vocab
        current_id = len(self.special_tokens)
        for char, _ in most_common:
            if vocab_size and current_id >= vocab_size:
                break
            if char not in self.char_to_id:
                self.char_to_id[char] = current_id
                self.id_to_char[current_id] = char
                current_id += 1

    def encode(self, text, add_special_tokens=False):
        """
        Converts a string to a list of IDs.
        """
        ids = []
        if add_special_tokens:
            ids.append(self.char_to_id["<sos>"])

        for char in str(text):
            ids.append(self.char_to_id.get(char, self.char_to_id["<unk>"]))

        if add_special_tokens:
            ids.append(self.char_to_id["<eos>"])
        return ids

    def decode(self, ids):
        """
        Converts a list of IDs back to a string.
        """
        chars = []
        for i in ids:
            token = self.id_to_char.get(i, "")
            if token not in self.special_tokens:
                chars.append(token)
        return "".join(chars)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char_to_id, f, ensure_ascii=False, indent=2)

    def load(self, path):
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            self.char_to_id = json.load(f)
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}
        return True

    def __len__(self):
        return len(self.char_to_id)

    @property
    def pad_token_id(self):
        return self.char_to_id["<pad>"]

    @property
    def sep_token_id(self):
        return self.char_to_id["<sep>"]


# --- Data Loading & Grouping ---


def load_and_group_data(split, load_cached_data=True):
    """
    Loads data from metadata CSVs and groups tokens into sentences.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        pd.DataFrame: DataFrame where each row is a sentence with lists of tokens.
    """
    # Define cache path using the config hash to ensure versioning
    cache_filename = f"{split}_sequences.parquet"
    cache_path = get_artifact_path(cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        # print(f"Loading grouped {split} data from cache: {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass  # Fallback to processing

    # 2. Process from scratch
    # print(f"Processing {split} data from metadata...")
    df_raw = load_metadata(split)

    # Ensure columns are strings
    if "before" in df_raw.columns:
        df_raw["before"] = df_raw["before"].fillna("").astype(str)
    if "after" in df_raw.columns:
        df_raw["after"] = df_raw["after"].fillna("").astype(str)
    if "class" in df_raw.columns:
        df_raw["class"] = df_raw["class"].fillna("UNKNOWN").astype(str)

    # Group by sentence_id
    # We aggregate into lists. This preserves the order (token_id) assuming input is sorted.
    # The metadata loading does not guarantee sort, so we sort first.
    if "token_id" in df_raw.columns:
        df_raw["token_id"] = pd.to_numeric(df_raw["token_id"])
        df_raw = df_raw.sort_values(["sentence_id", "token_id"])

    agg_dict = {"before": list}
    if "after" in df_raw.columns:
        agg_dict["after"] = list
    if "class" in df_raw.columns:
        agg_dict["class"] = list
    if "token_id" in df_raw.columns:
        agg_dict["token_id"] = list

    df_grouped = df_raw.groupby("sentence_id").agg(agg_dict).reset_index()

    # 3. Save to cache
    # print(f"Saving grouped {split} data to cache: {cache_path}")
    df_grouped.to_parquet(cache_path)

    return df_grouped


# --- Symbolic Statistics (N-grams) ---


def build_ngram_statistics(train_grouped_df, load_cached_data=True):
    """
    Computes hierarchical N-gram statistics (Trigram, Bigram, Unigram) from training data.

    Returns:
        dict: A dictionary containing 'unigram', 'bigram', 'trigram' lookup tables.
    """
    cache_name = "ngram_stats.npy"

    # 1. Try load
    if load_cached_data:
        data = load_npy(cache_name)
        if data is not None:
            # print("Loaded N-gram statistics from cache.")
            return data

    # 2. Compute
    # print("Computing N-gram statistics...")

    # Counters for frequency
    # Unigram: curr -> after
    unigram_counts = defaultdict(Counter)
    # Bigram: (prev, curr) -> after
    bigram_counts = defaultdict(Counter)
    # Trigram: (prev, curr, next) -> after
    trigram_counts = defaultdict(Counter)

    # Iterate over sentences
    # Using itertuples for speed
    for row in train_grouped_df.itertuples(index=False):
        befores = row.before
        afters = row.after
        n = len(befores)

        for i in range(n):
            curr_w = befores[i]
            target = afters[i]

            # Unigram
            unigram_counts[curr_w][target] += 1

            # Contexts
            prev_w = befores[i - 1] if i > 0 else "<start>"
            next_w = befores[i + 1] if i < n - 1 else "<end>"

            # Bigram (Prev, Curr)
            bigram_counts[(prev_w, curr_w)][target] += 1

            # Trigram (Prev, Curr, Next)
            trigram_counts[(prev_w, curr_w, next_w)][target] += 1

    # Convert counters to deterministic maps (taking the most frequent)
    def condense(counts_dict):
        result = {}
        for key, counter in counts_dict.items():
            result[key] = counter.most_common(1)[0][0]
        return result

    stats = {
        "unigram": condense(unigram_counts),
        "bigram": condense(bigram_counts),
        "trigram": condense(trigram_counts),
    }

    # 3. Save
    save_npy(stats, cache_name)

    return stats


# --- Neural Dataset ---


class NeuralDataset(Dataset):
    """
    Dataset for the Character-Level Transformer.
    Extracts contexts and targets for tokens requiring neural normalization.
    """

    def __init__(
        self, grouped_df, tokenizer, mode="train", context_window=1, sample_ratio=0.01
    ):
        """
        Args:
            grouped_df (pd.DataFrame): Data grouped by sentence.
            tokenizer (CharTokenizer): Fitted tokenizer.
            mode (str): 'train', 'val', or 'test'.
            context_window (int): Number of tokens to left/right to include.
            sample_ratio (float): Ratio of PLAIN tokens to include in training (for robustness).
        """
        self.tokenizer = tokenizer
        self.mode = mode
        self.context_window = context_window
        self.samples = []

        # Pre-compile regex for digit check
        self.digit_pattern = re.compile(r"\d")

        # Prepare samples
        self._prepare_samples(grouped_df, sample_ratio)

    def _prepare_samples(self, df, sample_ratio):
        # Rename 'class' column to 'class_' if it exists to avoid itertuples issues
        if "class" in df.columns:
            df = df.rename(columns={"class": "class_"})

        # Iterate over sentences
        for row in df.itertuples(index=False):
            befores = row.before

            # Targets and classes only exist in train/val
            has_targets = hasattr(row, "after")
            afters = row.after if has_targets else None
            classes = getattr(row, "class_", None)

            n_tokens = len(befores)

            for i in range(n_tokens):
                token_text = befores[i]

                # --- Filtering Logic ---
                should_include = False

                if self.mode == "test":
                    # In test, we don't have classes.
                    # We include tokens that look like they need neural help (digits, symbols)
                    # The router in inference will decide whether to use the prediction,
                    # but the dataset should provide the option.
                    if self.digit_pattern.search(token_text):
                        should_include = True
                    # Also include if it looks like a symbol or special format?
                    # For simplicity in this task, we focus on digits for the neural part.

                else:
                    # Train/Val
                    if classes is not None:
                        token_class = classes[i]
                    else:
                        token_class = "PLAIN"

                    # Always include semiotic classes (non-PLAIN, non-PUNCT)
                    if token_class not in ["PLAIN", "PUNCT"]:
                        should_include = True
                    # Include PLAIN/PUNCT if they contain digits (e.g. "H2O", "3D")
                    elif self.digit_pattern.search(token_text):
                        should_include = True
                    # Random sampling of PLAIN text to teach the model identity/copying
                    elif self.mode == "train" and random.random() < sample_ratio:
                        should_include = True

                if should_include:
                    # Construct Context
                    # Left context
                    left_ctx = []
                    for k in range(1, self.context_window + 1):
                        if i - k >= 0:
                            left_ctx.insert(0, befores[i - k])
                        else:
                            break  # Or insert padding token? Usually just empty is fine for concat

                    # Right context
                    right_ctx = []
                    for k in range(1, self.context_window + 1):
                        if i + k < n_tokens:
                            right_ctx.append(befores[i + k])
                        else:
                            break

                    target_text = afters[i] if has_targets else ""

                    # Store sample
                    # We store the raw strings and tokenize on the fly or pre-tokenize?
                    # Tokenizing on fly saves memory.
                    self.samples.append(
                        {
                            "left": " ".join(left_ctx),
                            "center": token_text,
                            "right": " ".join(right_ctx),
                            "target": target_text,
                            "original_idx": (
                                row.sentence_id,
                                i,
                            ),  # Useful for reconstruction if needed
                        }
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Format Input: "left <sep> center <sep> right"
        # Note: We use the tokenizer's special tokens

        # We construct the string representation first
        # Ideally: [Left Chars] <sep> [Center Chars] <sep> [Right Chars]
        # But CharTokenizer encodes a single string.
        # We will manually assemble the IDs.

        sep_id = self.tokenizer.sep_token_id

        left_ids = self.tokenizer.encode(sample["left"])
        center_ids = self.tokenizer.encode(sample["center"])
        right_ids = self.tokenizer.encode(sample["right"])

        # Input IDs: left + sep + center + sep + right
        input_ids = left_ids + [sep_id] + center_ids + [sep_id] + right_ids
        if len(input_ids) > Config.MAX_SEQ_LEN:
            input_ids = input_ids[: Config.MAX_SEQ_LEN]

        # Target IDs
        if self.mode != "test":
            target_ids = self.tokenizer.encode(
                sample["target"], add_special_tokens=True
            )
            if len(target_ids) > Config.MAX_SEQ_LEN:
                target_ids = target_ids[: Config.MAX_SEQ_LEN]
        else:
            target_ids = []  # Dummy for test

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "raw_text": sample["center"],
            "target_text": sample["target"],
        }

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to pad sequences.
        """
        input_ids = [item["input_ids"] for item in batch]
        target_ids = [item["target_ids"] for item in batch]
        raw_texts = [item["raw_text"] for item in batch]
        target_texts = [item["target_text"] for item in batch]

        # Pad inputs
        # We need a pad_token_id.
        # Assuming tokenizer is available in scope or passed?
        # Standard practice is to use nn.utils.rnn.pad_sequence, default pad is 0.
        # We need to know the specific pad ID.
        # Let's assume 0 is pad (CharTokenizer init puts <pad> first).

        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=0
        )
        padded_targets = torch.nn.utils.rnn.pad_sequence(
            target_ids, batch_first=True, padding_value=0
        )

        # Create attention masks (1 for real token, 0 for pad)
        attention_mask = (padded_inputs != 0).long()

        return {
            "input_ids": padded_inputs,
            "attention_mask": attention_mask,
            "target_ids": padded_targets,
            "raw_texts": raw_texts,
            "target_texts": target_texts,
        }


def get_tokenizer(train_grouped_df=None, load_cached=True):
    """
    Gets or creates a tokenizer.
    """
    tokenizer_path = get_artifact_path("char_tokenizer.json")
    tokenizer = CharTokenizer()

    if load_cached and os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
    elif train_grouped_df is not None:
        # Collect all text to fit
        all_text = []
        # Sample to avoid OOM if dataset is huge, but char vocab is small so iterating all is fine for chars
        # Just concat a subset of 'before' and 'after'
        sample_df = train_grouped_df.sample(
            n=min(len(train_grouped_df), 50000), random_state=42
        )
        for row in sample_df.itertuples(index=False):
            all_text.extend(row.before)
            all_text.extend(row.after)

        tokenizer.fit(all_text, vocab_size=Config.VOCAB_SIZE)
        tokenizer.save(tokenizer_path)

    return tokenizer
