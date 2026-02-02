import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from library.config import Config
from library.utils import seed_everything

# Regex to identify "hard" PLAIN tokens (digits, uppercase, symbols)
# Matches any digit, uppercase letter, or non-word character (excluding space)
HARD_PLAIN_PATTERN = re.compile(r"[0-9A-Z]|[^\w\s]")


def prepare_router_data(split: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads, filters, and groups data for the Router model.
    Applies Hard-Negative Mining on the training set.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"router_{split}_processed.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached router data for {split} from {cache_path}...")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Load Raw Data
    if split == "train":
        file_path = Config.TRAIN_FILE
    elif split == "val":
        file_path = Config.VAL_FILE
    elif split == "test":
        file_path = Config.TEST_FILE
    else:
        raise ValueError(f"Unknown split: {split}")

    print(f"Processing router data for {split} from {file_path}...")
    df = pd.read_csv(
        file_path, keep_default_na=False, dtype={"sentence_id": int, "token_id": int}
    )

    # 3. Group by Sentence
    # We need to aggregate tokens and labels into lists
    # For test set, we don't have 'class' or 'after'
    is_test = split == "test"

    if is_test:
        # Just aggregate tokens
        grouped = (
            df.groupby("sentence_id", sort=False)
            .agg({"before": list, "id": list})
            .reset_index()
        )
        grouped.rename(columns={"before": "tokens", "id": "token_ids"}, inplace=True)
        grouped["labels"] = None
    else:
        # Aggregate tokens and classes
        grouped = (
            df.groupby("sentence_id", sort=False)
            .agg({"before": list, "class": list})
            .reset_index()
        )
        grouped.rename(columns={"before": "tokens", "class": "labels"}, inplace=True)

    # 4. Filter / Hard-Negative Mining (Train only)
    if split == "train":
        print("Applying Hard-Negative Mining...")

        def filter_logic(row):
            labels = row["labels"]
            tokens = row["tokens"]

            # Check if sentence has any non-PLAIN token
            has_non_plain = any(l != "PLAIN" for l in labels)
            if has_non_plain:
                return True

            # If all PLAIN, check if "hard" (digits, caps, symbols)
            # We check if ANY token in the sentence is "hard"
            # Optimization: Join and regex search might be faster than iterating
            # But iterating is safer for token-level logic.
            # Let's check if any token matches the pattern.
            is_hard = any(HARD_PLAIN_PATTERN.search(t) for t in tokens)
            if is_hard:
                return True

            # If easy PLAIN (lowercase words), downsample
            # Keep 1% of easy plain sentences to maintain baseline distribution
            return np.random.rand() < 0.01

        # Apply filter
        # Note: This can be slow. Vectorized approach is preferred if possible.
        # Construct boolean masks

        # Helper to check non-plain
        # We can't easily vectorize list column operations in pandas without apply
        # But apply on 500k rows is acceptable (seconds to minutes).
        mask = grouped.apply(filter_logic, axis=1)
        grouped = grouped[mask].reset_index(drop=True)
        print(f"Filtered training data to {len(grouped)} sentences.")

    # 5. Cache
    print(f"Saving router data to {cache_path}...")
    grouped.to_parquet(cache_path)

    return grouped


def prepare_generator_data(split: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads and processes data for the Generator model.
    Filters for Path B classes and constructs context-augmented input strings.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"generator_{split}_processed.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached generator data for {split} from {cache_path}...")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Load Raw Data
    if split == "train":
        file_path = Config.TRAIN_FILE
    elif split == "val":
        file_path = Config.VAL_FILE
    else:
        # Generator training data not needed for test split logic here
        # (Test inference is dynamic)
        raise ValueError(f"Generator data preparation only for train/val. Got {split}")

    print(f"Processing generator data for {split} from {file_path}...")
    df = pd.read_csv(
        file_path, keep_default_na=False, dtype={"sentence_id": int, "token_id": int}
    )

    # 3. Build Context Map
    # We need random access to sentence tokens to build context windows.
    # Grouping by sentence_id -> list of tokens
    print("Building sentence context map...")
    sentence_tokens = (
        df.groupby("sentence_id", sort=False)["before"].apply(list).to_dict()
    )

    # 4. Filter for Path B Classes
    # We only train the generator on ambiguous classes
    print("Filtering for Path B classes...")
    path_b_mask = df["class"].isin(Config.PATH_B_CLASSES)
    target_df = df[path_b_mask].copy()

    if len(target_df) == 0:
        print("Warning: No Path B tokens found.")
        return pd.DataFrame(columns=["input_text", "target_text"])

    # 5. Construct Context-Augmented Inputs
    print("Constructing context-augmented inputs...")

    input_texts = []
    target_texts = []

    # Iterate over the filtered targets
    # Using itertuples is faster than iterrows
    for row in target_df.itertuples(index=False):
        s_id = row.sentence_id
        t_id = row.token_id
        label = (
            row._2
        )  # 'class' column is usually 3rd index (0=sent, 1=tok, 2=class) but let's be safe
        # Access by name is safer: row.class is invalid syntax, use getattr
        label = getattr(row, "class")
        target_val = row.after

        # Get full sentence context
        ctx_tokens = sentence_tokens[s_id]
        seq_len = len(ctx_tokens)

        # Define Window
        start = max(0, t_id - Config.CONTEXT_WINDOW_SIZE)
        end = min(seq_len, t_id + Config.CONTEXT_WINDOW_SIZE + 1)

        # Build String
        # Format: "[CLASS] left ... <extra_id_0> target <extra_id_1> ... right"

        # Left context
        left_ctx = " ".join(ctx_tokens[start:t_id])

        # Target placeholder (actual token text)
        target_token = ctx_tokens[t_id]

        # Right context
        right_ctx = " ".join(ctx_tokens[t_id + 1 : end])

        # Construct final string
        # We prepend the class label to give the model a hint about the semantic type
        input_str = (
            f"{label} {left_ctx} "
            f"{Config.TARGET_START_TOKEN} {target_token} {Config.TARGET_END_TOKEN} "
            f"{right_ctx}"
        )

        input_texts.append(input_str)
        target_texts.append(target_val)

    # Create Result DataFrame
    result_df = pd.DataFrame({"input_text": input_texts, "target_text": target_texts})

    # 6. Cache
    print(f"Saving generator data ({len(result_df)} samples) to {cache_path}...")
    result_df.to_parquet(cache_path)

    return result_df


class TextNormalizationRouterDataset(Dataset):
    """
    Dataset for the Router (Token Classification) model.
    Handles subword tokenization and label alignment.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        is_test: bool = False,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.label2id = Config.LABEL2ID

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = row["tokens"]  # List of strings

        # Tokenize
        # is_split_into_words=True indicates that the input is already pre-tokenized (split by space/rule)
        tokenized_inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=Config.ROUTER_MAX_LEN,
            padding="max_length",  # Pad here or in collator? Pad here for simplicity with standard loaders
            return_tensors="pt",
        )

        # Extract tensors
        input_ids = tokenized_inputs["input_ids"].squeeze(0)
        attention_mask = tokenized_inputs["attention_mask"].squeeze(0)

        item = {"input_ids": input_ids, "attention_mask": attention_mask}

        if not self.is_test:
            labels = row["labels"]  # List of strings
            word_ids = tokenized_inputs.word_ids(batch_index=0)

            # Align labels
            aligned_labels = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens (CLS, SEP, PAD) -> -100
                    aligned_labels.append(-100)
                elif word_idx != previous_word_idx:
                    # First subword of a token -> Assign label
                    label_str = labels[word_idx]
                    label_id = self.label2id.get(
                        label_str, self.label2id["PLAIN"]
                    )  # Fallback to PLAIN if issue
                    aligned_labels.append(label_id)
                else:
                    # Subsequent subwords -> -100 (ignore in loss)
                    aligned_labels.append(-100)
                previous_word_idx = word_idx

            item["labels"] = torch.tensor(aligned_labels, dtype=torch.long)

        return item


class TextNormalizationGeneratorDataset(Dataset):
    """
    Dataset for the Generator (Seq2Seq) model.
    """

    def __init__(self, data: pd.DataFrame, tokenizer: PreTrainedTokenizerBase):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_text = row["input_text"]
        target_text = row["target_text"]

        # Tokenize Input
        model_inputs = self.tokenizer(
            input_text,
            max_length=Config.GENERATOR_MAX_INPUT_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize Target
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_text,
                max_length=Config.GENERATOR_MAX_TARGET_LEN,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        target_ids = labels["input_ids"].squeeze(0)

        # Replace padding token id with -100 for loss calculation
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": target_ids,
        }
