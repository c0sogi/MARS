import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from typing import List, Optional, Dict, Union
from library.config import Config
from library.utils import set_seed

# Set seed for reproducibility
set_seed(Config.SEED)


class TAPTDataset(Dataset):
    """
    Dataset for Task-Adaptive Pretraining (Masked Language Modeling).
    Accepts a list of text strings (Question + Context pairs).
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = Config.MAX_LEN,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize with truncation and padding
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


class QADataset(Dataset):
    """
    Dataset for Question Answering (Token Classification).
    Serves preprocessed features including input_ids, attention_mask, and labels.
    """

    def __init__(self, features_df: pd.DataFrame, is_test: bool = False):
        self.features = features_df
        self.is_test = is_test

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        row = self.features.iloc[idx]

        item = {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "index": idx,  # Useful for retrieving metadata during eval
        }

        if not self.is_test:
            # Labels: 0=O, 1=B-ANS, 2=I-ANS
            item["labels"] = torch.tensor(row["labels"], dtype=torch.long)

        return item


def prepare_tapt_data(
    tokenizer: PreTrainedTokenizerBase, load_cached_data: bool = True
) -> List[str]:
    """
    Prepares text data for TAPT by combining Question and Context.
    Combines Train, Val, and Test sets.

    Format: "Question </s> Context" (or tokenizer specific separator)
    """
    cache_path = os.path.join(Config.TAPT_CACHE_DIR, "corpus.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached TAPT data from {cache_path}")
        df = pd.read_parquet(cache_path)
        return df["text"].tolist()

    print("Preparing TAPT data from scratch...")

    # Load all available data
    dfs = []
    for path in [Config.TRAIN_META_PATH, Config.VAL_META_PATH, Config.TEST_META_PATH]:
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))

    full_df = pd.concat(dfs, ignore_index=True)

    # Create QC pairs: Question + Separator + Context
    # XLM-R uses </s> as separator. We explicitly add it for clarity in structure,
    # though tokenizer(q, c) handles it too. Here we prepare raw text for MLM.
    # We use a simple space concatenation with the separator token if known,
    # or just space if we rely on tokenizer later.
    # Strategy says: "concatenate the Question and Context (separated by the </s> token)"
    sep_token = tokenizer.sep_token if tokenizer.sep_token else "</s>"

    texts = []
    for _, row in full_df.iterrows():
        q = str(row["question"]).strip()
        c = str(row["context"]).strip()
        text = f"{q} {sep_token} {c}"
        texts.append(text)

    # Cache
    os.makedirs(Config.TAPT_CACHE_DIR, exist_ok=True)
    pd.DataFrame({"text": texts}).to_parquet(cache_path)
    print(f"Saved {len(texts)} TAPT examples to {cache_path}")

    return texts


def prepare_qa_features(
    tokenizer: PreTrainedTokenizerBase,
    examples_path: str,
    split_name: str,
    load_cached_data: bool = True,
    is_test: bool = False,
) -> pd.DataFrame:
    """
    Prepares sliding window features for QA.
    Implements 'Strict Containment' labeling.
    """
    cache_file = f"{split_name}_features.parquet"
    cache_path = os.path.join(Config.QA_CACHE_DIR, cache_file)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached QA features for {split_name} from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing QA features for {split_name}...")

    if not os.path.exists(examples_path):
        raise FileNotFoundError(f"Data file not found: {examples_path}")

    df = pd.read_csv(examples_path)

    # Lists to store features
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    all_offset_mappings = []
    all_example_ids = []
    all_context_texts = []
    all_sequence_ids = []  # To distinguish question vs context

    # Tokenization with sliding window
    # We tokenize Question + Context
    # stride and max_length from Config

    for idx, row in df.iterrows():
        question = str(row["question"]).strip()
        context = str(row["context"]).strip()
        example_id = row["id"]

        # Tokenize
        tokenized_inputs = tokenizer(
            question,
            context,
            truncation="only_second",  # Truncate context, keep question
            max_length=Config.MAX_LEN,
            stride=Config.DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="np",  # Get numpy arrays directly
        )

        sample_map = tokenized_inputs.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_inputs.pop("offset_mapping")
        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]

        # Determine labels if not test
        if not is_test:
            answer_text = str(row["answer_text"])
            start_char = row["answer_start"]
            end_char = start_char + len(answer_text)

        for i, offsets in enumerate(offset_mapping):
            # Sequence IDs: None for special tokens, 0 for question, 1 for context
            seq_ids = tokenized_inputs.sequence_ids(i)

            # Store basic features
            all_input_ids.append(input_ids[i])
            all_attention_masks.append(attention_mask[i])
            all_offset_mappings.append(offsets)  # Store as list/array
            all_example_ids.append(example_id)
            all_context_texts.append(context)

            # Label generation
            if is_test:
                all_labels.append(None)
                continue

            # Initialize labels as O (0)
            labels = np.zeros(len(offsets), dtype=int)

            # Find context bounds in tokens
            # We only label answers if they appear in the context part (sequence_id == 1)
            context_start_idx = 0
            while context_start_idx < len(seq_ids) and seq_ids[context_start_idx] != 1:
                context_start_idx += 1

            context_end_idx = len(seq_ids) - 1
            while context_end_idx >= 0 and seq_ids[context_end_idx] != 1:
                context_end_idx -= 1

            # Check if valid context exists in this window
            if context_start_idx <= context_end_idx:
                # Check strict containment
                # The window's context span in chars
                window_start_char = offsets[context_start_idx][0]
                window_end_char = offsets[context_end_idx][1]

                # If the answer is fully contained in this window
                if (start_char >= window_start_char) and (end_char <= window_end_char):
                    # Find start token index
                    current_idx = context_start_idx
                    while (
                        current_idx <= context_end_idx
                        and offsets[current_idx][0] <= start_char
                    ):
                        current_idx += 1
                    start_token_idx = current_idx - 1

                    # Find end token index
                    current_idx = context_end_idx
                    while (
                        current_idx >= context_start_idx
                        and offsets[current_idx][1] >= end_char
                    ):
                        current_idx -= 1
                    end_token_idx = current_idx + 1

                    # Assign labels
                    # B-ANS = 1, I-ANS = 2
                    if start_token_idx <= end_token_idx:
                        labels[start_token_idx] = 1  # B-ANS
                        if start_token_idx < end_token_idx:
                            labels[start_token_idx + 1 : end_token_idx + 1] = 2  # I-ANS

            all_labels.append(labels)

    # Create DataFrame
    feature_dict = {
        "input_ids": [x.tolist() for x in all_input_ids],
        "attention_mask": [x.tolist() for x in all_attention_masks],
        "offset_mapping": [x.tolist() for x in all_offset_mappings],
        "example_id": all_example_ids,
        "context": all_context_texts,
    }

    if not is_test:
        feature_dict["labels"] = [x.tolist() for x in all_labels]
    else:
        feature_dict["labels"] = [[] for _ in all_labels]  # Placeholder

    features_df = pd.DataFrame(feature_dict)

    # Save to cache
    os.makedirs(Config.QA_CACHE_DIR, exist_ok=True)
    features_df.to_parquet(cache_path)
    print(f"Saved {len(features_df)} features to {cache_path}")

    return features_df


# Wrapper for convenience matching the prompt description
def prepare_train_features(tokenizer, load_cached_data=True):
    """
    Wrapper to prepare training features using the standard train split.
    """
    return prepare_qa_features(
        tokenizer,
        Config.TRAIN_META_PATH,
        "train",
        load_cached_data=load_cached_data,
        is_test=False,
    )
