import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything

# Label Definitions
LABEL_O = 0
LABEL_B = 1
LABEL_I = 2
LABEL_IGNORE = -100


class QADataset(Dataset):
    """
    PyTorch Dataset for Question Answering.
    Bundles input tensors with metadata (offset_mapping, example_id) for robust inference.
    """

    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = self.features[idx]

        # Convert lists to tensors
        # Cite debug_lesson_3: Explicitly convert numpy arrays (especially object arrays) to lists

        # input_ids
        input_ids_data = item["input_ids"]
        if isinstance(input_ids_data, np.ndarray):
            input_ids_data = input_ids_data.tolist()
        input_ids = torch.tensor(input_ids_data, dtype=torch.long)

        # attention_mask
        attention_mask_data = item["attention_mask"]
        if isinstance(attention_mask_data, np.ndarray):
            attention_mask_data = attention_mask_data.tolist()
        attention_mask = torch.tensor(attention_mask_data, dtype=torch.long)

        # Labels: list of integers (or None/dummy for test)
        # We ensure labels are always present (dummy -100 for test) to simplify collator
        labels_data = item["labels"]
        if isinstance(labels_data, np.ndarray):
            labels_data = labels_data.tolist()
        labels = torch.tensor(labels_data, dtype=torch.long)

        # Offset mapping: List of [start, end]
        offset_mapping_data = item["offset_mapping"]
        if isinstance(offset_mapping_data, np.ndarray):
            offset_mapping_data = offset_mapping_data.tolist()
        offset_mapping = torch.tensor(offset_mapping_data, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "offset_mapping": offset_mapping,
            "example_id": str(item["example_id"]),
        }


def prepare_features(df, tokenizer, mode="train"):
    """
    Tokenizes data with sliding window and generates labels (Soft Overlap).
    """
    # Ensure text columns are strings
    df["question"] = df["question"].fillna("").astype(str)
    df["context"] = df["context"].fillna("").astype(str)

    questions = df["question"].tolist()
    contexts = df["context"].tolist()

    # Batch tokenization with sliding window
    encodings = tokenizer(
        questions,
        contexts,
        max_length=Config.MAX_LENGTH,
        stride=Config.DOC_STRIDE,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Mappings from window index to original sample index
    sample_mapping = encodings.pop("overflow_to_sample_mapping")
    offset_mapping = encodings.pop("offset_mapping")
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    features = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        row = df.iloc[sample_idx]

        # Base feature dictionary
        feature = {
            "input_ids": input_ids[i],
            "attention_mask": attention_mask[i],
            "offset_mapping": offsets,
            "example_id": row["id"],
        }

        # Label Generation
        if mode == "test":
            # For test set, fill labels with IGNORE
            feature["labels"] = [LABEL_IGNORE] * len(input_ids[i])
        else:
            # Train/Val: Generate BIO labels
            answer_text = (
                str(row["answer_text"]) if pd.notna(row.get("answer_text")) else ""
            )
            try:
                answer_start = int(row["answer_start"])
            except (ValueError, TypeError):
                answer_start = -1

            # If invalid answer, treat as no answer (all O or Ignore)
            if answer_start == -1 or not answer_text:
                # If no answer is present, we label context as O?
                # Usually QA datasets have answers. If missing, we might ignore.
                # Let's label context as O.
                answer_end = -1
            else:
                answer_end = answer_start + len(answer_text)

            sequence_ids = encodings.sequence_ids(i)
            labels = []

            for token_idx, (seq_id, offset) in enumerate(zip(sequence_ids, offsets)):
                # 0 = Question, 1 = Context, None = Special
                if seq_id != 1:
                    labels.append(LABEL_IGNORE)
                    continue

                start_char, end_char = offset

                # Skip special tokens inside context (e.g. if tokenizer inserts them) or empty spans
                if start_char >= end_char:
                    labels.append(LABEL_IGNORE)
                    continue

                if answer_start == -1:
                    # No answer in this example -> all context is O
                    labels.append(LABEL_O)
                    continue

                # Soft Overlap Logic
                # Check for any overlap between token span and answer span
                overlap = max(start_char, answer_start) < min(end_char, answer_end)

                if overlap:
                    # If token contains the start of the answer, mark as B-ANS
                    if start_char <= answer_start < end_char:
                        labels.append(LABEL_B)
                    else:
                        labels.append(LABEL_I)
                else:
                    labels.append(LABEL_O)

            feature["labels"] = labels

        features.append(feature)

    return features


def get_qa_data(mode="train", load_cached_data=True):
    """
    Retrieves the dataset, utilizing caching to save time.
    """
    seed_everything(42)

    # Define paths
    if mode == "train":
        meta_path = Config.TRAIN_META_PATH
        cache_name = "train_features.parquet"
    elif mode == "val":
        meta_path = Config.VAL_META_PATH
        cache_name = "val_features.parquet"
    elif mode == "test":
        meta_path = Config.TEST_META_PATH
        cache_name = "test_features.parquet"
    else:
        raise ValueError(f"Invalid mode: {mode}")

    cache_path = os.path.join(Config.QA_CACHE_DIR, cache_name)

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {mode} features from cache: {cache_path}")
        try:
            df_features = pd.read_parquet(cache_path)
            # Convert DataFrame back to list of dicts
            features = df_features.to_dict("records")
            return QADataset(features)
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing {mode} data from {meta_path}...")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file missing: {meta_path}")

    df_meta = pd.read_csv(meta_path)

    # Initialize tokenizer (Base model)
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL_NAME)

    features = prepare_features(df_meta, tokenizer, mode=mode)

    # 3. Save to cache
    os.makedirs(Config.QA_CACHE_DIR, exist_ok=True)
    df_features = pd.DataFrame(features)
    # Parquet handles nested lists (input_ids, etc.) efficiently via PyArrow
    df_features.to_parquet(cache_path, index=False)
    print(f"Saved {len(features)} features to {cache_path}")

    return QADataset(features)


def get_dataloader(
    mode="train",
    batch_size=Config.TRAIN_BATCH_SIZE,
    shuffle=True,
    load_cached_data=True,
):
    """
    Returns a PyTorch DataLoader for the requested split.
    """
    dataset = get_qa_data(mode, load_cached_data)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )
