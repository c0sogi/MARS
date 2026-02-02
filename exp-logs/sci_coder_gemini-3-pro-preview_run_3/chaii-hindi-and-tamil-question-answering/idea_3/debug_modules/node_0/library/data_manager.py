import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import set_seed


class QADataset(Dataset):
    """
    PyTorch Dataset for Question Answering.
    Wraps processed features (input_ids, attention_mask, labels, etc.).
    """

    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = self.features[idx]

        # Convert lists to tensors
        batch = {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
        }

        # Add labels if they exist (Training mode)
        if "labels" in item and item["labels"] is not None:
            batch["labels"] = torch.tensor(item["labels"], dtype=torch.long)

        # Add metadata for inference/evaluation
        # Note: offset_mapping is needed for post-processing predictions
        if "offset_mapping" in item:
            # We don't convert offset_mapping to tensor here usually because it's a list of tuples/lists
            # But for the collator or loop, we might need it.
            # We'll return it as a tensor or keep it in a separate structure if needed.
            # For simplicity in standard loops, we can return it as a tensor.
            batch["offset_mapping"] = torch.tensor(
                item["offset_mapping"], dtype=torch.long
            )

        if "example_id" in item:
            batch["example_id"] = item["example_id"]

        if "context" in item:
            batch["context"] = item["context"]

        return batch


def prepare_tapt_corpus(output_path, load_cached_data=True):
    """
    Extracts raw context text from Train, Val, and Test sets for Task-Adaptive Pretraining.
    Saves as a text file.
    """
    # Check if file exists and we want to load cached
    if load_cached_data and os.path.exists(output_path):
        print(f"TAPT corpus already exists at {output_path}. Skipping generation.")
        return

    print("Generating TAPT corpus...")

    # Load all metadata
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_val = pd.read_csv(Config.VAL_META_PATH)
    df_test = pd.read_csv(Config.TEST_META_PATH)

    # Concatenate
    all_dfs = [df_train, df_val, df_test]
    full_df = pd.concat(all_dfs, ignore_index=True)

    # Extract unique contexts to avoid duplication bias
    contexts = full_df["context"].dropna().unique()

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        for text in contexts:
            # Clean text slightly: remove newlines to keep one document per line for LineByLineDataset
            clean_text = text.replace("\n", " ").strip()
            if len(clean_text) > 10:  # Filter out very short noise
                f.write(clean_text + "\n")

    print(f"TAPT corpus saved to {output_path} with {len(contexts)} documents.")


def process_data_to_features(df, tokenizer, max_length, doc_stride, is_training=True):
    """
    Tokenizes data with sliding window and generates labels for QA.
    Returns a list of dictionaries.
    """
    # Prepare lists
    examples = df.to_dict("records")
    questions = [str(e["question"]).strip() for e in examples]
    contexts = [str(e["context"]) for e in examples]

    # Tokenize
    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    features = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]

        # Get sequence ids to distinguish question from context
        # None for special tokens, 0 for question, 1 for context
        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        example = examples[sample_index]

        feature = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offsets,
            "example_id": example["id"],
            "context": example[
                "context"
            ],  # Store context for post-processing if needed
        }

        if is_training:
            # Label generation
            start_char = example["answer_start"]
            answer_text = example["answer_text"]
            end_char = start_char + len(answer_text)

            # Initialize labels as O (0)
            labels = [0] * len(input_ids)

            # Find the start and end of the context in the current window
            # sequence_ids: [None, 0, 0, ..., None, 1, 1, ..., None]
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span
            # offsets[token_start_index][0] is the start char of the first context token
            # offsets[token_end_index][1] is the end char of the last context token
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                # Answer not fully inside window -> Label all as O
                pass
            else:
                # Move the token_start_index and token_end_index to the answer start and end
                # We want to find the token that starts at 'start_char' and ends at 'end_char'

                # Find start token
                current_idx = token_start_index
                while (
                    current_idx <= token_end_index
                    and offsets[current_idx][0] <= start_char
                ):
                    current_idx += 1
                start_token = current_idx - 1

                # Find end token
                current_idx = token_end_index
                while (
                    current_idx >= token_start_index
                    and offsets[current_idx][1] >= end_char
                ):
                    current_idx -= 1
                end_token = current_idx + 1

                # Assign labels
                # B-ANS = 1
                labels[start_token] = 1
                # I-ANS = 2
                for k in range(start_token + 1, end_token + 1):
                    labels[k] = 2

            feature["labels"] = labels

        features.append(feature)

    return features


def get_cached_features(
    df, tokenizer, cache_name, load_cached_data=True, is_training=True
):
    """
    Handles caching of processed features using Parquet.
    """
    cache_path = os.path.join(Config.QA_CACHE_DIR, f"{cache_name}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        try:
            # Read parquet
            df_features = pd.read_parquet(cache_path)

            # Convert back to list of dicts
            # Parquet stores lists as numpy arrays or lists, we ensure they are lists
            features = df_features.to_dict("records")

            # Ensure types are correct (lists, not numpy arrays)
            for f in features:
                f["input_ids"] = list(f["input_ids"])
                f["attention_mask"] = list(f["attention_mask"])
                f["offset_mapping"] = [
                    list(x) for x in f["offset_mapping"]
                ]  # Convert back to lists
                if "labels" in f and f["labels"] is not None:
                    # check if it's nan (which happens if column is missing/null)
                    if isinstance(f["labels"], float) and np.isnan(f["labels"]):
                        f["labels"] = None
                    else:
                        f["labels"] = list(f["labels"])

            return features
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Processing features for {cache_name}...")
    features = process_data_to_features(
        df, tokenizer, Config.MAX_LENGTH, Config.DOC_STRIDE, is_training=is_training
    )

    # Save to cache
    # Convert to DataFrame for Parquet
    # We need to ensure offset_mapping (list of tuples) is compatible.
    # Convert tuples to lists for storage.
    features_for_df = []
    for f in features:
        f_copy = f.copy()
        f_copy["offset_mapping"] = [list(x) for x in f["offset_mapping"]]
        features_for_df.append(f_copy)

    df_features = pd.DataFrame(features_for_df)

    # Ensure directory
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Save
    df_features.to_parquet(cache_path, index=False)
    print(f"Saved features to {cache_path}")

    return features


def get_fold_dataloaders(tokenizer, k_folds=3, load_cached_data=True):
    """
    Generates Stratified K-Fold DataLoaders.
    Returns a generator yielding (train_loader, val_loader) for each fold.
    """
    set_seed(Config.SEED)

    # 1. Load Raw Data
    df_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    df_val_meta = pd.read_csv(Config.VAL_META_PATH)

    # Combine for K-Fold
    full_train_df = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

    # 2. Process ALL data once (or check cache)
    # We cache the FULL processed features. Splitting happens afterwards to ensure consistency.
    # Cache key depends on dataset size to avoid stale cache on subset debugging
    cache_name = f"train_full_features_len{len(full_train_df)}"
    all_features = get_cached_features(
        full_train_df,
        tokenizer,
        cache_name,
        load_cached_data=load_cached_data,
        is_training=True,
    )

    # Map example_id to feature indices
    # One example might generate multiple features (sliding window)
    # We need to split based on example_id, not feature_id, to avoid leakage
    example_id_to_feature_indices = {}
    for idx, f in enumerate(all_features):
        eid = f["example_id"]
        if eid not in example_id_to_feature_indices:
            example_id_to_feature_indices[eid] = []
        example_id_to_feature_indices[eid].append(idx)

    unique_example_ids = list(example_id_to_feature_indices.keys())

    # Get labels for stratification (Language)
    # We need to look up the language for each unique example_id
    id_to_lang = full_train_df.set_index("id")["language"].to_dict()
    stratify_labels = [id_to_lang[eid] for eid in unique_example_ids]

    # 3. Stratified K-Fold Split
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=Config.SEED)

    for fold, (train_idx_arr, val_idx_arr) in enumerate(
        skf.split(unique_example_ids, stratify_labels)
    ):
        print(f"\n--- Preparing Fold {fold + 1}/{k_folds} ---")

        # Get Train/Val Example IDs
        train_example_ids = [unique_example_ids[i] for i in train_idx_arr]
        val_example_ids = [unique_example_ids[i] for i in val_idx_arr]

        # Gather Feature Indices
        train_feature_indices = []
        for eid in train_example_ids:
            train_feature_indices.extend(example_id_to_feature_indices[eid])

        val_feature_indices = []
        for eid in val_example_ids:
            val_feature_indices.extend(example_id_to_feature_indices[eid])

        # Create Subsets of features
        train_features = [all_features[i] for i in train_feature_indices]
        val_features = [all_features[i] for i in val_feature_indices]

        # Create Datasets
        train_dataset = QADataset(train_features)
        val_dataset = QADataset(val_features)

        # Create DataLoaders
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

        yield train_loader, val_loader


def get_test_dataloader(tokenizer, load_cached_data=True):
    """
    Creates the test dataloader.
    """
    df_test = pd.read_csv(Config.TEST_META_PATH)

    cache_name = f"test_features_len{len(df_test)}"
    test_features = get_cached_features(
        df_test,
        tokenizer,
        cache_name,
        load_cached_data=load_cached_data,
        is_training=False,
    )

    test_dataset = QADataset(test_features)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
