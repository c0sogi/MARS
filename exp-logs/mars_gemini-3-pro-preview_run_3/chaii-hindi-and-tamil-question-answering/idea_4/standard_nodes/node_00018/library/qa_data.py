import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config
from library.utils import set_seed

# Label mappings
LABEL_TO_ID = {"O": 0, "B": 1, "I": 2}
ID_TO_LABEL = {0: "O", 1: "B", 2: "I"}


class BIOQADataset(Dataset):
    """
    PyTorch Dataset for Token Classification (BIO tagging) in QA.
    """

    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]

        item = {
            "input_ids": torch.tensor(feat["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(feat["attention_mask"], dtype=torch.long),
        }

        # Add labels if they exist (Train/Val)
        if "labels" in feat and feat["labels"] is not None:
            item["labels"] = torch.tensor(feat["labels"], dtype=torch.long)

        # We don't return offset_mapping or example_id in the tensor batch usually,
        # but they are needed for evaluation. We can access them via the dataset object directly
        # during inference loops, or return them here if a custom collator handles them.
        # For standard Trainer/Loop, we usually stick to tensors.

        return item


def process_data_to_features(df, tokenizer, is_training=True):
    """
    Converts raw dataframe into sliding window features with BIO labels.
    """
    features = []

    # Ensure columns exist
    if "question" not in df.columns or "context" not in df.columns:
        raise ValueError("DataFrame must contain 'question' and 'context' columns.")

    # Iterate over examples
    for idx, row in df.iterrows():
        example_id = row["id"]
        question = str(row["question"]).strip()
        context = str(row["context"])  # Do not strip context to preserve offsets

        # Tokenize with sliding window
        # truncation="only_second" ensures question is kept, context is truncated/strided
        tokenized_inputs = tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=Config.MAX_LENGTH,
            stride=Config.DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # The tokenizer returns a list of windows for this single example
        sample_map = tokenized_inputs.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_inputs.pop("offset_mapping")

        # Iterate over each window generated from this example
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_inputs["input_ids"][i]
            attention_mask = tokenized_inputs["attention_mask"][i]
            sequence_ids = tokenized_inputs.sequence_ids(i)

            # Base feature dict
            feature = {
                "example_id": example_id,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "offset_mapping": offsets,
                "sequence_ids": sequence_ids,  # Useful for inference to identify context tokens
                "context_text": context,  # Store reference for decoding
            }

            if is_training:
                # Retrieve ground truth
                answer_text = str(row["answer_text"])
                start_char = row["answer_start"]
                end_char = start_char + len(answer_text)

                # Create BIO labels
                # Initialize all as 'O' (0)
                # We ignore special tokens and question tokens by masking or just treating as O
                # Standard practice: Set labels, ignore loss on special tokens if needed.
                # Here we just label them 'O'.
                labels = [LABEL_TO_ID["O"]] * len(input_ids)

                # Find the context span in the token list
                # sequence_ids: None (special), 0 (question), 1 (context)
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1
                    if token_start_index >= len(sequence_ids):
                        break

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1
                    if token_end_index < 0:
                        break

                # Detect if the answer is fully inside this window
                # offsets[token_start_index] is the (start, end) char of the first context token
                # offsets[token_end_index] is the (start, end) char of the last context token

                if token_start_index <= token_end_index:
                    window_start_char = offsets[token_start_index][0]
                    window_end_char = offsets[token_end_index][1]

                    # Check if answer is fully contained
                    if not (
                        window_start_char <= start_char and end_char <= window_end_char
                    ):
                        # Answer not fully in window -> Label all as O
                        pass
                    else:
                        # Answer is in window. Map chars to tokens.
                        # We iterate through context tokens and check overlap
                        for k in range(token_start_index, token_end_index + 1):
                            token_span = offsets[k]
                            t_start, t_end = token_span

                            # Skip 0-length tokens (some special chars)
                            if t_start == t_end:
                                continue

                            # Soft Overlap Logic (Cite solution_lesson_node_00017)
                            # Instead of strict containment (t_start >= start and t_end <= end),
                            # we check if the token overlaps with the answer span.
                            overlap_start = max(t_start, start_char)
                            overlap_end = min(t_end, end_char)

                            if overlap_start < overlap_end:
                                # There is an overlap.
                                # Determine if this is the Beginning (B) or Inside (I).
                                # If the token contains the start of the answer, it gets the B tag.
                                # This handles cases where the token starts before the answer (e.g. subword prefix).
                                if t_start <= start_char < t_end:
                                    labels[k] = LABEL_TO_ID["B"]
                                else:
                                    labels[k] = LABEL_TO_ID["I"]

                feature["labels"] = labels

            features.append(feature)

    return features


def prepare_qa_data(load_cached_data=True):
    """
    Main function to load metadata, tokenize, and prepare datasets for training/inference.

    Args:
        load_cached_data (bool): Whether to load from parquet cache if available.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, test_features_list)
               test_features_list is returned raw to assist with inference mapping.
    """
    set_seed(42)

    # Ensure cache directory exists
    os.makedirs(Config.QA_CACHE_DIR, exist_ok=True)

    # Determine which model path to use (TAPT or Base)
    # If TAPT model exists, use it. Else use base.
    if os.path.exists(Config.TAPT_MODEL_DIR) and os.path.exists(
        os.path.join(Config.TAPT_MODEL_DIR, "config.json")
    ):
        print(f"Loading tokenizer from TAPT model: {Config.TAPT_MODEL_DIR}")
        tokenizer_path = Config.TAPT_MODEL_DIR
    else:
        print(f"Loading tokenizer from base model: {Config.MODEL_CHECKPOINT}")
        tokenizer_path = Config.MODEL_CHECKPOINT

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # ---------------------------------------------------------
    # Helper to handle cache logic
    # ---------------------------------------------------------
    def get_cached_features(split_name, meta_path, is_training):
        cache_path = os.path.join(Config.QA_CACHE_DIR, f"{split_name}_features.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {split_name} features from {cache_path}")
            # Read parquet
            df_cache = pd.read_parquet(cache_path)
            # Convert back to list of dicts
            # Note: Parquet stores lists as arrays, need to ensure types
            features = df_cache.to_dict("records")
            # Ensure numpy arrays are converted to lists if needed, though PyTorch handles numpy
            return features

        print(f"Processing {split_name} data from {meta_path}...")
        if not os.path.exists(meta_path):
            print(f"Warning: {meta_path} not found. Returning empty list.")
            return []

        df_meta = pd.read_csv(meta_path)
        features = process_data_to_features(df_meta, tokenizer, is_training=is_training)

        # Save to cache
        print(f"Saving {len(features)} {split_name} features to {cache_path}")
        # Convert to DataFrame for Parquet
        # We need to be careful with 'sequence_ids' which might contain None.
        # Parquet doesn't like mixed types (int and None).
        # We replace None with -1 for storage.
        features_for_df = []
        for f in features:
            f_copy = f.copy()
            f_copy["sequence_ids"] = [-1 if x is None else x for x in f["sequence_ids"]]
            features_for_df.append(f_copy)

        df_out = pd.DataFrame(features_for_df)
        df_out.to_parquet(cache_path, index=False)

        return features

    # ---------------------------------------------------------
    # Process Splits
    # ---------------------------------------------------------
    train_features = get_cached_features(
        "train", Config.TRAIN_META_PATH, is_training=True
    )
    val_features = get_cached_features("val", Config.VAL_META_PATH, is_training=True)
    test_features = get_cached_features(
        "test", Config.TEST_META_PATH, is_training=False
    )

    # ---------------------------------------------------------
    # Create Datasets
    # ---------------------------------------------------------
    train_dataset = BIOQADataset(train_features)
    val_dataset = BIOQADataset(val_features)
    test_dataset = BIOQADataset(test_features)

    print(
        f"Dataset Sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset, test_features
