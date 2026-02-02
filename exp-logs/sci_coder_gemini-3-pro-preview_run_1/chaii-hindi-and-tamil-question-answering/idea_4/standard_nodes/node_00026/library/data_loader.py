import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config
from library.utils import set_seed

# Ensure deterministic behavior
set_seed(Config.SEED)


class QADataset(Dataset):
    """
    PyTorch Dataset for Question Answering.
    Handles both training (with labels) and inference (without labels) modes.
    """

    def __init__(self, data, mode="train"):
        self.data = data
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        item = {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
        }

        # MuRIL/BERT token_type_ids are not strictly necessary for all models
        # but good to have if the tokenizer produces them.
        if "token_type_ids" in row:
            item["token_type_ids"] = torch.tensor(
                row["token_type_ids"], dtype=torch.long
            )

        if self.mode == "train":
            item["start_positions"] = torch.tensor(
                row["start_positions"], dtype=torch.long
            )
            item["end_positions"] = torch.tensor(row["end_positions"], dtype=torch.long)
        else:
            # For inference, we need mappings to reconstruct the answer
            # We return these as non-tensor items or handle them in the collate_fn/loop
            # Here we just pass them; the DataLoader will stack tensors and list others
            item["example_id"] = row["example_id"]

            # Fix for Lesson 00010: Sanitize deserialized data types
            # Parquet/Pandas may load lists as numpy arrays, which slows down tensor creation
            offsets = row["offset_mapping"]
            if isinstance(offsets, np.ndarray):
                offsets = offsets.tolist()

            item["offset_mapping"] = torch.tensor(offsets, dtype=torch.long)

            # We also need to know which part of the context this is, but offset_mapping handles that.
            # We might need the original context text later, but that's in the raw df.

        return item


def prepare_train_features(df, tokenizer):
    """
    Tokenizes training data with sliding window and applies negative sampling.
    """
    # Lists to store processed samples
    features = []

    # Process row by row to handle alignment carefully
    # (Batch processing is faster but row-by-row is easier to debug for complex offset logic)
    for _, row in df.iterrows():
        context = row["context"]
        question = row["question"]
        answer_text = row["answer_text"]
        start_char = row["answer_start"]
        end_char = start_char + len(answer_text)

        # Tokenize
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

        sample_map = tokenized_inputs.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_inputs.pop("offset_mapping")

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_inputs["input_ids"][i]
            attention_mask = tokenized_inputs["attention_mask"][i]
            # Some tokenizers don't return token_type_ids, handle gracefully
            token_type_ids = tokenized_inputs.get(
                "token_type_ids", [0] * len(input_ids)
            )[i]

            # Sequence ids: None for special tokens, 0 for question, 1 for context
            sequence_ids = tokenized_inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # Check if answer is fully contained in this window
            # If the answer is not fully inside the context, label is (0, 0)
            if not (
                offsets[context_start][0] <= start_char
                and offsets[context_end][1] >= end_char
            ):
                start_label = 0
                end_label = 0
                is_positive = False
            else:
                # Map char indices to token indices
                idx = context_start
                while idx <= context_end and offsets[idx][0] <= start_char:
                    idx += 1
                start_label = idx - 1

                idx = context_end
                while idx >= context_start and offsets[idx][1] >= end_char:
                    idx -= 1
                end_label = idx + 1
                is_positive = True

            features.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                    "start_positions": start_label,
                    "end_positions": end_label,
                    "is_positive": is_positive,
                }
            )

    feature_df = pd.DataFrame(features)

    # Stratified Negative Sampling
    positives = feature_df[feature_df["is_positive"] == True]
    negatives = feature_df[feature_df["is_positive"] == False]

    n_pos = len(positives)
    n_neg_keep = int(n_pos * Config.NEGATIVE_SAMPLING_RATIO)

    # If we have fewer negatives than we want to keep, take all of them
    if len(negatives) > n_neg_keep:
        negatives = negatives.sample(n=n_neg_keep, random_state=Config.SEED)

    final_df = (
        pd.concat([positives, negatives])
        .sample(frac=1, random_state=Config.SEED)
        .reset_index(drop=True)
    )

    # Drop the helper column
    final_df = final_df.drop(columns=["is_positive"])

    return final_df


def prepare_test_features(df, tokenizer):
    """
    Tokenizes test/validation data with sliding window.
    Retains offset mappings and example IDs for inference reconstruction.
    """
    features = []

    for _, row in df.iterrows():
        example_id = row["id"]
        context = row["context"]
        question = row["question"]

        # Tokenize
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

        offset_mapping = tokenized_inputs.pop("offset_mapping")

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_inputs["input_ids"][i]
            attention_mask = tokenized_inputs["attention_mask"][i]
            token_type_ids = tokenized_inputs.get(
                "token_type_ids", [0] * len(input_ids)
            )[i]

            # We set sequence_ids to None for special tokens in offset_mapping
            # to avoid selecting them as answers
            sequence_ids = tokenized_inputs.sequence_ids(i)

            # Clean up offset mapping: set offsets for non-context tokens to None or (0,0)
            # Actually, standard practice is to keep them but filter via sequence_ids during inference.
            # But let's follow the standard torch pattern:
            # We just save the offset_mapping as is.

            features.append(
                {
                    "example_id": example_id,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                    "offset_mapping": offsets,
                    # Store sequence_ids to know which part is context
                    # sequence_ids contains Nones, which Parquet might not like mixed with ints.
                    # We replace None with -1
                    "sequence_ids": [-1 if s is None else s for s in sequence_ids],
                }
            )

    return pd.DataFrame(features)


def load_or_process_data(split_name, df, tokenizer, load_cached_data):
    """
    Handles caching logic.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{split_name}_features.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split_name} features from {cache_path}...")
        try:
            # Read parquet
            processed_df = pd.read_parquet(cache_path, engine="pyarrow")
            return processed_df
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing {split_name} features...")
    if split_name == "train":
        # For training split, we have answers and do negative sampling
        processed_df = prepare_train_features(df, tokenizer)
    else:
        # For val/test, we don't assume answers are present (or we don't use them for sampling)
        # Note: Val set usually has answers for evaluation, but we process it like test
        # to evaluate the full retrieval pipeline, OR we process like train to calculate loss.
        # Given the prompt implies using Val for model selection, we might want to evaluate metrics.
        # However, usually we want to validate on the full set without sampling.
        # So we use prepare_test_features for val as well, but we might need to attach labels separately if we want loss.
        # To keep it simple and consistent with 'Test' logic (inference), we use prepare_test_features.
        # If we need to compute Val Loss, we would need labels.
        # Let's check if Val has labels.
        if "answer_text" in df.columns and split_name == "val_for_loss":
            # Special case if we wanted val loss with sampling (not implemented here)
            pass

        processed_df = prepare_test_features(df, tokenizer)

    # Save to cache
    print(f"Saving {split_name} features to {cache_path}...")
    processed_df.to_parquet(cache_path, engine="pyarrow", index=False)

    return processed_df


def get_tokenizer():
    return AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)


def get_train_data(load_cached_data=True, debug=Config.DEBUG):
    """
    Returns the training dataset.
    """
    tokenizer = get_tokenizer()

    # Load Metadata
    df = pd.read_csv(Config.TRAIN_CSV)

    if debug:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    # Process
    features_df = load_or_process_data("train", df, tokenizer, load_cached_data)

    return QADataset(features_df, mode="train")


def get_val_data(load_cached_data=True, debug=Config.DEBUG):
    """
    Returns the validation dataset (processed for inference/eval).
    """
    tokenizer = get_tokenizer()
    df = pd.read_csv(Config.VAL_CSV)

    if debug:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    features_df = load_or_process_data("val", df, tokenizer, load_cached_data)

    return QADataset(features_df, mode="eval")


def get_test_data(load_cached_data=True, debug=Config.DEBUG):
    """
    Returns the test dataset.
    """
    tokenizer = get_tokenizer()
    df = pd.read_csv(Config.TEST_CSV)

    if debug:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    features_df = load_or_process_data("test", df, tokenizer, load_cached_data)

    return QADataset(features_df, mode="test")
