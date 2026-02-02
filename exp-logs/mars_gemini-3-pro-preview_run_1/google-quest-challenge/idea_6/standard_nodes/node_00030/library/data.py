import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


def get_target_columns():
    """
    Reads the sample submission file to retrieve the list of 30 target column names
    in the correct order.
    """
    sample_df = pd.read_csv(Config.sample_submission_path)
    # Filter out qa_id to get only target labels
    target_cols = [col for col in sample_df.columns if col != Config.qa_id_col]
    return target_cols


def get_tokenizer():
    """
    Initializes and returns the tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    return tokenizer


def process_text_data(df, tokenizer, target_cols=None, is_test=False):
    """
    Tokenizes text data and generates specific masks for questions and answers.
    """
    # 1. Prepare Text
    # Concatenate title and body for the question part
    questions = (
        df[Config.question_title_col].fillna("")
        + " "
        + df[Config.question_body_col].fillna("")
    ).tolist()
    answers = df[Config.answer_col].fillna("").tolist()

    # 2. Tokenize
    # We tokenize pairs (question, answer) to get [CLS] Q [SEP] A [SEP]
    # return_token_type_ids=True is default for many, but we rely on sequence_ids() for robustness
    encodings = tokenizer(
        questions,
        answers,
        truncation=True,
        max_length=Config.max_len,
        padding=False,  # We pad dynamically in Collate
        add_special_tokens=True,
    )

    # 3. Process into list of dicts for DataFrame storage
    processed_samples = []

    # Pre-fetch labels if not test
    if not is_test and target_cols is not None:
        labels = df[target_cols].values.astype(np.float32)

    qa_ids = df[Config.qa_id_col].values

    for i in range(len(questions)):
        input_ids = encodings.input_ids[i]

        # Generate Partition Masks using sequence_ids
        # sequence_ids: None (special), 0 (question), 1 (answer)
        seq_ids = encodings.sequence_ids(i)

        # Create binary masks
        # question_mask: 1 where token belongs to Question (seq_id=0), else 0
        question_mask = [1 if s == 0 else 0 for s in seq_ids]

        # answer_mask: 1 where token belongs to Answer (seq_id=1), else 0
        answer_mask = [1 if s == 1 else 0 for s in seq_ids]

        # Standard attention mask (1 for all real tokens)
        attention_mask = [1] * len(input_ids)

        sample = {
            "qa_id": qa_ids[i],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "question_mask": question_mask,
            "answer_mask": answer_mask,
        }

        if not is_test:
            sample["labels"] = labels[i]

        processed_samples.append(sample)

    return processed_samples


def load_data(split="train"):
    """
    Loads data for a specific split (train, val, test).
    Handles caching logic: checks for parquet file, otherwise processes from metadata CSV.
    """
    # Determine paths based on split
    if split == "train":
        csv_path = Config.train_path
        cache_path = Config.train_cache_path
        is_test = False
    elif split == "val":
        csv_path = Config.val_path
        cache_path = Config.val_cache_path
        is_test = False
    elif split == "test":
        csv_path = Config.test_path
        cache_path = Config.test_cache_path
        is_test = True
    else:
        raise ValueError(f"Unknown split: {split}")

    # 1. Try Load Cache
    # Cite debug_lesson_2: Bypass cache for test set to handle runtime swapping
    if Config.load_cached_data and os.path.exists(cache_path) and split != "test":
        print(f"Loading {split} data from cache: {cache_path}")
        try:
            # PyArrow handles list columns in Parquet efficiently
            df = pd.read_parquet(cache_path)
            # Convert DataFrame back to list of dicts for the Dataset
            data = df.to_dict("records")

            # Debug mode: slice data
            if Config.debug:
                data = data[:100]
                print(f"Debug mode: sampled {len(data)} rows.")

            return data
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing {split} data from {csv_path}...")
    df = pd.read_csv(csv_path)

    if Config.debug:
        df = df.head(100)
        print(f"Debug mode: sampled {len(df)} rows.")

    target_cols = get_target_columns()
    tokenizer = get_tokenizer()

    processed_data = process_text_data(df, tokenizer, target_cols, is_test)

    # 3. Save to Cache
    # We convert the list of dicts to a DataFrame for Parquet storage
    # This requires pyarrow engine which handles list columns
    try:
        cache_df = pd.DataFrame(processed_data)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache_df.to_parquet(cache_path, index=False)
        print(f"Saved processed {split} data to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return processed_data


class QUESTDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Convert lists to numpy arrays or tensors usually happens in collate or here
        # Returning raw lists/arrays here, Collate will tensorize and pad
        return {
            "qa_id": item["qa_id"],
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "question_mask": item["question_mask"],
            "answer_mask": item["answer_mask"],
            "labels": item.get("labels", None),
        }


class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = 0  # Fallback

    def __call__(self, batch):
        # Extract fields
        qa_ids = [item["qa_id"] for item in batch]
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        question_mask = [item["question_mask"] for item in batch]
        answer_mask = [item["answer_mask"] for item in batch]

        # Check if labels exist
        has_labels = batch[0]["labels"] is not None
        labels = [item["labels"] for item in batch] if has_labels else None

        # Dynamic Padding
        # Find max length in this batch
        max_len = max(len(ids) for ids in input_ids)

        # Helper to pad list of lists
        def pad_sequence(sequences, pad_value, max_len):
            padded = np.full((len(sequences), max_len), pad_value, dtype=np.int64)
            for i, seq in enumerate(sequences):
                padded[i, : len(seq)] = seq
            return torch.tensor(padded)

        # Pad inputs
        input_ids_tensor = pad_sequence(input_ids, self.pad_token_id, max_len)
        attention_mask_tensor = pad_sequence(attention_mask, 0, max_len)

        # Pad custom masks with 0 (inactive)
        question_mask_tensor = pad_sequence(question_mask, 0, max_len)
        answer_mask_tensor = pad_sequence(answer_mask, 0, max_len)

        batch_out = {
            "qa_ids": torch.tensor(qa_ids, dtype=torch.long),
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "question_mask": question_mask_tensor,
            "answer_mask": answer_mask_tensor,
        }

        if has_labels:
            # Labels are fixed size 30, no padding needed, just stack
            # Ensure they are float32 tensors
            # Convert list of arrays to single tensor
            labels_tensor = torch.tensor(np.array(labels), dtype=torch.float32)
            batch_out["labels"] = labels_tensor

        return batch_out


def get_dataloaders():
    """
    Factory function to create Train, Validation, and Test dataloaders.
    """
    tokenizer = get_tokenizer()
    collate_fn = Collate(tokenizer)

    # Train Loader
    train_data = load_data("train")
    train_dataset = QUESTDataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Val Loader
    val_data = load_data("val")
    val_dataset = QUESTDataset(val_data)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    # Test Loader
    test_data = load_data("test")
    test_dataset = QUESTDataset(test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
