import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config


class QuestDataset(Dataset):
    """
    PyTorch Dataset for StackExchange Question-Answer pairs.
    Expects a DataFrame with pre-processed 'question_text' and 'answer_text' columns.
    """

    def __init__(self, df, is_test=False):
        self.df = df.reset_index(drop=True)
        self.is_test = is_test
        self.target_cols = Config.target_cols

        # Extract text lists for efficient indexing
        self.questions = self.df["question_text"].tolist()
        self.answers = self.df["answer_text"].tolist()

        # Extract labels if training/validation
        if not self.is_test:
            # Verify all target columns exist
            missing = [col for col in self.target_cols if col not in self.df.columns]
            if missing:
                raise ValueError(f"The following target columns are missing: {missing}")

            self.labels = self.df[self.target_cols].values.astype(np.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Return raw text; tokenization happens in the collator
        sample = {
            "question": str(self.questions[idx]),
            "answer": str(self.answers[idx]),
        }

        if not self.is_test:
            sample["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return sample


class CollateFactory:
    """
    Collator that handles dynamic padding and tokenization using the DeBERTa-v3 tokenizer.
    """

    def __init__(self, tokenizer, max_len=None):
        self.tokenizer = tokenizer
        self.max_len = max_len if max_len is not None else Config.max_len

    def __call__(self, batch):
        # Extract batch items
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]

        # Tokenize Questions
        # Dynamic padding: pad to the longest sequence in this specific batch
        q_enc = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        # Tokenize Answers
        a_enc = self.tokenizer(
            answers,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        # Construct output dictionary
        batch_out = {
            "q_input_ids": q_enc["input_ids"],
            "q_attention_mask": q_enc["attention_mask"],
            "a_input_ids": a_enc["input_ids"],
            "a_attention_mask": a_enc["attention_mask"],
        }

        # Stack labels if they exist
        if "labels" in batch[0]:
            labels = torch.stack([item["labels"] for item in batch])
            batch_out["labels"] = labels

        return batch_out


def load_data(load_cached_data=True, debug=Config.debug):
    """
    Loads data from metadata, processes text columns, and caches the result.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache first.
        debug (bool): If True, returns a small subset of the data.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    def _process_split(meta_path, cache_path, is_test=False):
        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception:
                # If load fails, proceed to re-process
                pass

        # 2. Process from scratch
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df = pd.read_csv(meta_path)

        # Fill NaNs in text columns
        df["question_title"] = df["question_title"].fillna("").astype(str)
        df["question_body"] = df["question_body"].fillna("").astype(str)
        df["answer"] = df["answer"].fillna("").astype(str)

        # Feature Engineering: Concatenate Title + Body for Question Input
        df["question_text"] = df["question_title"] + " " + df["question_body"]
        df["answer_text"] = df["answer"]

        # 3. Save to cache
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            print(f"Warning: Failed to save cache to {cache_path}. Error: {e}")

        return df

    # Process all splits
    train_df = _process_split(Config.train_path, Config.train_cache_path)
    val_df = _process_split(Config.val_path, Config.val_cache_path)
    test_df = _process_split(Config.test_path, Config.test_cache_path, is_test=True)

    # Apply debug slicing if requested
    if debug:
        train_df = train_df.head(100)
        val_df = val_df.head(100)
        test_df = test_df.head(100)

    return train_df, val_df, test_df


def get_tokenizer():
    """
    Instantiates the tokenizer defined in Config.
    """
    return AutoTokenizer.from_pretrained(Config.model_name)
