import os
import ast
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


def clean_text(text):
    """
    Cleans the text column.
    Handles unicode-escaped text surrounded by double-quotes.
    Example: "You are an idiot." -> You are an idiot.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Attempt to use literal_eval to handle python-style string escaping and quotes
    try:
        # If it starts and ends with quotes, it might be a string literal
        if text.startswith('"') and text.endswith('"'):
            # This handles escaped characters like \n, \xe2, etc.
            cleaned = ast.literal_eval(text)
            return cleaned
    except (ValueError, SyntaxError):
        pass

    # Fallback cleanup if literal_eval fails
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    # Basic unicode unescape if needed
    try:
        text = text.encode("utf-8").decode("unicode_escape")
    except:
        pass

    return text


class InsultDataset(Dataset):
    """
    Dataset class for Insult Detection.
    """

    def __init__(self, df, tokenizer, max_length, is_test=False):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'Comment' and optionally 'Insult'.
            tokenizer: Transformers tokenizer.
            max_length (int): Maximum sequence length.
            is_test (bool): If True, does not return targets.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.texts = df["Comment"].values

        if not self.is_test:
            self.targets = df["Insult"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        out = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }

        if not self.is_test:
            out["target"] = torch.tensor(self.targets[idx], dtype=torch.float)

        return out


def load_processed_data(config, load_cached_data=True):
    """
    Loads train, val, and test data. Applies cleaning and caching.

    Args:
        config (Config): Configuration object.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    os.makedirs(config.output_dir, exist_ok=True)

    train_cache = os.path.join(config.output_dir, "train_cleaned.parquet")
    val_cache = os.path.join(config.output_dir, "val_cleaned.parquet")
    test_cache = os.path.join(config.output_dir, "test_cleaned.parquet")

    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        print("Loading processed data from cache...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
    else:
        print("Processing data from scratch...")
        # Load from metadata
        train_df = pd.read_csv(config.train_path)
        val_df = pd.read_csv(config.val_path)
        test_df = pd.read_csv(config.test_path)

        # Apply cleaning
        train_df["Comment"] = train_df["Comment"].apply(clean_text)
        val_df["Comment"] = val_df["Comment"].apply(clean_text)
        test_df["Comment"] = test_df["Comment"].apply(clean_text)

        # Save to cache
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df


def prepare_tapt_data(config, load_cached_data=True):
    """
    Aggregates text from train, val, and test sets for Task-Adaptive Pre-Training (TAPT).
    Writes the corpus to a text file.

    Args:
        config (Config): Configuration object.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        str: Path to the text file containing the corpus.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    tapt_file_path = os.path.join(config.output_dir, "tapt_corpus.txt")

    if load_cached_data and os.path.exists(tapt_file_path):
        print("Loading TAPT corpus from cache...")
        return tapt_file_path

    print("Generating TAPT corpus...")
    # Load processed dataframes to ensure text is cleaned
    train_df, val_df, test_df = load_processed_data(
        config, load_cached_data=load_cached_data
    )

    # Concatenate all texts
    all_texts = (
        pd.concat([train_df["Comment"], val_df["Comment"], test_df["Comment"]])
        .astype(str)
        .tolist()
    )

    # Filter out empty strings
    all_texts = [t for t in all_texts if len(t.strip()) > 0]

    # Save to text file (one line per document)
    with open(tapt_file_path, "w", encoding="utf-8") as f:
        for text in all_texts:
            # Replace newlines with spaces to ensure one sample per line for LineByLineDataset compatibility
            clean_line = text.replace("\n", " ").replace("\r", " ")
            f.write(clean_line + "\n")

    return tapt_file_path
