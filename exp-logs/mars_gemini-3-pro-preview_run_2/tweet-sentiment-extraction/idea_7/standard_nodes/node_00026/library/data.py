import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import seed_everything


class TweetDataset(Dataset):
    """
    Dataset class for Sentiment Analysis Span Extraction.
    Handles tokenization, offset mapping, and target generation.
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.texts = df["text"].values
        self.sentiments = df["sentiment"].values
        if not is_test:
            self.selected_texts = df["selected_text"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Normalize whitespace to ensure consistency between text and selected_text
        text = str(self.texts[idx])
        text = " ".join(text.split())

        sentiment = str(self.sentiments[idx])

        # Tokenize: [CLS] <Sentiment> [SEP] <Tweet Text> [SEP]
        # We use encode_plus to handle the two sequences (sentiment and text)
        encoded = self.tokenizer.encode_plus(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            return_attention_mask=True,
        )

        ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)
        token_type_ids = torch.tensor(encoded["token_type_ids"], dtype=torch.long)

        # Offsets are (start_char, end_char) for each token
        offsets = encoded["offset_mapping"]

        # sequence_ids identifies which part of the input the token belongs to
        # None: Special tokens, 0: Sentiment, 1: Tweet Text
        sequence_ids = encoded.sequence_ids()

        start_idx = 0
        end_idx = 0

        if not self.is_test:
            selected_text = str(self.selected_texts[idx])
            selected_text = " ".join(selected_text.split())

            # Find the character start and end indices of the selected_text within the cleaned text
            start_char = text.find(selected_text)

            # Handle edge cases where cleaning might cause mismatch (fallback to full text)
            if start_char == -1:
                start_char = 0
                end_char = len(text)
            else:
                end_char = start_char + len(selected_text)

            token_start_index = 0
            token_end_index = 0
            found_start = False

            # Iterate through tokens to find the ones that overlap with the character span
            for i, (offset, seq_id) in enumerate(zip(offsets, sequence_ids)):
                # We only care about tokens corresponding to the tweet text (seq_id == 1)
                if seq_id != 1:
                    continue

                # offset is (start, end) relative to the text string
                # Check if this token contains the start of the selected text
                if not found_start:
                    if offset[0] >= start_char:
                        token_start_index = i
                        found_start = True

                # Check if this token is within the end of the selected text
                # We extend the end index as long as the token overlaps with the target span
                if offset[1] <= end_char:
                    token_end_index = i
                elif offset[0] < end_char:
                    # Handle partial overlap at the end of the span
                    token_end_index = i

            start_idx = token_start_index
            end_idx = token_end_index

            # Ensure validity
            if start_idx > end_idx:
                end_idx = start_idx

        # Create a mask that is 1 for text tokens and 0 for others (sentiment/special)
        # Used during inference to restrict predictions to the text area
        text_sequence_mask = [1 if s == 1 else 0 for s in sequence_ids]

        data = {
            "ids": ids,
            "mask": mask,
            "token_type_ids": token_type_ids,
            "orig_text": text,  # Return the cleaned text used for offset mapping
            "sentiment": sentiment,
            "offsets": torch.tensor(offsets, dtype=torch.long),
            "text_sequence_mask": torch.tensor(text_sequence_mask, dtype=torch.long),
        }

        if not self.is_test:
            data["start_targets"] = torch.tensor(start_idx, dtype=torch.long)
            data["end_targets"] = torch.tensor(end_idx, dtype=torch.long)

        return data


def process_data(load_cached_data=True):
    """
    Loads the full training data, cleans it, and creates 5-Fold Stratified splits.
    Caches the processed dataframe to a parquet file to ensure consistency across runs.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: The dataframe with a 'kfold' column.
    """
    cache_path = os.path.join(Config.OUTPUT_DIR, "train_folds.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        # print(f"Loading cached training data from {cache_path}")
        df = pd.read_parquet(cache_path)
        return df

    # 2. Process from scratch
    # print("Processing data and creating folds...")

    # Load raw data
    df = pd.read_csv(Config.TRAIN_CSV)

    # Basic cleaning
    df["text"] = df["text"].astype(str)
    df["selected_text"] = df["selected_text"].astype(str)
    df = df.dropna().reset_index(drop=True)

    # Create Stratified K-Folds
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    df["kfold"] = -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["sentiment"])):
        df.loc[val_idx, "kfold"] = fold

    # 3. Save to cache
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def get_loaders(fold, tokenizer, load_cached_data=True):
    """
    Creates PyTorch DataLoaders for the training and validation sets of a specific fold.

    Args:
        fold (int): The fold index (0-4).
        tokenizer: The HuggingFace tokenizer.
        load_cached_data (bool): Whether to use cached fold splits.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load processed data with folds
    df = process_data(load_cached_data=load_cached_data)

    # Split into train and validation
    train_df = df[df["kfold"] != fold].reset_index(drop=True)
    val_df = df[df["kfold"] == fold].reset_index(drop=True)

    # Debug mode: subset data
    if Config.DEBUG:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Create Datasets
    train_dataset = TweetDataset(train_df, tokenizer, Config.MAX_LEN, is_test=False)
    val_dataset = TweetDataset(val_df, tokenizer, Config.MAX_LEN, is_test=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(tokenizer):
    """
    Creates a PyTorch DataLoader for the test set.

    Args:
        tokenizer: The HuggingFace tokenizer.

    Returns:
        DataLoader: The test data loader.
    """
    df_test = pd.read_csv(Config.TEST_CSV)
    df_test["text"] = df_test["text"].astype(str)

    dataset = TweetDataset(df_test, tokenizer, Config.MAX_LEN, is_test=True)

    loader = DataLoader(
        dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return loader
