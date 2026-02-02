import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.features import StructuralFeatureGenerator
from library.utils import seed_everything


class ChatbotDataset(Dataset):
    """
    PyTorch Dataset for the Siamese Chatbot Preference model.
    Handles tokenization of (Prompt + Response) pairs and retrieval of structural features.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        structural_features: np.ndarray,
        tokenizer,
        max_length: int,
        is_test: bool = False,
    ):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'prompt', 'response_a', 'response_b'.
            structural_features (np.ndarray): Pre-computed structural features (N, F).
            tokenizer: HuggingFace tokenizer.
            max_length (int): Maximum sequence length.
            is_test (bool): If True, does not look for target columns.
        """
        self.df = df
        self.structural_features = structural_features
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        # Pre-extract text columns to avoid overhead in __getitem__
        # Fill NaNs with empty strings to prevent tokenization errors
        self.prompts = df["prompt"].fillna("").astype(str).values
        self.responses_a = df["response_a"].fillna("").astype(str).values
        self.responses_b = df["response_b"].fillna("").astype(str).values

        if not self.is_test:
            # Targets: winner_model_a, winner_model_b, winner_tie
            self.labels = df[
                ["winner_model_a", "winner_model_b", "winner_tie"]
            ].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        resp_a = self.responses_a[idx]
        resp_b = self.responses_b[idx]

        # Tokenize Branch A: [CLS] Prompt [SEP] Response A [SEP]
        encoded_a = self.tokenizer(
            prompt,
            resp_a,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize Branch B: [CLS] Prompt [SEP] Response B [SEP]
        encoded_b = self.tokenizer(
            prompt,
            resp_b,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Structural features
        struct_feats = torch.tensor(self.structural_features[idx], dtype=torch.float32)

        item = {
            "input_ids_a": encoded_a["input_ids"].squeeze(0),
            "attention_mask_a": encoded_a["attention_mask"].squeeze(0),
            "input_ids_b": encoded_b["input_ids"].squeeze(0),
            "attention_mask_b": encoded_b["attention_mask"].squeeze(0),
            "structural_features": struct_feats,
        }

        if not self.is_test:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return item


def load_data(load_cached_features: bool = True):
    """
    Loads dataframes and structural features.
    Handles DEBUG subsetting logic.

    Args:
        load_cached_features (bool): Whether to try loading features from disk.

    Returns:
        tuple: ((train_df, train_feats), (val_df, val_feats), (test_df, test_feats))
    """
    # Load DataFrames
    if not os.path.exists(Config.TRAIN_PATH):
        raise FileNotFoundError(f"Training metadata not found at {Config.TRAIN_PATH}")

    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Generate/Load Structural Features
    # The generator handles caching internally via .npy files
    gen = StructuralFeatureGenerator()
    train_feats = gen.get_features("train", load_cached_data=load_cached_features)
    val_feats = gen.get_features("val", load_cached_data=load_cached_features)
    test_feats = gen.get_features("test", load_cached_data=load_cached_features)

    # Handle Debug Mode
    if Config.DEBUG:
        print(
            f"DEBUG mode enabled. Subsetting data to {Config.DEBUG_SUBSET_SIZE} rows."
        )
        train_df = train_df.head(Config.DEBUG_SUBSET_SIZE)
        val_df = val_df.head(Config.DEBUG_SUBSET_SIZE)
        test_df = test_df.head(Config.DEBUG_SUBSET_SIZE)

        train_feats = train_feats[: Config.DEBUG_SUBSET_SIZE]
        val_feats = val_feats[: Config.DEBUG_SUBSET_SIZE]
        test_feats = test_feats[: Config.DEBUG_SUBSET_SIZE]

    return (train_df, train_feats), (val_df, val_feats), (test_df, test_feats)


def get_dataloaders(load_cached_features: bool = True):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        load_cached_features (bool): Whether to use cached structural features.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    seed_everything()

    # Load raw data and features
    (train_df, train_feats), (val_df, val_feats), (test_df, test_feats) = load_data(
        load_cached_features
    )

    # Initialize Tokenizer
    # We use the tokenizer corresponding to the backbone model
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Create Datasets
    train_dataset = ChatbotDataset(
        train_df, train_feats, tokenizer, Config.MAX_LENGTH, is_test=False
    )

    val_dataset = ChatbotDataset(
        val_df, val_feats, tokenizer, Config.MAX_LENGTH, is_test=False
    )

    test_dataset = ChatbotDataset(
        test_df, test_feats, tokenizer, Config.MAX_LENGTH, is_test=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,  # Drop incomplete batch to maintain shape consistency
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader
