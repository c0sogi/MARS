import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from library.config import Config
from library.features import FeatureEngineer


class ChatbotDataset(Dataset):
    """
    PyTorch Dataset for the Chatbot Arena task.
    Handles tokenization of (Prompt, Response) pairs and integration of scalar features.
    """

    def __init__(
        self, df: pd.DataFrame, split_name: str, tokenizer, debug: bool = False
    ):
        """
        Args:
            df (pd.DataFrame): The dataframe containing text and labels.
            split_name (str): 'train', 'val', or 'test' for caching identification.
            tokenizer: HuggingFace tokenizer instance.
            debug (bool): If True, limits data size.
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.split_name = split_name

        # Handle debug mode by slicing the dataframe
        if debug:
            self.df = self.df.head(50)

        # Extract/Load scalar features using the provided FeatureEngineer
        # This handles the caching logic internally as per requirements
        self.feature_engineer = FeatureEngineer()
        self.scalar_features = self.feature_engineer.process_and_cache(
            self.df,
            split_name=f"{split_name}{'_debug' if debug else ''}",
            load_cached_data=True,
        )

        # Pre-extract text columns to lists for faster access
        self.prompts = self.df["prompt"].fillna("").astype(str).tolist()
        self.responses_a = self.df["response_a"].fillna("").astype(str).tolist()
        self.responses_b = self.df["response_b"].fillna("").astype(str).tolist()

        # Handle targets
        self.has_targets = "winner_model_a" in self.df.columns
        if self.has_targets:
            self.targets = self.df[
                ["winner_model_a", "winner_model_b", "winner_tie"]
            ].values.astype(np.float32)
        else:
            self.targets = None

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
            max_length=Config.MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize Branch B: [CLS] Prompt [SEP] Response B [SEP]
        encoded_b = self.tokenizer(
            prompt,
            resp_b,
            truncation=True,
            max_length=Config.MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )

        # Retrieve scalar features
        features = torch.tensor(self.scalar_features[idx], dtype=torch.float32)

        item = {
            "input_ids_a": encoded_a["input_ids"].squeeze(0),
            "attention_mask_a": encoded_a["attention_mask"].squeeze(0),
            "input_ids_b": encoded_b["input_ids"].squeeze(0),
            "attention_mask_b": encoded_b["attention_mask"].squeeze(0),
            "scalar_features": features,
        }

        if self.has_targets:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


def get_dataloaders(debug: bool = False):
    """
    Factory function to create DataLoaders for train, val, and test sets.

    Args:
        debug (bool): If True, uses a small subset of data for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load DataFrames
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Create Datasets
    train_dataset = ChatbotDataset(train_df, "train", tokenizer, debug=debug)
    val_dataset = ChatbotDataset(val_df, "val", tokenizer, debug=debug)
    test_dataset = ChatbotDataset(test_df, "test", tokenizer, debug=debug)

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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
