import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config
from library.features import prepare_scalar_features
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("dataset")


class ChatbotDataset(Dataset):
    """
    PyTorch Dataset for the Siamese Chatbot preference task.
    Handles tokenization of (Prompt + Response) pairs and retrieval of scalar features.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        scalar_features: pd.DataFrame,
        tokenizer,
        max_length: int,
        is_test: bool = False,
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing text and ids.
            scalar_features (pd.DataFrame): Dataframe of pre-computed scalar features.
            tokenizer: Hugging Face tokenizer instance.
            max_length (int): Maximum sequence length for tokenization.
            is_test (bool): If True, does not look for target columns.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        # Reset indices to ensure alignment between metadata and features
        self.df = df.reset_index(drop=True)
        self.scalar_features = scalar_features.reset_index(drop=True)

        # Sanity check for alignment
        if len(self.df) != len(self.scalar_features):
            raise ValueError(
                f"Mismatch in length: DF ({len(self.df)}) vs Features ({len(self.scalar_features)})"
            )

        # Pre-convert text columns to lists for faster access in __getitem__
        self.prompts = self.df["prompt"].fillna("").astype(str).tolist()
        self.responses_a = self.df["response_a"].fillna("").astype(str).tolist()
        self.responses_b = self.df["response_b"].fillna("").astype(str).tolist()

        # Convert scalar features to float32 numpy array
        self.scalar_features_values = self.scalar_features.values.astype(np.float32)

        # Pre-process targets if training/validation
        if not self.is_test:
            target_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
            # Ensure columns exist
            for col in target_cols:
                if col not in self.df.columns:
                    raise ValueError(f"Missing target column: {col}")
            self.targets = self.df[target_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Retrieve Text
        prompt = self.prompts[idx]
        resp_a = self.responses_a[idx]
        resp_b = self.responses_b[idx]

        # 2. Tokenize Branch A: [CLS] Prompt [SEP] Response A [SEP]
        # Transformers' tokenizer handles the special token placement automatically
        # when two sequences are provided.
        encoded_a = self.tokenizer(
            prompt,
            resp_a,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # 3. Tokenize Branch B: [CLS] Prompt [SEP] Response B [SEP]
        encoded_b = self.tokenizer(
            prompt,
            resp_b,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # 4. Retrieve Scalar Features
        scalar_feats = torch.tensor(self.scalar_features_values[idx], dtype=torch.float)

        # 5. Construct Output Dictionary
        item = {
            "input_ids_a": encoded_a["input_ids"].squeeze(0),
            "attention_mask_a": encoded_a["attention_mask"].squeeze(0),
            "input_ids_b": encoded_b["input_ids"].squeeze(0),
            "attention_mask_b": encoded_b["attention_mask"].squeeze(0),
            "scalar_features": scalar_feats,
        }

        # 6. Add Label if available
        if not self.is_test:
            item["label"] = torch.tensor(self.targets[idx], dtype=torch.float)

        return item


def get_tokenizer():
    """
    Factory function to load the tokenizer defined in Config.
    """
    logger.info(f"Loading tokenizer: {Config.model_name}")
    return AutoTokenizer.from_pretrained(Config.model_name)


def build_datasets(load_cached_data: bool = True, tokenizer=None):
    """
    Main function to construct Train, Validation, and Test datasets.
    Handles loading metadata, synchronizing with cached features, and
    initializing the Dataset objects.

    Args:
        load_cached_data (bool): Passed to feature engineering module.
        tokenizer: Optional tokenizer instance. If None, loads a new one.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # 1. Load Scalar Features (Handles caching internally)
    logger.info("Preparing scalar features...")
    train_feats, val_feats, test_feats = prepare_scalar_features(
        load_cached_data=load_cached_data
    )

    # 2. Load Metadata
    logger.info("Loading metadata CSVs...")
    df_train = pd.read_csv(Config.train_path)
    df_val = pd.read_csv(Config.val_path)
    df_test = pd.read_csv(Config.test_path)

    # 3. Apply Debug Subsetting (Must match logic in library/features.py)
    if hasattr(Config, "debug") and Config.debug:
        logger.info("Debug mode enabled: Subsetting metadata to match features.")
        df_train = df_train.head(100)
        df_val = df_val.head(50)
        df_test = df_test.head(50)

    # 4. Initialize Tokenizer if not provided
    if tokenizer is None:
        tokenizer = get_tokenizer()

    # 5. Instantiate Datasets
    logger.info("Creating ChatbotDataset instances...")
    train_dataset = ChatbotDataset(
        df=df_train,
        scalar_features=train_feats,
        tokenizer=tokenizer,
        max_length=Config.max_length,
        is_test=False,
    )

    val_dataset = ChatbotDataset(
        df=df_val,
        scalar_features=val_feats,
        tokenizer=tokenizer,
        max_length=Config.max_length,
        is_test=False,
    )

    test_dataset = ChatbotDataset(
        df=df_test,
        scalar_features=test_feats,
        tokenizer=tokenizer,
        max_length=Config.max_length,
        is_test=True,
    )

    logger.info(
        f"Datasets created. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )
    return train_dataset, val_dataset, test_dataset
