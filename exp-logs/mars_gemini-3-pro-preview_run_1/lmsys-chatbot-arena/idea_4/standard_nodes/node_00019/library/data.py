import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from library.config import Config
from library.utils import seed_everything

# Prevent tokenizer parallelism issues in DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def extract_scalar_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts scalar features from the dataframe:
    - Character counts (log1p)
    - Word counts (log1p)
    - Newline counts (log1p)
    - Length ratio (A / B)
    - Length difference (A - B)
    """
    # Ensure string type
    resp_a = df["response_a"].fillna("").astype(str)
    resp_b = df["response_b"].fillna("").astype(str)

    # 1. Character Lengths
    len_a_char = resp_a.str.len().values
    len_b_char = resp_b.str.len().values

    # 2. Word Counts
    len_a_word = resp_a.apply(lambda x: len(x.split())).values
    len_b_word = resp_b.apply(lambda x: len(x.split())).values

    # 3. Newline Counts
    newline_a = resp_a.str.count("\n").values
    newline_b = resp_b.str.count("\n").values

    # 4. Interactions
    # Add epsilon to denominator to avoid division by zero
    ratio = len_a_char / (len_b_char + 1e-6)
    diff = len_a_char - len_b_char

    # Stack features
    # Shape: (N, 8)
    features = np.vstack(
        [
            len_a_char,
            len_b_char,
            len_a_word,
            len_b_word,
            newline_a,
            newline_b,
            ratio,
            diff,
        ]
    ).T

    # Apply log1p to count-based features (indices 0-5) to normalize distribution
    features[:, 0:6] = np.log1p(features[:, 0:6])

    return features.astype(np.float32)


def get_or_compute_features(
    df: pd.DataFrame, cache_path: str, load_cached_data: bool
) -> np.ndarray:
    """
    Loads features from cache if available and requested; otherwise computes and saves them.
    """
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        try:
            features = np.load(cache_path)
            # Verify shape matches dataframe
            if features.shape[0] == len(df):
                return features
            else:
                print(
                    f"Cache shape mismatch ({features.shape[0]} vs {len(df)}). Recomputing..."
                )
        except Exception as e:
            print(f"Error loading cache {cache_path}: {e}. Recomputing...")

    # Compute features
    features = extract_scalar_features(df)

    # Save to cache
    np.save(cache_path, features)
    return features


class ChatbotDataset(Dataset):
    """
    PyTorch Dataset for Siamese DeBERTa model.
    Returns tokenized inputs for both branches (A and B) and scalar features.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        features: np.ndarray,
        max_len: int,
        is_test: bool = False,
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.features = features
        self.max_len = max_len
        self.is_test = is_test

        # Pre-extract text columns to avoid overhead in __getitem__
        self.prompts = df["prompt"].fillna("").astype(str).values
        self.resps_a = df["response_a"].fillna("").astype(str).values
        self.resps_b = df["response_b"].fillna("").astype(str).values

        if not self.is_test:
            # Targets: winner_model_a, winner_model_b, winner_tie
            self.targets = df[
                ["winner_model_a", "winner_model_b", "winner_tie"]
            ].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        resp_a = self.resps_a[idx]
        resp_b = self.resps_b[idx]

        # Tokenize Branch A: [CLS] Prompt [SEP] Response A [SEP]
        encoded_a = self.tokenizer(
            prompt,
            resp_a,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize Branch B: [CLS] Prompt [SEP] Response B [SEP]
        encoded_b = self.tokenizer(
            prompt,
            resp_b,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids_a": encoded_a["input_ids"].squeeze(0),
            "attention_mask_a": encoded_a["attention_mask"].squeeze(0),
            "input_ids_b": encoded_b["input_ids"].squeeze(0),
            "attention_mask_b": encoded_b["attention_mask"].squeeze(0),
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
        }

        if not self.is_test:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


def get_dataloaders(
    debug: bool = False,
    load_cached_data: bool = True,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.
    Handles data loading, debug sampling, feature caching, and tokenization.
    """
    seed_everything(Config.SEED)

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # 2. Debug Sampling
    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE).reset_index(drop=True)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE).reset_index(drop=True)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE).reset_index(drop=True)

    # 3. Feature Engineering & Caching
    # Note: If debug is True, we might want to avoid overwriting the main cache.
    # We append '_debug' to cache paths if in debug mode.
    suffix = "_debug" if debug else ""

    train_feat_path = Config.TRAIN_FEATURES_PATH.replace(".npy", f"{suffix}.npy")
    val_feat_path = Config.VAL_FEATURES_PATH.replace(".npy", f"{suffix}.npy")
    test_feat_path = Config.TEST_FEATURES_PATH.replace(".npy", f"{suffix}.npy")

    train_features = get_or_compute_features(
        train_df, train_feat_path, load_cached_data
    )
    val_features = get_or_compute_features(val_df, val_feat_path, load_cached_data)
    test_features = get_or_compute_features(test_df, test_feat_path, load_cached_data)

    # 4. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # 5. Create Datasets
    train_dataset = ChatbotDataset(
        train_df, tokenizer, train_features, Config.MAX_LEN, is_test=False
    )
    val_dataset = ChatbotDataset(
        val_df, tokenizer, val_features, Config.MAX_LEN, is_test=False
    )
    test_dataset = ChatbotDataset(
        test_df, tokenizer, test_features, Config.MAX_LEN, is_test=True
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
