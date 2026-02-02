import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase
from library.config import Config
from library.utils import set_seed


class ChatbotDataset(Dataset):
    """
    Dataset class for Siamese DeBERTa model.
    Handles tokenization, scalar feature extraction, and response masking.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        mode: str = "train",
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

        # Pre-extract text columns to avoid overhead in __getitem__
        self.prompts = self.df["prompt"].fillna("").astype(str).values
        self.responses_a = self.df["response_a"].fillna("").astype(str).values
        self.responses_b = self.df["response_b"].fillna("").astype(str).values

        # Pre-extract targets if not in test mode
        if self.mode != "test":
            self.targets = self.df[
                ["winner_model_a", "winner_model_b", "winner_tie"]
            ].values.astype(np.float32)

        # Pre-extract IDs for test submission
        if self.mode == "test":
            self.ids = self.df["id"].values

    def __len__(self):
        return len(self.df)

    def _tokenize_pair(self, prompt, response):
        """
        Tokenizes a (Prompt, Response) pair and generates a response mask.
        """
        encoding = self.tokenizer(
            prompt,
            response,
            truncation="only_second",  # Prioritize preserving the prompt
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,  # Return lists, convert to tensor later
            add_special_tokens=True,
        )

        # Create Response Mask
        # sequence_ids: None (special), 0 (prompt), 1 (response)
        seq_ids = encoding.sequence_ids()

        # We want 1 for response tokens, 0 for everything else (prompt, special, padding)
        response_mask = [1 if s == 1 else 0 for s in seq_ids]

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                encoding["attention_mask"], dtype=torch.long
            ),
            "response_mask": torch.tensor(response_mask, dtype=torch.long),
        }

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        resp_a = self.responses_a[idx]
        resp_b = self.responses_b[idx]

        # 1. Scalar Features: Log-transformed lengths
        # Adding 1 to avoid log(0)
        feat_prompt = np.log1p(len(prompt))
        feat_resp_a = np.log1p(len(resp_a))
        feat_resp_b = np.log1p(len(resp_b))
        scalars = torch.tensor(
            [feat_prompt, feat_resp_a, feat_resp_b], dtype=torch.float32
        )

        # 2. Tokenization for Siamese Branches
        tokenized_a = self._tokenize_pair(prompt, resp_a)
        tokenized_b = self._tokenize_pair(prompt, resp_b)

        data = {
            "input_ids_a": tokenized_a["input_ids"],
            "attention_mask_a": tokenized_a["attention_mask"],
            "response_mask_a": tokenized_a["response_mask"],
            "input_ids_b": tokenized_b["input_ids"],
            "attention_mask_b": tokenized_b["attention_mask"],
            "response_mask_b": tokenized_b["response_mask"],
            "scalars": scalars,
        }

        if self.mode != "test":
            data["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        else:
            data["id"] = self.ids[idx]

        return data


def augment_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs Symmetric Augmentation.
    Creates a copy of the dataframe with A and B swapped, then concatenates.
    """
    df_aug = df.copy()

    # Swap Responses
    df_aug = df_aug.rename(
        columns={
            "response_a": "response_b",
            "response_b": "response_a",
            "model_a": "model_b",
            "model_b": "model_a",
            "winner_model_a": "winner_model_b",
            "winner_model_b": "winner_model_a",
        }
    )

    # Concatenate original and swapped
    # Reset index to ensure unique indices in the new dataframe
    df_combined = pd.concat([df, df_aug], axis=0).reset_index(drop=True)
    return df_combined


def get_dataloaders(tokenizer, load_cached_data=True):
    """
    Loads data, performs augmentation (with caching), and returns DataLoaders.
    """
    set_seed(Config.seed)

    cache_train_path = os.path.join(Config.cache_dir, "train_data_aug.parquet")
    cache_val_path = os.path.join(Config.cache_dir, "val_data.parquet")

    # --- Loading / Caching Logic ---
    train_df = None
    val_df = None

    if load_cached_data:
        if os.path.exists(cache_train_path) and os.path.exists(cache_val_path):
            print(f"Loading cached data from {Config.cache_dir}...")
            try:
                train_df = pd.read_parquet(cache_train_path)
                val_df = pd.read_parquet(cache_val_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
                train_df = None
                val_df = None
        else:
            print("Cache not found. Processing data...")

    if train_df is None:
        # Load Metadata
        print(f"Loading raw metadata from {Config.train_path} and {Config.val_path}...")
        train_df = pd.read_csv(Config.train_path)
        val_df = pd.read_csv(Config.val_path)

        # Apply Symmetric Augmentation to Train only
        print("Applying Symmetric Augmentation to training set...")
        train_df = augment_data(train_df)

        # Save to Cache
        print(f"Saving processed data to {Config.cache_dir}...")
        train_df.to_parquet(cache_train_path, index=False)
        val_df.to_parquet(cache_val_path, index=False)

    print(f"Training Data Shape: {train_df.shape}")
    print(f"Validation Data Shape: {val_df.shape}")

    # --- Dataset Creation ---
    train_dataset = ChatbotDataset(
        train_df, tokenizer, max_length=Config.max_length, mode="train"
    )

    val_dataset = ChatbotDataset(
        val_df, tokenizer, max_length=Config.max_length, mode="val"
    )

    # --- DataLoader Creation ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.n_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.n_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(tokenizer):
    """
    Loads test data and returns DataLoader.
    """
    set_seed(Config.seed)

    print(f"Loading test data from {Config.test_path}...")
    test_df = pd.read_csv(Config.test_path)

    test_dataset = ChatbotDataset(
        test_df, tokenizer, max_length=Config.max_length, mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.n_workers,
        pin_memory=True,
    )

    return test_loader
