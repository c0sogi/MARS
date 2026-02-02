import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


def prepare_data(
    config: Config, partition: str = "train", load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Loads, preprocesses, and optionally augments data.
    Handles caching to parquet files in the working directory.

    Args:
        config: Configuration object.
        partition: 'train', 'val', or 'test'.
        load_cached_data: Whether to try loading from cache.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    cache_path = config.get_cache_path(partition)

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If load fails, proceed to process from scratch
            pass

    # 2. Load metadata
    if partition == "train":
        input_path = config.train_path
    elif partition == "val":
        input_path = config.val_path
    elif partition == "test":
        input_path = config.test_path
    else:
        raise ValueError(f"Unknown partition: {partition}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # 3. Debug subset
    if config.debug:
        df = df.head(config.debug_sample_size).copy()

    # 4. Symmetric Augmentation (Train only)
    # Double the dataset by swapping (A, B) -> (B, A) with inverted targets
    if partition == "train" and config.use_symmetric_augmentation:
        swapped_df = df.copy()

        # Swap response columns and model names
        swapped_df.rename(
            columns={
                "response_a": "response_b_temp",
                "response_b": "response_a_temp",
                "model_a": "model_b_temp",
                "model_b": "model_a_temp",
                "winner_model_a": "winner_model_b_temp",
                "winner_model_b": "winner_model_a_temp",
            },
            inplace=True,
        )

        swapped_df.rename(
            columns={
                "response_b_temp": "response_b",
                "response_a_temp": "response_a",
                "model_b_temp": "model_b",
                "model_a_temp": "model_a",
                "winner_model_b_temp": "winner_model_b",
                "winner_model_a_temp": "winner_model_a",
            },
            inplace=True,
        )

        # winner_tie remains the same

        # Concatenate original and swapped
        df = pd.concat([df, swapped_df], axis=0, ignore_index=True)

    # 5. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        df.to_parquet(cache_path, index=False)
    except Exception:
        # If save fails (e.g. disk full), just continue
        pass

    return df


class ChatbotDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, tokenizer, max_length: int, mode: str = "train"
    ):
        """
        Args:
            df: Preprocessed DataFrame.
            tokenizer: Transformers tokenizer.
            max_length: Maximum sequence length.
            mode: 'train', 'val', or 'test'.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

        # Convert columns to lists for efficiency during __getitem__
        self.ids = df["id"].tolist()
        self.prompts = df["prompt"].fillna("").astype(str).tolist()
        self.responses_a = df["response_a"].fillna("").astype(str).tolist()
        self.responses_b = df["response_b"].fillna("").astype(str).tolist()

        # Targets are only available in train/val
        if self.mode != "test":
            self.targets = df[
                ["winner_model_a", "winner_model_b", "winner_tie"]
            ].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        resp_a = self.responses_a[idx]
        resp_b = self.responses_b[idx]

        # --- Scalar Features ---
        # Calculate log-transformed lengths of tokens.
        # We tokenize without special tokens to get raw content length.
        tok_prompt = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        tok_resp_a = self.tokenizer(resp_a, add_special_tokens=False)["input_ids"]
        tok_resp_b = self.tokenizer(resp_b, add_special_tokens=False)["input_ids"]

        len_prompt = len(tok_prompt)
        len_resp_a = len(tok_resp_a)
        len_resp_b = len(tok_resp_b)

        features = torch.tensor(
            [np.log1p(len_prompt), np.log1p(len_resp_a), np.log1p(len_resp_b)],
            dtype=torch.float32,
        )

        # --- Model Inputs ---
        # Tokenize Prompt + Response with truncation on Response only.
        # This preserves the Prompt context which is crucial for the "Decoupled Pooling".
        # We do NOT pad here; we pad in collate_fn for efficiency.

        encoded_a = self.tokenizer(
            prompt,
            resp_a,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        encoded_b = self.tokenizer(
            prompt,
            resp_b,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        item = {
            "id": self.ids[idx],
            "input_ids_a": encoded_a["input_ids"],
            "attention_mask_a": encoded_a["attention_mask"],
            "token_type_ids_a": encoded_a.get(
                "token_type_ids", [0] * len(encoded_a["input_ids"])
            ),
            "input_ids_b": encoded_b["input_ids"],
            "attention_mask_b": encoded_b["attention_mask"],
            "token_type_ids_b": encoded_b.get(
                "token_type_ids", [0] * len(encoded_b["input_ids"])
            ),
            "features": features,
        }

        if self.mode != "test":
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


class CollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Extract components
        ids = [x["id"] for x in batch]
        features = torch.stack([x["features"] for x in batch])

        input_ids_a = [x["input_ids_a"] for x in batch]
        attention_mask_a = [x["attention_mask_a"] for x in batch]
        token_type_ids_a = [x["token_type_ids_a"] for x in batch]

        input_ids_b = [x["input_ids_b"] for x in batch]
        attention_mask_b = [x["attention_mask_b"] for x in batch]
        token_type_ids_b = [x["token_type_ids_b"] for x in batch]

        # Helper for padding
        def pad_list(seqs, pad_val):
            max_len = max(len(s) for s in seqs)
            # Create tensor of shape (batch, max_len) filled with pad_val
            padded = torch.full((len(seqs), max_len), pad_val, dtype=torch.long)
            for i, s in enumerate(seqs):
                padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            return padded

        # Use pad_token_id from tokenizer, default to 0 if not set
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )

        batch_input_ids_a = pad_list(input_ids_a, pad_id)
        batch_mask_a = pad_list(attention_mask_a, 0)
        batch_type_ids_a = pad_list(token_type_ids_a, 0)

        batch_input_ids_b = pad_list(input_ids_b, pad_id)
        batch_mask_b = pad_list(attention_mask_b, 0)
        batch_type_ids_b = pad_list(token_type_ids_b, 0)

        out = {
            "id": ids,
            "input_ids_a": batch_input_ids_a,
            "attention_mask_a": batch_mask_a,
            "token_type_ids_a": batch_type_ids_a,
            "input_ids_b": batch_input_ids_b,
            "attention_mask_b": batch_mask_b,
            "token_type_ids_b": batch_type_ids_b,
            "features": features,
        }

        if "target" in batch[0]:
            out["target"] = torch.stack([x["target"] for x in batch])

        return out


def get_dataloader(
    config: Config, partition: str = "train", load_cached_data: bool = True
):
    """
    Factory function to create a DataLoader for a specific partition.
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Prepare data
    df = prepare_data(config, partition, load_cached_data)

    # Determine mode
    mode = "test" if partition == "test" else "train"
    if partition == "val":
        mode = "val"

    # Create Dataset
    dataset = ChatbotDataset(df, tokenizer, config.max_length, mode=mode)

    # Create CollateFn
    collate_fn = CollateFn(tokenizer)

    # Determine DataLoader params
    shuffle = partition == "train"
    batch_size = (
        config.train_batch_size if partition == "train" else config.valid_batch_size
    )

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return loader
