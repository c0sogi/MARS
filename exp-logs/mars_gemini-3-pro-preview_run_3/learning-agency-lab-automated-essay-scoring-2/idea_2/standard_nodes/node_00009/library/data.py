import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything

# Ensure deterministic behavior
seed_everything()


def get_tokenizer():
    """
    Initializes and returns the tokenizer defined in Config.
    """
    return AutoTokenizer.from_pretrained(Config.MODEL_NAME)


def get_data(input_path: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads data from the given path with caching mechanism.

    Args:
        input_path (str): Path to the input CSV file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    # Define cache directory and filename
    cache_dir = os.path.join(Config.WORKING_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Create a unique cache filename based on the input filename and debug state
    base_name = os.path.basename(input_path).replace(".csv", "")
    if Config.DEBUG:
        base_name += "_debug"
    cache_path = os.path.join(cache_dir, f"{base_name}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            # If load fails, proceed to compute from scratch
            pass

    # 2. Compute/Process (Load from source)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Apply Debug Sampling
    if Config.DEBUG:
        df = df.iloc[: Config.DEBUG_SAMPLES].reset_index(drop=True)

    # 3. Save to cache
    df.to_parquet(cache_path, index=False)

    return df


class EssayDataset(Dataset):
    """
    Dataset for Stage 1 Training (Fine-tuning).
    Tokenizes text with truncation to max_length and returns regression targets.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = Config.MAX_LENGTH,
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = df["full_text"].values
        # Score might not exist in test set
        self.scores = df["score"].values if "score" in df.columns else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize with truncation and padding
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Squeeze to remove batch dimension added by tokenizer
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        item = {"input_ids": input_ids, "attention_mask": attention_mask}

        if self.scores is not None:
            # Regression target: float
            item["labels"] = torch.tensor(self.scores[idx], dtype=torch.float)

        return item


class SlidingWindowDataset(Dataset):
    """
    Dataset for Stage 2 Feature Extraction.
    Splits text into overlapping chunks to handle long essays.
    Returns a batch of chunks for a single essay.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        window_size: int = Config.WINDOW_SIZE,
        stride: int = Config.STRIDE,
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.stride = stride
        self.texts = df["full_text"].values
        self.ids = df["essay_id"].values

        # Calculate effective content length (window - [CLS] - [SEP])
        self.content_len = window_size - 2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        essay_id = self.ids[idx]

        # Tokenize without special tokens first to handle sliding window manually.
        # Truncation is False to capture the full text.
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]

        chunks_input_ids = []
        chunks_attention_mask = []

        total_tokens = len(tokens)

        # Handle empty text edge case
        if total_tokens == 0:
            input_ids = [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id] + [
                self.tokenizer.pad_token_id
            ] * (self.window_size - 2)
            mask = [1, 1] + [0] * (self.window_size - 2)
            chunks_input_ids.append(input_ids)
            chunks_attention_mask.append(mask)
        else:
            # Slide window over tokens
            for i in range(0, max(1, total_tokens), self.stride):
                # Stop if we've gone past the end (unless it's the first chunk)
                if i >= total_tokens and i > 0:
                    break

                # Extract slice
                chunk = tokens[i : i + self.content_len]

                # Construct valid input: [CLS] + chunk + [SEP]
                input_ids = (
                    [self.tokenizer.cls_token_id]
                    + chunk
                    + [self.tokenizer.sep_token_id]
                )
                mask = [1] * len(input_ids)

                # Pad to window_size
                pad_len = self.window_size - len(input_ids)
                if pad_len > 0:
                    input_ids.extend([self.tokenizer.pad_token_id] * pad_len)
                    mask.extend([0] * pad_len)

                chunks_input_ids.append(input_ids)
                chunks_attention_mask.append(mask)

                # If this chunk covered the end of the text, stop
                if i + self.content_len >= total_tokens:
                    break

        # Stack into tensors
        # Shape: (num_chunks, window_size)
        return {
            "essay_id": essay_id,
            "input_ids": torch.tensor(chunks_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(chunks_attention_mask, dtype=torch.long),
        }
