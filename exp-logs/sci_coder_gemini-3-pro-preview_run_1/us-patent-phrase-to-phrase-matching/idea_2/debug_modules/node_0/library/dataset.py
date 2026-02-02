import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


def load_and_preprocess_data(
    split: str, load_cached_data: bool = True, debug: bool = Config.debug
):
    """
    Loads and preprocesses data for the specified split.
    Implements caching to parquet format to speed up subsequent loads.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, returns a small subset of the data.

    Returns:
        pd.DataFrame: The loaded data.
    """
    # Ensure cache directory exists
    os.makedirs(Config.cache_dir, exist_ok=True)

    cache_path = os.path.join(Config.cache_dir, f"cached_{split}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # If debug is True but cached data is full size, we slice it.
            # If cached data was already debug size, we just return it.
            if debug and len(df) > Config.debug_sample_size:
                df = df.head(Config.debug_sample_size)
            print(f"Loaded {split} data from cache: {cache_path}")
            return df
        except Exception as e:
            print(f"Failed to load cache for {split}: {e}. Reloading from source.")

    # 2. Load from source
    print(f"Loading {split} data from source...")
    if split == "train":
        file_path = Config.train_path
    elif split == "val":
        file_path = Config.val_path
    elif split == "test":
        file_path = Config.test_path
    else:
        raise ValueError(f"Invalid split name: {split}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Basic preprocessing: Ensure string types
    text_cols = ["anchor", "target", "context"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("")

    # 3. Save to cache
    try:
        # Save full dataframe before debug slicing
        df.to_parquet(cache_path, index=False)
        print(f"Saved {split} data to cache: {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache for {split}: {e}")

    # 4. Apply debug slicing if requested
    if debug:
        df = df.head(Config.debug_sample_size)
        print(f"Debug mode: sampled {len(df)} rows.")

    return df


class PhraseDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=Config.max_length, mode="train"):
        """
        Dataset class for Patent Phrase Similarity.

        Args:
            df (pd.DataFrame): DataFrame containing 'anchor', 'target', 'context', and optionally 'score'.
            tokenizer: PreTrainedTokenizer (e.g. DeBERTa tokenizer).
            max_length (int): Maximum sequence length for tokenization.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

        # Pre-fetch columns to avoid overhead in __getitem__
        self.anchors = self.df["anchor"].tolist()
        self.targets = self.df["target"].tolist()
        self.contexts = self.df["context"].tolist()

        if self.mode != "test" and "score" in self.df.columns:
            self.scores = self.df["score"].astype(float).tolist()
        else:
            self.scores = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        target = self.targets[idx]
        context = self.contexts[idx]

        # Context Injection Strategy:
        # We want the model to see: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
        # Using tokenizer(text, text_pair) with DeBERTa usually produces:
        # [CLS] text [SEP] text_pair [SEP]
        # So we construct:
        # text = context + [SEP] + anchor
        # text_pair = target

        sep = self.tokenizer.sep_token
        first_segment = f"{context}{sep}{anchor}"
        second_segment = target

        # Tokenize
        inputs = self.tokenizer(
            first_segment,
            second_segment,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        # Convert to tensors
        # inputs values are lists, so we convert them to LongTensors
        input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if "token_type_ids" in inputs:
            sample["token_type_ids"] = torch.tensor(
                inputs["token_type_ids"], dtype=torch.long
            )

        # Add labels if available
        if self.scores is not None:
            label = self.scores[idx]
            # Regression target: float
            sample["labels"] = torch.tensor(label, dtype=torch.float)

        return sample
