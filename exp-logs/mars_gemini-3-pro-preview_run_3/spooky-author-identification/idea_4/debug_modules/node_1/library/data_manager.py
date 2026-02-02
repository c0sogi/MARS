import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config

# Global label mapping for consistency
LABEL_MAP = {"EAP": 0, "HPL": 1, "MWS": 2}


class AuthorDataset(Dataset):
    """
    Custom PyTorch Dataset for Author Identification.
    Handles tokenization and label encoding.
    """

    def __init__(
        self, texts, labels=None, tokenizer=None, max_length=Config.MAX_LENGTH
    ):
        """
        Args:
            texts (list or pd.Series): List of text samples.
            labels (list or pd.Series, optional): List of author labels. Defaults to None.
            tokenizer (PreTrainedTokenizer): Transformer tokenizer.
            max_length (int): Maximum sequence length for tokenization.
        """
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Flatten to remove batch dimension added by tokenizer
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        # Handle labels if they exist
        if self.labels is not None:
            label_val = self.labels[idx]
            # Convert string label to integer if necessary
            if isinstance(label_val, str):
                label_val = LABEL_MAP.get(label_val, 0)

            item["labels"] = torch.tensor(label_val, dtype=torch.long)

        return item


def get_tokenizer(model_name):
    """
    Initializes and returns the tokenizer for the specified model.
    """
    return AutoTokenizer.from_pretrained(model_name)


def load_raw_data(debug=False, debug_size=Config.DEBUG_SAMPLE_SIZE):
    """
    Loads raw data from the metadata directory defined in Config.

    Args:
        debug (bool): If True, loads only a small subset of data.
        debug_size (int): Number of samples to load in debug mode.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Use paths from Config
    train_path = Config.TRAIN_DATA_PATH
    val_path = Config.VAL_DATA_PATH
    test_path = Config.TEST_DATA_PATH

    # Load CSVs
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    if debug:
        train_df = train_df.head(debug_size)
        val_df = val_df.head(debug_size)
        test_df = test_df.head(debug_size)

    return train_df, val_df, test_df


def prepare_mlm_corpus(train_df, val_df, test_df, load_cached_data=True):
    """
    Prepares the corpus for Masked Language Modeling by concatenating text
    from all splits (train, val, test).

    Implements strict caching logic using Parquet.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        list: A list of text strings representing the corpus.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "mlm_corpus.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached MLM corpus from {cache_path}")
            df_corpus = pd.read_parquet(cache_path)
            return df_corpus["text"].tolist()
        except Exception as e:
            print(f"Failed to load cache (Error: {e}). Recomputing...")

    # 2. Compute from scratch (if cache missing, corrupt, or load_cached_data=False)
    print("Preparing MLM corpus from scratch...")

    # Concatenate all available text for domain adaptation
    all_texts = pd.concat(
        [train_df["text"], val_df["text"], test_df["text"]], axis=0, ignore_index=True
    )

    # Create DataFrame for Parquet storage
    df_corpus = pd.DataFrame({"text": all_texts})

    # Save to cache
    df_corpus.to_parquet(cache_path, index=False)
    print(f"Saved MLM corpus to {cache_path}")

    return df_corpus["text"].tolist()
