import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import get_logger
from library.cpc_loader import ContextMapper

# Initialize logger
logger = get_logger(os.path.join(Config.output_dir, "dataset.log"))


class PhraseDataset(Dataset):
    """
    PyTorch Dataset for the Phrase Matching Task.
    Tokenizes inputs using the DeBERTa tokenizer format:
    [CLS] Context [SEP] Anchor [SEP] Target [SEP]
    """

    def __init__(self, df, tokenizer, max_length=140, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'anchor', 'target', 'context_text', and optionally 'score'.
            tokenizer (PreTrainedTokenizer): Transformer tokenizer.
            max_length (int): Maximum sequence length.
            is_test (bool): Whether this is a test set (no labels).
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        # Pre-extract columns to lists for faster access
        self.anchors = df["anchor"].astype(str).tolist()
        self.targets = df["target"].astype(str).tolist()
        self.contexts = df["context_text"].astype(str).tolist()

        if not self.is_test:
            self.scores = df["score"].astype(float).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        target = self.targets[idx]
        context = self.contexts[idx]

        # Construct input text
        # We want the structure: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
        # We treat 'Context' as the first sentence, and 'Anchor [SEP] Target' as the second.
        # This leverages the tokenizer's ability to handle pairs and add special tokens.

        # Note: DeBERTa V3 tokenizer handles the [SEP] token correctly.
        # We manually insert a separator between anchor and target for the second segment.
        second_segment = anchor + self.tokenizer.sep_token + target

        inputs = self.tokenizer(
            text=context,
            text_pair=second_segment,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Squeeze to remove batch dimension added by tokenizer
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # DeBERTa might not return token_type_ids by default depending on config,
        # but we include it if present (usually 0 for context, 1 for anchor/target)
        token_type_ids = (
            inputs["token_type_ids"].squeeze(0)
            if "token_type_ids" in inputs
            else torch.zeros_like(input_ids)
        )

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        if not self.is_test:
            item["label"] = torch.tensor(self.scores[idx], dtype=torch.float)

        return item


def prepare_loaders(tokenizer, load_cached_data=True):
    """
    Loads data, processes contexts, and creates DataLoaders.
    Implements caching for the processed DataFrames.

    Args:
        tokenizer: The AutoTokenizer instance.
        load_cached_data (bool): Whether to attempt loading from parquet cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure output directory exists
    os.makedirs(Config.output_dir, exist_ok=True)

    # 1. Define Cache Paths
    train_cache = Config.train_cache_path
    val_cache = Config.val_cache_path
    test_cache = Config.test_cache_path

    # 2. Check Cache Availability
    cache_exists = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    )

    if load_cached_data and cache_exists:
        logger.info("Loading processed datasets from cache...")
        df_train = pd.read_parquet(train_cache)
        df_val = pd.read_parquet(val_cache)
        df_test = pd.read_parquet(test_cache)
    else:
        logger.info("Cache not found or ignored. Processing data from scratch...")

        # Load Metadata
        if not os.path.exists(Config.train_path):
            raise FileNotFoundError(f"Metadata file not found: {Config.train_path}")

        df_train = pd.read_csv(Config.train_path)
        df_val = pd.read_csv(Config.val_path)
        df_test = pd.read_csv(Config.test_path)

        # Initialize and Fit Context Mapper
        # We collect all unique contexts to minimize mapping overhead
        all_contexts = pd.concat(
            [df_train["context"], df_val["context"], df_test["context"]]
        ).unique()

        mapper = ContextMapper()
        mapper.fit(all_contexts, load_cached_data=load_cached_data)

        # Apply Mapping
        logger.info("Mapping context codes to text descriptions...")
        df_train["context_text"] = df_train["context"].map(mapper.context_map)
        df_val["context_text"] = df_val["context"].map(mapper.context_map)
        df_test["context_text"] = df_test["context"].map(mapper.context_map)

        # Fill any missing mappings (fallback safety)
        df_train["context_text"] = df_train["context_text"].fillna("")
        df_val["context_text"] = df_val["context_text"].fillna("")
        df_test["context_text"] = df_test["context_text"].fillna("")

        # Save to Cache
        logger.info(f"Saving processed datasets to {Config.output_dir}...")
        df_train.to_parquet(train_cache, index=False)
        df_val.to_parquet(val_cache, index=False)
        df_test.to_parquet(test_cache, index=False)

    # 3. Handle Debug Mode
    if Config.debug:
        logger.info("Debug mode enabled: Truncating datasets to 100 samples.")
        df_train = df_train.iloc[:100].reset_index(drop=True)
        df_val = df_val.iloc[:100].reset_index(drop=True)
        df_test = df_test.iloc[:100].reset_index(drop=True)

    logger.info(f"Train set size: {len(df_train)}")
    logger.info(f"Val set size:   {len(df_val)}")
    logger.info(f"Test set size:  {len(df_test)}")

    # 4. Create Datasets
    train_dataset = PhraseDataset(
        df_train, tokenizer, max_length=Config.max_length, is_test=False
    )
    val_dataset = PhraseDataset(
        df_val, tokenizer, max_length=Config.max_length, is_test=False
    )
    test_dataset = PhraseDataset(
        df_test, tokenizer, max_length=Config.max_length, is_test=True
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.inference_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
