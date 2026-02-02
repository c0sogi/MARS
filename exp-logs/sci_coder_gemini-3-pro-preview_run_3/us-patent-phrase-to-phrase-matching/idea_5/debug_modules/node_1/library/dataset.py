import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from library.config import Config
from library.utils import get_logger
from library.cpc_utils import load_context_enriched_data

logger = get_logger("dataset")


class PhraseDataset(Dataset):
    """
    Dataset for the supervised phrase matching task.
    Constructs the input sequence:
    [CLS] Context Code [SEP] Context Description [SEP] Anchor [SEP] Target [SEP]
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        # Pre-fetch columns to numpy arrays for faster access
        self.contexts = df["context"].astype(str).values
        self.context_texts = df["context_text"].astype(str).values
        self.anchors = df["anchor"].astype(str).values
        self.targets = df["target"].astype(str).values

        if not self.is_test:
            self.scores = df["score"].values
            # Mapping for classification head
            self.score_map = {0.0: 0, 0.25: 1, 0.5: 2, 0.75: 3, 1.0: 4}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        context = self.contexts[idx]
        context_text = self.context_texts[idx]
        anchor = self.anchors[idx]
        target = self.targets[idx]

        # Manually construct input_ids to ensure correct [SEP] placement
        # 1. Tokenize components without special tokens
        tok_context = self.tokenizer.encode(context, add_special_tokens=False)
        tok_c_text = self.tokenizer.encode(context_text, add_special_tokens=False)
        tok_anchor = self.tokenizer.encode(anchor, add_special_tokens=False)
        tok_target = self.tokenizer.encode(target, add_special_tokens=False)

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        # 2. Concatenate: [CLS] Ctx [SEP] Desc [SEP] Anc [SEP] Tgt [SEP]
        input_ids = (
            [cls_id]
            + tok_context
            + [sep_id]
            + tok_c_text
            + [sep_id]
            + tok_anchor
            + [sep_id]
            + tok_target
            + [sep_id]
        )

        # 3. Truncate
        if len(input_ids) > self.max_len:
            # Keep the final [SEP]
            input_ids = input_ids[: self.max_len - 1] + [sep_id]

        # 4. Pad
        mask_len = len(input_ids)
        attention_mask = [1] * mask_len

        padding_len = self.max_len - mask_len
        if padding_len > 0:
            input_ids = input_ids + [pad_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len

        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

        if not self.is_test:
            score = self.scores[idx]
            out["labels"] = torch.tensor(score, dtype=torch.float)

            # Get bin label for classification head
            # Use get with default to handle potential float inaccuracies, though data is clean
            bin_label = self.score_map.get(score, 0)
            out["bin_labels"] = torch.tensor(bin_label, dtype=torch.long)

        return out


class MLMDataset(Dataset):
    """
    Dataset for Domain-Adaptive Pre-training (Masked Language Modeling).
    """

    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Standard tokenization for MLM
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )

        return {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
        }


def get_tokenizer():
    """Returns the tokenizer for the configured model."""
    return AutoTokenizer.from_pretrained(Config.model_name)


def get_train_dataloader(tokenizer):
    """Creates DataLoader for training data."""
    df = load_context_enriched_data("train", load_cached_data=True)

    if Config.debug:
        logger.info("Debug mode: utilizing small subset of train data.")
        df = df.head(100)

    dataset = PhraseDataset(df, tokenizer, Config.max_len, is_test=False)

    return DataLoader(
        dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_val_dataloader(tokenizer):
    """Creates DataLoader for validation data."""
    df = load_context_enriched_data("val", load_cached_data=True)

    if Config.debug:
        df = df.head(50)

    dataset = PhraseDataset(df, tokenizer, Config.max_len, is_test=False)

    return DataLoader(
        dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,  # No shuffle for validation
        num_workers=Config.num_workers,
        pin_memory=True,
    )


def get_test_dataloader(tokenizer):
    """Creates DataLoader for test data and returns IDs for submission."""
    df = load_context_enriched_data("test", load_cached_data=True)

    if Config.debug:
        df = df.head(50)

    dataset = PhraseDataset(df, tokenizer, Config.max_len, is_test=True)

    loader = DataLoader(
        dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )
    return loader, df["id"].values


def get_dapt_dataloader(tokenizer):
    """
    Creates DataLoader for Domain-Adaptive Pre-training.
    Aggregates unique texts from Train, Val, and Test sets.
    """
    logger.info("Preparing DAPT corpus...")

    # Load all available data
    train_df = load_context_enriched_data("train", load_cached_data=True)
    val_df = load_context_enriched_data("val", load_cached_data=True)
    test_df = load_context_enriched_data("test", load_cached_data=True)

    full_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)

    if Config.debug:
        full_df = full_df.head(200)

    # Extract unique texts from Context Descriptions, Anchors, and Targets
    unique_texts = set()

    # Add context descriptions (handle potential NaNs though enrichment handles it)
    unique_texts.update(full_df["context_text"].dropna().astype(str).unique())
    unique_texts.update(full_df["anchor"].dropna().astype(str).unique())
    unique_texts.update(full_df["target"].dropna().astype(str).unique())

    text_list = list(unique_texts)
    logger.info(f"DAPT Corpus: {len(text_list)} unique text segments found.")

    dataset = MLMDataset(text_list, tokenizer, Config.max_len)

    # Data Collator handles the random masking
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.mlm_probability
    )

    return DataLoader(
        dataset,
        batch_size=Config.dapt_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        collate_fn=data_collator,
    )
