import os
import ast
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def clean_text(text):
    """
    Cleans the text column by handling unicode escapes and quote artifacts.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Attempt to use literal_eval to handle python-style string escaping and quotes
    try:
        if text.startswith('"') and text.endswith('"'):
            cleaned = ast.literal_eval(text)
            return cleaned
    except (ValueError, SyntaxError):
        pass

    # Fallback cleanup if literal_eval fails
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    try:
        text = text.encode("utf-8").decode("unicode_escape")
    except:
        pass

    return text


def load_and_process_data(load_cached_data=True):
    """
    Loads data from metadata, cleans it, and caches it to parquet.
    """
    cache_dir = Config.output_dir
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "train": (Config.train_path, os.path.join(cache_dir, "train_cleaned.parquet")),
        "val": (Config.val_path, os.path.join(cache_dir, "val_cleaned.parquet")),
        "test": (Config.test_path, os.path.join(cache_dir, "test_cleaned.parquet")),
    }

    dfs = {}

    for key, (input_path, cache_path) in files.items():
        loaded = False
        if load_cached_data and os.path.exists(cache_path):
            try:
                dfs[key] = pd.read_parquet(cache_path)
                loaded = True
            except Exception as e:
                print(f"Failed to load cache for {key}: {e}")

        if not loaded:
            df = pd.read_csv(input_path)
            # Clean text
            if Config.text_col in df.columns:
                df[Config.text_col] = df[Config.text_col].apply(clean_text)

            # Save cache
            df.to_parquet(cache_path, index=False)
            dfs[key] = df

    return dfs["train"], dfs["val"], dfs["test"]


class InsultDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.texts = df[Config.text_col].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        if not is_test:
            self.labels = df[Config.target_col].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        if not self.is_test:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


class MLMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len, mlm_probability=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_probability = mlm_probability

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].flatten()
        attention_mask = inputs["attention_mask"].flatten()
        labels = input_ids.clone()

        # Create mask
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Mask special tokens (0 probability)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels.tolist(), already_has_special_tokens=True
        )
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )

        # Mask padding tokens (0 probability)
        if self.tokenizer.pad_token_id is not None:
            probability_matrix.masked_fill_(
                labels == self.tokenizer.pad_token_id, value=0.0
            )

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set labels for unmasked tokens to -100 (ignored in loss)
        labels[~masked_indices] = -100

        # 80% replace with [MASK]
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% replace with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        # The remaining 10% are kept original (masked_indices is True, but input_ids not changed)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def get_dataloaders(tokenizer, load_cached_data=True):
    """
    Prepares and returns DataLoaders for TAPT, Train, Val, and Test.
    """
    # Load and clean data
    df_train, df_val, df_test = load_and_process_data(load_cached_data=load_cached_data)

    # Debugging: Subset data
    if Config.debug:
        df_train = df_train.head(Config.debug_subset_size)
        df_val = df_val.head(Config.debug_subset_size)
        df_test = df_test.head(Config.debug_subset_size)
        print(f"DEBUG Mode: Subsetting data to {Config.debug_subset_size} rows.")

    # 1. TAPT Dataset (Unsupervised)
    # Combine all texts for domain adaptation
    all_texts = np.concatenate(
        [
            df_train[Config.text_col].values,
            df_val[Config.text_col].values,
            df_test[Config.text_col].values,
        ]
    )

    tapt_dataset = MLMDataset(
        texts=all_texts,
        tokenizer=tokenizer,
        max_len=Config.max_len,
        mlm_probability=Config.mlm_probability,
    )

    tapt_loader = DataLoader(
        tapt_dataset,
        batch_size=Config.tapt_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # 2. Supervised Datasets
    train_dataset = InsultDataset(df_train, tokenizer, Config.max_len)
    val_dataset = InsultDataset(df_val, tokenizer, Config.max_len)
    test_dataset = InsultDataset(df_test, tokenizer, Config.max_len, is_test=True)

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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    print(f"TAPT Dataset Size: {len(tapt_dataset)}")
    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size:   {len(val_dataset)}")
    print(f"Test Dataset Size:  {len(test_dataset)}")

    return tapt_loader, train_loader, val_loader, test_loader
