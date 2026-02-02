import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

# Import from provided library files
from library.config import Config
from library.utils import decode_text


def get_tokenizer():
    """
    Loads and returns the tokenizer defined in Config.
    """
    return AutoTokenizer.from_pretrained(Config.tokenizer_name)


class InsultDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning (SFT) and Inference.
    Returns input_ids, attention_mask, and labels (if available).
    """

    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


class MLMDataset(Dataset):
    """
    Dataset for Domain-Adaptive Pre-training (DAPT).
    Returns tokenized inputs suitable for Masked Language Modeling.
    """

    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # For MLM, we just need to return the tokenized sequence.
        # The masking is typically handled by the DataCollatorForLanguageModeling.
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }


def load_dapt_data(load_cached_data=True):
    """
    Prepares data for Domain-Adaptive Pre-training.
    Concatenates Train, Val, and Test texts.
    """
    cache_path = os.path.join(Config.cache_dir, "dapt_data.parquet")
    os.makedirs(Config.cache_dir, exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached DAPT data from {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        print("Processing DAPT data from scratch...")
        # Load all metadata files
        df_train = pd.read_csv(Config.train_path)
        df_val = pd.read_csv(Config.val_path)
        df_test = pd.read_csv(Config.test_path)

        # Extract comments
        texts_train = df_train["Comment"].apply(decode_text)
        texts_val = df_val["Comment"].apply(decode_text)
        texts_test = df_test["Comment"].apply(decode_text)

        # Concatenate
        all_texts = pd.concat([texts_train, texts_val, texts_test], axis=0).reset_index(
            drop=True
        )
        df = pd.DataFrame({"Comment": all_texts})

        # Save to cache
        df.to_parquet(cache_path, index=False)
        print(f"Saved DAPT data to {cache_path}")

    return df


def load_supervised_data(load_cached_data=True):
    """
    Prepares data for Supervised Fine-Tuning.
    Concatenates Train and Val datasets (as per strategy to use full data).
    """
    cache_path = os.path.join(Config.cache_dir, "supervised_data.parquet")
    os.makedirs(Config.cache_dir, exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached Supervised data from {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        print("Processing Supervised data from scratch...")
        df_train = pd.read_csv(Config.train_path)
        df_val = pd.read_csv(Config.val_path)

        # Apply decoding
        df_train["Comment"] = df_train["Comment"].apply(decode_text)
        df_val["Comment"] = df_val["Comment"].apply(decode_text)

        # Concatenate
        df = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

        # Save to cache
        df.to_parquet(cache_path, index=False)
        print(f"Saved Supervised data to {cache_path}")

    return df


def load_test_data(load_cached_data=True):
    """
    Prepares data for Inference.
    """
    cache_path = os.path.join(Config.cache_dir, "test_data.parquet")
    os.makedirs(Config.cache_dir, exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached Test data from {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        print("Processing Test data from scratch...")
        df = pd.read_csv(Config.test_path)
        df["Comment"] = df["Comment"].apply(decode_text)

        # Save to cache
        df.to_parquet(cache_path, index=False)
        print(f"Saved Test data to {cache_path}")

    return df


def create_dataloaders(stage, tokenizer, load_cached_data=True):
    """
    Creates DataLoaders for the specified stage ('dapt', 'supervised', 'test').

    Args:
        stage (str): One of 'dapt', 'supervised', 'test'.
        tokenizer: The tokenizer instance.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        DataLoader or tuple of DataLoaders depending on stage.
    """

    if stage == "dapt":
        df = load_dapt_data(load_cached_data=load_cached_data)
        dataset = MLMDataset(
            texts=df["Comment"].values,
            tokenizer=tokenizer,
            max_length=Config.max_length,
        )

        # For MLM, we need a collator that handles masking
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=Config.mlm_probability
        )

        dataloader = DataLoader(
            dataset,
            batch_size=Config.dapt_batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            collate_fn=data_collator,
            pin_memory=True,
        )
        return dataloader

    elif stage == "supervised":
        df = load_supervised_data(load_cached_data=load_cached_data)

        # In the Seed Averaging strategy, we use the FULL dataset for training.
        # However, for code completeness/debugging, if we wanted a split, we'd do it here.
        # But per instructions: "Train 3 independent models on entire labeled dataset".
        # So we return a single training loader.

        dataset = InsultDataset(
            texts=df["Comment"].values,
            labels=df["Insult"].values,
            tokenizer=tokenizer,
            max_length=Config.max_length,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=Config.train_batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
        )
        return dataloader

    elif stage == "test":
        df = load_test_data(load_cached_data=load_cached_data)

        dataset = InsultDataset(
            texts=df["Comment"].values,
            labels=None,  # No labels for test
            tokenizer=tokenizer,
            max_length=Config.max_length,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=Config.valid_batch_size,  # Use larger batch size for inference
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )
        return dataloader

    else:
        raise ValueError(f"Unknown stage: {stage}")
