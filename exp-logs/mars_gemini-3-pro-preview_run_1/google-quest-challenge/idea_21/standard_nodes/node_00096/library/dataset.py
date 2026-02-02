import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything


def load_data(mode, config, load_cached_data=True):
    """
    Loads data from CSV or Parquet cache.

    Args:
        mode (str): 'train', 'val', or 'test'.
        config (Config): Configuration object.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    if mode == "train":
        input_path = config.TRAIN_PATH
        cache_path = config.TRAIN_CACHE
    elif mode == "val":
        input_path = config.VAL_PATH
        cache_path = config.VAL_CACHE
    elif mode == "test":
        input_path = config.TEST_PATH
        cache_path = config.TEST_CACHE
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(
                f"Failed to load cache from {cache_path}: {e}. Reloading from source."
            )

    # 2. Process from Source
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Basic text cleaning: ensure strings
    text_cols = ["question_title", "question_body", "answer"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # 3. Save Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class QuestDataset(Dataset):
    def __init__(self, df, tokenizer, config, mode="train"):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        self.mode = mode
        self.max_len = config.MAX_LEN

        # Identify target columns
        # We read sample submission to get the correct order and names of targets
        sample_sub = pd.read_csv(config.SAMPLE_SUB_PATH, nrows=1)
        self.target_cols = [c for c in sample_sub.columns if c != "qa_id"]

        # Check if targets exist in df (they won't for test set)
        self.has_targets = all(col in self.df.columns for col in self.target_cols)

        if self.has_targets:
            self.labels = self.df[self.target_cols].values.astype(float)
        else:
            self.labels = None

        # Pre-extract text data to avoid dataframe overhead in __getitem__
        self.q_titles = self.df["question_title"].values
        self.q_bodies = self.df["question_body"].values
        self.answers = self.df["answer"].values
        self.ids = self.df["qa_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        title = self.q_titles[idx]
        body = self.q_bodies[idx]
        answer = self.answers[idx]

        # Tokenize Question Stream: [CLS] Title [SEP] Body [SEP]
        # tokenizer(text, text_pair) handles the special tokens automatically
        q_enc = self.tokenizer(
            title,
            body,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=False,  # RoBERTa doesn't use token_type_ids usually, but safe to ignore
        )

        # Tokenize Answer Stream: [CLS] Answer [SEP]
        a_enc = self.tokenizer(
            answer,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        item = {
            "qa_id": self.ids[idx],
            "input_ids_q": q_enc["input_ids"],
            "attention_mask_q": q_enc["attention_mask"],
            "input_ids_a": a_enc["input_ids"],
            "attention_mask_a": a_enc["attention_mask"],
        }

        if self.has_targets:
            item["labels"] = self.labels[idx]

        return item


class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        # Extract lists
        input_ids_q = [
            torch.tensor(item["input_ids_q"], dtype=torch.long) for item in batch
        ]
        attention_mask_q = [
            torch.tensor(item["attention_mask_q"], dtype=torch.long) for item in batch
        ]
        input_ids_a = [
            torch.tensor(item["input_ids_a"], dtype=torch.long) for item in batch
        ]
        attention_mask_a = [
            torch.tensor(item["attention_mask_a"], dtype=torch.long) for item in batch
        ]

        qa_ids = [item["qa_id"] for item in batch]

        # Dynamic Padding
        # batch_first=True makes output (Batch, Seq_Len)
        input_ids_q = pad_sequence(
            input_ids_q, batch_first=True, padding_value=self.pad_token_id
        )
        attention_mask_q = pad_sequence(
            attention_mask_q, batch_first=True, padding_value=0
        )

        input_ids_a = pad_sequence(
            input_ids_a, batch_first=True, padding_value=self.pad_token_id
        )
        attention_mask_a = pad_sequence(
            attention_mask_a, batch_first=True, padding_value=0
        )

        batch_out = {
            "qa_id": torch.tensor(qa_ids, dtype=torch.long),
            "input_ids_q": input_ids_q,
            "attention_mask_q": attention_mask_q,
            "input_ids_a": input_ids_a,
            "attention_mask_a": attention_mask_a,
        }

        if "labels" in batch[0]:
            labels = [torch.tensor(item["labels"], dtype=torch.float) for item in batch]
            batch_out["labels"] = torch.stack(labels)

        return batch_out


def get_dataloaders(config, load_cached_data=True):
    """
    Factory function to create dataloaders.
    """
    seed_everything(config.SEED)

    tokenizer = AutoTokenizer.from_pretrained(config.BACKBONE)

    loaders = {}

    # Helper to create loader
    def create_loader(mode, shuffle):
        df = load_data(mode, config, load_cached_data=load_cached_data)

        if config.DEBUG:
            df = df.head(config.SUBSET_SIZE).copy()

        dataset = QuestDataset(df, tokenizer, config, mode=mode)
        collate_fn = Collate(tokenizer)

        loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=(mode == "train"),  # Drop last for train to maintain batch stats
        )
        return loader

    # Create loaders based on existence of metadata
    if os.path.exists(config.TRAIN_PATH):
        loaders["train"] = create_loader("train", shuffle=True)

    if os.path.exists(config.VAL_PATH):
        loaders["val"] = create_loader("val", shuffle=False)

    if os.path.exists(config.TEST_PATH):
        loaders["test"] = create_loader("test", shuffle=False)

    return loaders
