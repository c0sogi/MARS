import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything, get_logger

# Initialize logger
logger = get_logger("data_module")


def load_dataset_df(mode="train", load_cached_data=True):
    """
    Loads the dataset dataframe.
    - Checks cache first (parquet).
    - If not in cache, loads from metadata CSVs.
    - Applies symmetric augmentation if mode == 'train'.
    - Saves to cache.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{mode}_data.parquet")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading {mode} data from cache: {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Reloading from source.")

    # 2. Load from source
    logger.info(f"Loading {mode} data from source metadata...")
    if mode == "train":
        source_path = Config.TRAIN_PATH
    elif mode == "val":
        source_path = Config.VAL_PATH
    elif mode == "test":
        source_path = Config.TEST_PATH
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    df = pd.read_csv(source_path)

    # 3. Apply Augmentation (Only for Train)
    if mode == "train" and Config.USE_SYMMETRIC_AUGMENTATION:
        logger.info("Applying symmetric augmentation to training data...")
        # Create swapped copy
        df_swapped = df.copy()

        # Swap responses
        df_swapped = df_swapped.rename(
            columns={
                "response_a": "response_b_temp",
                "response_b": "response_a_temp",
                "winner_model_a": "winner_model_b_temp",
                "winner_model_b": "winner_model_a_temp",
            }
        )

        # Fix column names
        df_swapped = df_swapped.rename(
            columns={
                "response_b_temp": "response_b",
                "response_a_temp": "response_a",
                "winner_model_b_temp": "winner_model_b",
                "winner_model_a_temp": "winner_model_a",
            }
        )

        # Concatenate
        df = pd.concat([df, df_swapped], axis=0, ignore_index=True)
        logger.info(f"Augmented train size: {len(df)}")

    # 4. Save to cache
    logger.info(f"Saving {mode} data to cache: {cache_path}")
    df.to_parquet(cache_path, index=False)

    return df


class ChatbotDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        # Pre-extract columns to lists for faster access
        self.prompts = df["prompt"].fillna("").astype(str).tolist()
        self.responses_a = df["response_a"].fillna("").astype(str).tolist()
        self.responses_b = df["response_b"].fillna("").astype(str).tolist()

        # Pre-calculate prompt lengths for scalar features to save time in __getitem__
        # We use a simple encode without special tokens to estimate content length
        logger.info(f"Pre-calculating prompt lengths for {len(df)} samples...")
        self.prompt_lengths = [
            len(self.tokenizer.encode(p, add_special_tokens=False))
            for p in self.prompts
        ]

        if not self.is_test:
            self.targets = df[
                ["winner_model_a", "winner_model_b", "winner_tie"]
            ].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        resp_a = self.responses_a[idx]
        resp_b = self.responses_b[idx]
        len_prompt = self.prompt_lengths[idx]

        # Tokenize Branch A
        # We use truncation=True (longest_first) as a robust default.
        # Ideally we want to preserve prompt, but if prompt > 512, we must truncate.
        encoded_a = self.tokenizer(
            prompt,
            resp_a,
            truncation=True,
            max_length=self.max_len,
            padding=False,  # We pad in collator
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        # Tokenize Branch B
        encoded_b = self.tokenizer(
            prompt,
            resp_b,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        # Calculate Scalar Features (Log Lengths)
        # Total length includes [CLS], Prompt, [SEP], Response, [SEP] (approx 3 special tokens)
        len_total_a = len(encoded_a["input_ids"])
        len_total_b = len(encoded_b["input_ids"])

        # Estimate response length.
        # If truncation occurred, it likely affected the response first or both.
        # We clamp at 0 to avoid negative values in edge cases.
        len_resp_a = max(0, len_total_a - len_prompt - 3)
        len_resp_b = max(0, len_total_b - len_prompt - 3)

        # Log transform: log(x + 1)
        scalars = [np.log1p(len_prompt), np.log1p(len_resp_a), np.log1p(len_resp_b)]

        item = {
            "input_ids_a": encoded_a["input_ids"],
            "attention_mask_a": encoded_a["attention_mask"],
            "input_ids_b": encoded_b["input_ids"],
            "attention_mask_b": encoded_b["attention_mask"],
            "scalars": torch.tensor(scalars, dtype=torch.float32),
        }

        if "token_type_ids" in encoded_a:
            item["token_type_ids_a"] = encoded_a["token_type_ids"]
            item["token_type_ids_b"] = encoded_b["token_type_ids"]

        if not self.is_test:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


class CollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        # Extract lists
        input_ids_a = [x["input_ids_a"] for x in batch]
        attention_mask_a = [x["attention_mask_a"] for x in batch]
        input_ids_b = [x["input_ids_b"] for x in batch]
        attention_mask_b = [x["attention_mask_b"] for x in batch]
        scalars = torch.stack([x["scalars"] for x in batch])

        # Pad sequences helper
        def pad_seq(seqs, pad_val):
            max_len = max(len(s) for s in seqs)
            padded = []
            for s in seqs:
                p = s + [pad_val] * (max_len - len(s))
                padded.append(p)
            return torch.tensor(padded, dtype=torch.long)

        # Pad inputs
        input_ids_a = pad_seq(input_ids_a, self.pad_token_id)
        attention_mask_a = pad_seq(attention_mask_a, 0)
        input_ids_b = pad_seq(input_ids_b, self.pad_token_id)
        attention_mask_b = pad_seq(attention_mask_b, 0)

        batch_out = {
            "input_ids_a": input_ids_a,
            "attention_mask_a": attention_mask_a,
            "input_ids_b": input_ids_b,
            "attention_mask_b": attention_mask_b,
            "scalars": scalars,
        }

        # Handle token_type_ids if present (some tokenizers return them)
        if "token_type_ids_a" in batch[0]:
            token_type_ids_a = [x["token_type_ids_a"] for x in batch]
            token_type_ids_b = [x["token_type_ids_b"] for x in batch]
            batch_out["token_type_ids_a"] = pad_seq(token_type_ids_a, 0)
            batch_out["token_type_ids_b"] = pad_seq(token_type_ids_b, 0)

        if "target" in batch[0]:
            batch_out["target"] = torch.stack([x["target"] for x in batch])

        return batch_out


def get_dataloaders(load_cached_data=True):
    """
    Factory function to create dataloaders.
    """
    seed_everything(Config.SEED)

    # Load DataFrames
    train_df = load_dataset_df("train", load_cached_data)
    val_df = load_dataset_df("val", load_cached_data)
    test_df = load_dataset_df("test", load_cached_data)

    # Initialize Tokenizer
    # DeBERTa-v3 uses SentencePiece, ensure dependencies are met (sentencepiece is installed)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Create Datasets
    train_dataset = ChatbotDataset(train_df, tokenizer, Config.MAX_LEN, is_test=False)
    val_dataset = ChatbotDataset(val_df, tokenizer, Config.MAX_LEN, is_test=False)
    test_dataset = ChatbotDataset(test_df, tokenizer, Config.MAX_LEN, is_test=True)

    # Create Collator
    collator = CollateFn(tokenizer)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
