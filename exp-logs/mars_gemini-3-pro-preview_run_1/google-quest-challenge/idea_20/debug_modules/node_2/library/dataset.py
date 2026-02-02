import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything


class QuestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, mode="train"):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

        # Extract text columns
        self.titles = df["question_title"].values
        self.bodies = df["question_body"].values
        self.answers = df["answer"].values

        # Extract targets if not in test mode
        if self.mode != "test":
            # Identify target columns based on Config or convention
            # Using the last 30 columns as per task description and Config.NUM_TARGETS
            # However, safer to look for columns present in sample_submission (excluding qa_id)
            # We assume the df passed here has the targets.
            # Based on metadata generation, targets are present in train/val.

            # Use explicit target columns from Config to avoid schema mismatches
            # (e.g., 'filepath' column added by metadata generation)
            self.targets = df[Config.TARGET_COLS].values.astype(np.float32)

        self.qa_ids = df["qa_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        title = str(self.titles[idx])
        body = str(self.bodies[idx])
        answer = str(self.answers[idx])

        # Question Stream: Pair encoding of Title and Body
        # RoBERTa tokenizer handles the separation: <s> Title </s> </s> Body </s>
        q_inputs = self.tokenizer(
            title,
            body,
            add_special_tokens=True,
            max_length=self.max_len,
            padding=False,  # Dynamic padding in collate
            truncation=True,
            return_token_type_ids=True,
        )

        # Answer Stream: Single encoding
        a_inputs = self.tokenizer(
            answer,
            add_special_tokens=True,
            max_length=self.max_len,
            padding=False,
            truncation=True,
            return_token_type_ids=True,
        )

        item = {
            "input_ids_q": q_inputs["input_ids"],
            "attention_mask_q": q_inputs["attention_mask"],
            "input_ids_a": a_inputs["input_ids"],
            "attention_mask_a": a_inputs["attention_mask"],
            "qa_id": self.qa_ids[idx],
        }

        if self.mode != "test":
            item["targets"] = self.targets[idx]

        return item


class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Separate inputs
        input_ids_q = [item["input_ids_q"] for item in batch]
        attention_mask_q = [item["attention_mask_q"] for item in batch]
        input_ids_a = [item["input_ids_a"] for item in batch]
        attention_mask_a = [item["attention_mask_a"] for item in batch]
        qa_ids = [item["qa_id"] for item in batch]

        # Pad sequences
        # We use the tokenizer's pad method or manually pad
        # transformers tokenizer.pad accepts a dict of lists

        q_batch = self.tokenizer.pad(
            {"input_ids": input_ids_q, "attention_mask": attention_mask_q},
            padding=True,
            return_tensors="pt",
        )

        a_batch = self.tokenizer.pad(
            {"input_ids": input_ids_a, "attention_mask": attention_mask_a},
            padding=True,
            return_tensors="pt",
        )

        output = {
            "input_ids_q": q_batch["input_ids"],
            "attention_mask_q": q_batch["attention_mask"],
            "input_ids_a": a_batch["input_ids"],
            "attention_mask_a": a_batch["attention_mask"],
            "qa_id": torch.tensor(qa_ids, dtype=torch.long),
        }

        if "targets" in batch[0]:
            targets = [item["targets"] for item in batch]
            output["targets"] = torch.tensor(np.array(targets), dtype=torch.float32)

        return output


def load_data(load_cached_data=True):
    """
    Loads data from metadata CSVs or cached Parquet files.
    Implements caching logic as required.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    train_cache = os.path.join(Config.WORKING_DIR, "train_processed.parquet")
    val_cache = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
    test_cache = os.path.join(Config.WORKING_DIR, "test_processed.parquet")

    # Logic:
    # 1. IF load_cached_data is True: Try to load.
    # 2. IF fail OR load_cached_data is False: Process and Save.

    data_loaded = False
    train_df, val_df, test_df = None, None, None

    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            try:
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)
                test_df = pd.read_parquet(test_cache)
                data_loaded = True
                # print("Loaded data from cache.")
            except Exception:
                # print("Failed to load cache, reloading from source.")
                data_loaded = False

    if not data_loaded:
        # Load from metadata
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Preprocessing: Fill NaNs in text columns
        text_cols = ["question_title", "question_body", "answer"]
        for col in text_cols:
            train_df[col] = train_df[col].fillna("").astype(str)
            val_df[col] = val_df[col].fillna("").astype(str)
            test_df[col] = test_df[col].fillna("").astype(str)

        # Save to cache
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)
        # print("Processed and cached data.")

    return train_df, val_df, test_df


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        debug (bool): If True, subsets data for quick debugging.
        load_cached_data (bool): Whether to use cached dataframes.
    """
    seed_everything(Config.SEED)

    train_df, val_df, test_df = load_data(load_cached_data=load_cached_data)

    if debug:
        train_df = train_df.iloc[:100]
        val_df = val_df.iloc[:50]
        test_df = test_df.iloc[:50]

    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)

    train_dataset = QuestDataset(train_df, tokenizer, Config.MAX_LEN, mode="train")
    val_dataset = QuestDataset(val_df, tokenizer, Config.MAX_LEN, mode="val")
    test_dataset = QuestDataset(test_df, tokenizer, Config.MAX_LEN, mode="test")

    collate_fn = Collate(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
