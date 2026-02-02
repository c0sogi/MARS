import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from library.config import Config
from library.utils import set_seed

# Prevent tokenizer parallelism issues in DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_data(load_cached_data=True):
    """
    Loads data from metadata CSVs or cached Parquet files.

    Args:
        load_cached_data (bool): If True, attempts to load from cached parquet files in WORKING_DIR.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache paths
    train_cache = os.path.join(Config.WORKING_DIR, "train_processed.parquet")
    val_cache = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
    test_cache = os.path.join(Config.WORKING_DIR, "test_processed.parquet")

    # Check if cache exists and loading is requested
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        print(f"Loading cached data from {Config.WORKING_DIR}...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
        return train_df, val_df, test_df

    print("Loading data from metadata...")
    # Load from metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Save to cache
    print(f"Saving processed data to {Config.WORKING_DIR}...")
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df


class QADataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512, is_test=False):
        """
        Args:
            df (pd.DataFrame): Dataframe containing the data.
            tokenizer: Transformers tokenizer.
            max_len (int): Maximum sequence length.
            is_test (bool): Whether this is the test set (no labels).
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        # Pre-extract columns to lists for faster access
        self.titles = df["question_title"].fillna("").astype(str).tolist()
        self.bodies = df["question_body"].fillna("").astype(str).tolist()
        self.answers = df["answer"].fillna("").astype(str).tolist()
        self.qa_ids = df["qa_id"].tolist()

        if not self.is_test:
            self.targets = df[Config.TARGET_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        title = self.titles[idx]
        body = self.bodies[idx]
        answer = self.answers[idx]

        # ---------------------------------------------------------
        # Question Branch: Title + Body
        # ---------------------------------------------------------
        # Tokenize as a pair: <s> Title </s> </s> Body </s>
        q_enc = self.tokenizer(
            title,
            body,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=False,  # RoBERTa doesn't use these
            return_tensors=None,  # Return lists
        )

        q_input_ids = q_enc["input_ids"]
        q_attention_mask = q_enc["attention_mask"]

        # Generate Partitioned Masks
        # RoBERTa sep_token_id is 2. Structure: [0, ...T..., 2, 2, ...B..., 2]
        # We find indices of '2' to determine boundaries.
        sep_indices = [
            i for i, t in enumerate(q_input_ids) if t == self.tokenizer.sep_token_id
        ]

        q_title_mask = [0] * len(q_input_ids)
        q_body_mask = [0] * len(q_input_ids)

        if len(sep_indices) >= 2:
            # First SEP is at sep_indices[0] (End of Title segment)
            # Second SEP is at sep_indices[1] (Start of Body segment is after this)
            # Last SEP is at sep_indices[-1] (End of Body segment)

            # Title range: [1, sep_indices[0])
            for i in range(1, sep_indices[0]):
                q_title_mask[i] = 1

            # Body range: [sep_indices[1] + 1, sep_indices[-1])
            # Note: If body is empty, sep_indices[1] might be the last token, so range is empty.
            start_body = sep_indices[1] + 1
            end_body = sep_indices[-1]
            for i in range(start_body, end_body):
                q_body_mask[i] = 1
        else:
            # Fallback for weird cases (should not happen with standard tokenizer behavior)
            pass

        # ---------------------------------------------------------
        # Answer Branch: Answer only
        # ---------------------------------------------------------
        a_enc = self.tokenizer(
            answer,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors=None,
        )

        a_input_ids = a_enc["input_ids"]
        a_attention_mask = a_enc["attention_mask"]

        item = {
            "q_input_ids": torch.tensor(q_input_ids, dtype=torch.long),
            "q_attention_mask": torch.tensor(q_attention_mask, dtype=torch.long),
            "q_title_mask": torch.tensor(q_title_mask, dtype=torch.float),
            "q_body_mask": torch.tensor(q_body_mask, dtype=torch.float),
            "a_input_ids": torch.tensor(a_input_ids, dtype=torch.long),
            "a_attention_mask": torch.tensor(a_attention_mask, dtype=torch.long),
            "qa_id": self.qa_ids[idx],
        }

        if not self.is_test:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float)

        return item


class Collate:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        # Extract fields
        q_input_ids = [item["q_input_ids"] for item in batch]
        q_attention_mask = [item["q_attention_mask"] for item in batch]
        q_title_mask = [item["q_title_mask"] for item in batch]
        q_body_mask = [item["q_body_mask"] for item in batch]

        a_input_ids = [item["a_input_ids"] for item in batch]
        a_attention_mask = [item["a_attention_mask"] for item in batch]

        qa_ids = [item["qa_id"] for item in batch]

        # Pad sequences
        # input_ids pad with pad_token_id
        q_input_ids = pad_sequence(
            q_input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        a_input_ids = pad_sequence(
            a_input_ids, batch_first=True, padding_value=self.pad_token_id
        )

        # masks pad with 0
        q_attention_mask = pad_sequence(
            q_attention_mask, batch_first=True, padding_value=0
        )
        q_title_mask = pad_sequence(q_title_mask, batch_first=True, padding_value=0)
        q_body_mask = pad_sequence(q_body_mask, batch_first=True, padding_value=0)
        a_attention_mask = pad_sequence(
            a_attention_mask, batch_first=True, padding_value=0
        )

        batch_out = {
            "q_input_ids": q_input_ids,
            "q_attention_mask": q_attention_mask,
            "q_title_mask": q_title_mask,
            "q_body_mask": q_body_mask,
            "a_input_ids": a_input_ids,
            "a_attention_mask": a_attention_mask,
            "qa_id": torch.tensor(qa_ids, dtype=torch.long),
        }

        if "labels" in batch[0]:
            labels = [item["labels"] for item in batch]
            batch_out["labels"] = torch.stack(labels)

        return batch_out


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached dataframes.
        debug (bool): If True, subsamples data for quick debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Data
    train_df, val_df, test_df = load_data(load_cached_data=load_cached_data)

    if debug:
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        test_df = test_df.head(50)
        print("DEBUG MODE: Data subsampled.")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Create Datasets
    train_dataset = QADataset(
        train_df, tokenizer, max_len=Config.MAX_LEN, is_test=False
    )
    val_dataset = QADataset(val_df, tokenizer, max_len=Config.MAX_LEN, is_test=False)
    test_dataset = QADataset(test_df, tokenizer, max_len=Config.MAX_LEN, is_test=True)

    # Create Collate Function
    collate_fn = Collate(pad_token_id=tokenizer.pad_token_id)

    # Create DataLoaders
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
