import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from library.config import GlobalConfig


class StackExchangeDataset(Dataset):
    """
    PyTorch Dataset for StackExchange Question-Answer pairs.
    Handles text tokenization and generation of segment masks for segment-aware pooling.
    """

    def __init__(self, data_path, tokenizer, max_length=512, is_test=False):
        """
        Args:
            data_path (str): Path to the metadata CSV file (train, val, or test).
            tokenizer (PreTrainedTokenizer): HuggingFace tokenizer.
            max_length (int): Maximum sequence length for tokenization.
            is_test (bool): If True, does not look for target columns.
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        # Load data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")

        self.df = pd.read_csv(data_path)
        self.target_cols = GlobalConfig.TARGET_COLS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Extract and clean text components
        # Handle potential NaNs by converting to empty string
        q_title = str(row["question_title"]) if pd.notna(row["question_title"]) else ""
        q_body = str(row["question_body"]) if pd.notna(row["question_body"]) else ""
        answer = str(row["answer"]) if pd.notna(row["answer"]) else ""

        # Construct text pair
        # Sequence 1: Question (Title + Body)
        # Sequence 2: Answer
        question_text = (q_title + " " + q_body).strip()
        answer_text = answer.strip()

        # Tokenize
        # We use return_tensors=None to get lists, facilitating sequence_ids access
        encoding = self.tokenizer(
            question_text,
            answer_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors=None,
        )

        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long)

        # Generate Segment Mask using sequence_ids()
        # sequence_ids() returns a list where:
        #   None -> Special tokens ([CLS], [SEP], <s>, </s>) or Padding
        #   0    -> First sequence (Question)
        #   1    -> Second sequence (Answer)
        #
        # We map this to a tensor:
        #   0 -> Special/Pad
        #   1 -> Question
        #   2 -> Answer

        raw_seq_ids = encoding.sequence_ids()
        segment_ids = []

        for sid in raw_seq_ids:
            if sid is None:
                segment_ids.append(0)
            elif sid == 0:
                segment_ids.append(1)
            elif sid == 1:
                segment_ids.append(2)

        segment_mask = torch.tensor(segment_ids, dtype=torch.long)

        # Construct return item
        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "segment_mask": segment_mask,
            "qa_id": row["qa_id"],
        }

        # Add targets if available and not in test mode
        if not self.is_test:
            # Ensure we have all target columns
            targets = row[self.target_cols].values.astype(np.float32)
            item["targets"] = torch.tensor(targets)

        return item


def get_dataloader(
    data_path,
    tokenizer,
    batch_size,
    max_length=512,
    is_test=False,
    shuffle=False,
    num_workers=GlobalConfig.NUM_WORKERS,
):
    """
    Factory function to create a DataLoader for the StackExchangeDataset.
    """
    dataset = StackExchangeDataset(
        data_path=data_path, tokenizer=tokenizer, max_length=max_length, is_test=is_test
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
    )

    return dataloader
