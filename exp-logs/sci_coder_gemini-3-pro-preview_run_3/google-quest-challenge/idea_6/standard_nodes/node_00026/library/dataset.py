import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import GroupKFold
from library.config import Config


def get_tokenizer(model_name):
    """
    Loads and returns the tokenizer for the specified model.

    Args:
        model_name (str): The HuggingFace model name (e.g., 'microsoft/deberta-v3-large').

    Returns:
        transformers.PreTrainedTokenizer: The loaded tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name)


def create_folds(load_cached_data=True):
    """
    Loads the training metadata and creates cross-validation folds using GroupKFold.
    Implements caching to store the fold assignments to disk.

    Args:
        load_cached_data (bool): If True, attempts to load pre-computed folds from disk.

    Returns:
        pd.DataFrame: The training dataframe with an added 'fold' column.
    """
    # Define cache path
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, "train_folds.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached folds from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing folds...")

    # 2. Compute folds from scratch
    print("Creating folds from metadata...")
    if not os.path.exists(Config.TRAIN_PATH):
        raise FileNotFoundError(f"Training metadata not found at {Config.TRAIN_PATH}")

    df = pd.read_csv(Config.TRAIN_PATH)

    # Ensure the group column exists
    if Config.GROUP_COL not in df.columns:
        raise ValueError(
            f"Group column '{Config.GROUP_COL}' not found in training data."
        )

    # Initialize folds
    df["fold"] = -1

    # Use GroupKFold to ensure all answers for a question stay in the same split
    gkf = GroupKFold(n_splits=Config.N_FOLDS)

    # Handle potential NaNs in grouping column by treating them as a distinct group
    groups = df[Config.GROUP_COL].fillna("UNKNOWN_GROUP").astype(str)

    for fold_id, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        df.loc[val_idx, "fold"] = fold_id

    # 3. Save to cache
    print(f"Saving folds to {cache_path}")
    df.to_parquet(cache_path, index=False)

    return df


class QuestDataset(Dataset):
    """
    PyTorch Dataset for Question-Answer pairs.
    Handles tokenization and generation of segment-specific masks.
    """

    def __init__(self, df, tokenizer, max_len=512, inference=False):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'question_title', 'question_body', 'answer'.
            tokenizer: HuggingFace tokenizer.
            max_len (int): Maximum sequence length.
            inference (bool): If True, does not return target labels.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.inference = inference

        # Pre-extract text columns to avoid pandas overhead in __getitem__
        # Concatenate Title and Body for the "Question" part
        self.titles = df["question_title"].fillna("").values.astype(str)
        self.bodies = df["question_body"].fillna("").values.astype(str)
        self.answers = df["answer"].fillna("").values.astype(str)

        if not self.inference:
            # Ensure target columns are present
            missing_cols = [c for c in Config.TARGET_COLS if c not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing target columns: {missing_cols}")
            self.targets = df[Config.TARGET_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct inputs
        question_text = self.titles[idx] + " " + self.bodies[idx]
        answer_text = self.answers[idx]

        # Tokenize
        # Format: [CLS] Question [SEP] Answer [SEP]
        inputs = self.tokenizer(
            question_text,
            answer_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Generate Segment Masks using sequence_ids()
        # sequence_ids returns:
        #   None for special tokens (CLS, SEP, PAD)
        #   0 for the first sequence (Question)
        #   1 for the second sequence (Answer)
        seq_ids = inputs.sequence_ids()

        # Create binary masks
        # We treat None as -1 so they don't trigger either mask
        q_mask = [1 if s == 0 else 0 for s in seq_ids]
        a_mask = [1 if s == 1 else 0 for s in seq_ids]

        # Convert to tensors
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "q_mask": torch.tensor(q_mask, dtype=torch.long),
            "a_mask": torch.tensor(a_mask, dtype=torch.long),
        }

        if not self.inference:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float)

        return item
