import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything


def load_raw_data(
    data_dir=Config.METADATA_DIR,
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Loads the train, validation, and test datasets from the metadata directory.

    Args:
        data_dir (str): Directory containing the metadata CSV files.
        debug (bool): If True, samples a small subset of the data for debugging.
        debug_sample_size (int): Number of samples to load in debug mode.

    Returns:
        tuple: (df_train, df_val, df_test) as pandas DataFrames.
    """
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "validation.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # Verify files exist
    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Metadata file not found: {p}")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    if debug:
        print(f"DEBUG mode enabled. Sampling {debug_sample_size} rows per dataset.")
        seed_everything()

        # Sample safely, ensuring we don't request more samples than exist
        df_train = df_train.sample(
            n=min(len(df_train), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)
        df_test = df_test.sample(
            n=min(len(df_test), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)

    return df_train, df_val, df_test


def get_stratified_split(df, target_col="author", test_size=0.2, seed=Config.SEED):
    """
    Performs a stratified split on a DataFrame.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_col (str): The column to stratify by.
        test_size (float): Proportion of the dataset to include in the validation split.
        seed (int): Random seed.

    Returns:
        tuple: (train_df, val_df)
    """
    seed_everything(seed)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=seed,
        shuffle=True,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


class AuthorDataset(Dataset):
    """
    PyTorch Dataset for Author Identification using DeBERTa.
    """

    def __init__(
        self,
        texts,
        labels=None,
        tokenizer_name=Config.DEBERTA_MODEL,
        max_len=Config.MAX_LEN,
    ):
        """
        Args:
            texts (list or pd.Series): Input text sequences.
            labels (list or pd.Series, optional): Target labels.
            tokenizer_name (str): HuggingFace tokenizer model name.
            max_len (int): Maximum sequence length.
        """
        self.texts = list(texts)  # Ensure list for indexing
        self.labels = list(labels) if labels is not None else None
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.label2id = Config.LABEL2ID

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Flatten to remove batch dimension added by return_tensors='pt'
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        if self.labels is not None:
            label_str = self.labels[idx]
            label_id = self.label2id[label_str]
            item["labels"] = torch.tensor(label_id, dtype=torch.long)

        return item
