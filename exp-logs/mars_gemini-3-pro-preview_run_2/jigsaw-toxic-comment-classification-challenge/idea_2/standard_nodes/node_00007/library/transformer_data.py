import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


class ToxicityDataset(Dataset):
    """
    PyTorch Dataset for the Toxicity Classification task.
    Handles tokenization and input formatting for Transformer models.
    """

    def __init__(self, dataframe, tokenizer, max_len, is_test=False):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing text and labels (if training/val).
            tokenizer (transformers.PreTrainedTokenizer): Hugging Face tokenizer.
            max_len (int): Maximum sequence length for tokenization.
            is_test (bool): Flag to indicate if this is the test set (no labels).
        """
        self.data = dataframe
        self.text = dataframe[Config.TEXT_COL]
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        if not self.is_test:
            self.targets = dataframe[Config.LABEL_COLS].values

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        # Use iloc to ensure integer-based indexing regardless of DataFrame index
        text = str(self.text.iloc[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        out = {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
        }

        # Add token_type_ids if the tokenizer returns them (e.g. for BERT)
        if "token_type_ids" in inputs:
            out["token_type_ids"] = torch.tensor(
                inputs["token_type_ids"], dtype=torch.long
            )

        if not self.is_test:
            out["targets"] = torch.tensor(self.targets[index], dtype=torch.float)

        return out


def create_dataloaders(train_df, val_df, test_df, model_name):
    """
    Creates PyTorch DataLoaders for train, validation, and test sets.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.
        model_name (str): Name of the model to load the corresponding tokenizer.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    print(f"Initializing Tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Creating Datasets...")
    train_dataset = ToxicityDataset(train_df, tokenizer, Config.MAX_LEN)
    val_dataset = ToxicityDataset(val_df, tokenizer, Config.MAX_LEN)
    test_dataset = ToxicityDataset(test_df, tokenizer, Config.MAX_LEN, is_test=True)

    print("Creating DataLoaders...")
    train_params = {
        "batch_size": Config.TRAIN_BATCH_SIZE,
        "shuffle": True,
        "num_workers": Config.NUM_WORKERS,
        "pin_memory": True,
        "drop_last": True,  # Drop last incomplete batch for training stability
    }

    val_params = {
        "batch_size": Config.VALID_BATCH_SIZE,
        "shuffle": False,
        "num_workers": Config.NUM_WORKERS,
        "pin_memory": True,
    }

    test_params = {
        "batch_size": Config.VALID_BATCH_SIZE,
        "shuffle": False,
        "num_workers": Config.NUM_WORKERS,
        "pin_memory": True,
    }

    train_loader = DataLoader(train_dataset, **train_params)
    val_loader = DataLoader(val_dataset, **val_params)
    test_loader = DataLoader(test_dataset, **test_params)

    return train_loader, val_loader, test_loader
