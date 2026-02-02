import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.text_utils import Tokenizer, pad_sequences


class ChatbotDataset(Dataset):
    """
    PyTorch Dataset for the Chatbot Arena task.
    Wraps pre-processed numpy arrays of token indices and targets.
    """

    def __init__(self, prompt_seqs, res_a_seqs, res_b_seqs, lengths, targets=None):
        """
        Args:
            prompt_seqs (np.ndarray): Padded sequences for prompts.
            res_a_seqs (np.ndarray): Padded sequences for response A.
            res_b_seqs (np.ndarray): Padded sequences for response B.
            lengths (np.ndarray): Log-transformed lengths (N, 3).
            targets (np.ndarray, optional): Target probabilities (Winner A, Winner B, Tie).
        """
        self.prompt_seqs = prompt_seqs
        self.res_a_seqs = res_a_seqs
        self.res_b_seqs = res_b_seqs
        self.lengths = lengths
        self.targets = targets

    def __len__(self):
        return len(self.prompt_seqs)

    def __getitem__(self, idx):
        # Convert numpy rows to LongTensor for embedding layers
        item = {
            "prompt": torch.tensor(self.prompt_seqs[idx], dtype=torch.long),
            "response_a": torch.tensor(self.res_a_seqs[idx], dtype=torch.long),
            "response_b": torch.tensor(self.res_b_seqs[idx], dtype=torch.long),
            "lengths": torch.tensor(self.lengths[idx], dtype=torch.float32),
        }

        if self.targets is not None:
            # Targets are float probabilities
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


def prepare_datasets(config: Config, load_cached_data: bool = True):
    """
    Loads data, handles tokenization/padding (with caching), and returns Datasets.

    Args:
        config (Config): Configuration object.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, tokenizer)
    """
    # Cache paths
    cache_dir = config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_data.npz")
    val_cache = os.path.join(cache_dir, "val_data.npz")
    test_cache = os.path.join(cache_dir, "test_data.npz")
    tokenizer_path = config.VOCAB_PATH

    # Initialize variables to hold data
    train_data = {}
    val_data = {}
    test_data = {}
    tokenizer = Tokenizer(config)

    # Check if we can load from cache
    can_load = (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(tokenizer_path)
    )

    if can_load:
        print("Loading cached data...")
        try:
            # Load Tokenizer
            if not tokenizer.load(tokenizer_path):
                raise FileNotFoundError("Tokenizer load failed")

            # Load Numpy Arrays
            # We convert to dict to ensure data is read into memory
            with np.load(train_cache) as t_np:
                train_data = {k: t_np[k] for k in t_np}

            with np.load(val_cache) as v_np:
                val_data = {k: v_np[k] for k in v_np}

            with np.load(test_cache) as te_np:
                test_data = {k: te_np[k] for k in te_np}

        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing data...")
            can_load = False

    if not can_load:
        print("Processing data from scratch...")

        # Load Raw Data
        train_df = pd.read_csv(config.TRAIN_DATA_PATH)
        val_df = pd.read_csv(config.VAL_DATA_PATH)
        test_df = pd.read_csv(config.TEST_DATA_PATH)

        # Fill NaNs
        text_cols = ["prompt", "response_a", "response_b"]
        for df in [train_df, val_df, test_df]:
            for col in text_cols:
                df[col] = df[col].fillna("").astype(str)

        # Fit Tokenizer on Training Data
        # We combine all text columns from train to build vocab
        all_train_text = pd.concat([train_df[c] for c in text_cols]).tolist()
        tokenizer.fit_on_texts(all_train_text)
        tokenizer.save(tokenizer_path)

        # Helper to process a dataframe
        def process_df(df, is_test=False):
            # Calculate Length Features (Log-transformed word counts)
            # We use simple split to estimate word count
            len_p = (
                df["prompt"]
                .astype(str)
                .apply(lambda x: np.log1p(len(x.split())))
                .values
            )
            len_a = (
                df["response_a"]
                .astype(str)
                .apply(lambda x: np.log1p(len(x.split())))
                .values
            )
            len_b = (
                df["response_b"]
                .astype(str)
                .apply(lambda x: np.log1p(len(x.split())))
                .values
            )

            # Stack lengths: (N, 3)
            lengths = np.vstack([len_p, len_a, len_b]).T.astype(np.float32)

            # Tokenize
            p_seq = tokenizer.texts_to_sequences(df["prompt"].tolist())
            ra_seq = tokenizer.texts_to_sequences(df["response_a"].tolist())
            rb_seq = tokenizer.texts_to_sequences(df["response_b"].tolist())

            # Pad
            p_pad = pad_sequences(p_seq, config.MAX_SEQ_LEN)
            ra_pad = pad_sequences(ra_seq, config.MAX_SEQ_LEN)
            rb_pad = pad_sequences(rb_seq, config.MAX_SEQ_LEN)

            data_dict = {
                "prompt": p_pad,
                "response_a": ra_pad,
                "response_b": rb_pad,
                "lengths": lengths,
            }

            if not is_test:
                # Extract targets
                target_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
                targets = df[target_cols].values.astype(np.float32)
                data_dict["targets"] = targets

            return data_dict

        # Process splits
        train_data = process_df(train_df, is_test=False)
        val_data = process_df(val_df, is_test=False)
        test_data = process_df(test_df, is_test=True)

        # Save to cache
        np.savez(train_cache, **train_data)
        np.savez(val_cache, **val_data)
        np.savez(test_cache, **test_data)

    # Create Dataset Objects
    train_dataset = ChatbotDataset(
        train_data["prompt"],
        train_data["response_a"],
        train_data["response_b"],
        train_data["lengths"],
        train_data["targets"],
    )

    val_dataset = ChatbotDataset(
        val_data["prompt"],
        val_data["response_a"],
        val_data["response_b"],
        val_data["lengths"],
        val_data["targets"],
    )

    test_dataset = ChatbotDataset(
        test_data["prompt"],
        test_data["response_a"],
        test_data["response_b"],
        test_data["lengths"],
        targets=None,
    )

    return train_dataset, val_dataset, test_dataset, tokenizer


def get_dataloaders(config: Config, load_cached_data: bool = True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        config (Config): Configuration object.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (train_loader, val_loader, test_loader, tokenizer)
    """
    train_ds, val_ds, test_ds, tokenizer = prepare_datasets(config, load_cached_data)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, test_loader, tokenizer
