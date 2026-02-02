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
    Implements Cross-Encoder logic with Data Augmentation (Cite solution_lesson_node_00006).
    """

    def __init__(
        self,
        prompt_seqs,
        res_a_seqs,
        res_b_seqs,
        lengths,
        targets=None,
        config=None,
        is_train=False,
    ):
        self.prompt_seqs = prompt_seqs
        self.res_a_seqs = res_a_seqs
        self.res_b_seqs = res_b_seqs
        self.lengths = lengths
        self.targets = targets
        self.config = config
        self.is_train = is_train
        self.sep_token_idx = 2  # Hardcoded based on Tokenizer change

    def __len__(self):
        return len(self.prompt_seqs)

    def __getitem__(self, idx):
        # Retrieve raw padded sequences
        p = self.prompt_seqs[idx]
        a = self.res_a_seqs[idx]
        b = self.res_b_seqs[idx]
        l = self.lengths[idx]  # [len_p, len_a, len_b]

        t = None
        if self.targets is not None:
            t = self.targets[idx]

        # Data Augmentation: Random Swap (Cite solution_lesson_node_00006)
        # Swapping A and B helps the model learn symmetry and doubles effective data
        if self.is_train and np.random.random() > 0.5:
            a, b = b, a
            # Swap lengths: [len_p, len_a, len_b] -> [len_p, len_b, len_a]
            l = np.array([l[0], l[2], l[1]], dtype=np.float32)
            # Swap targets: [win_a, win_b, tie] -> [win_b, win_a, tie]
            if t is not None:
                t = np.array([t[1], t[0], t[2]], dtype=np.float32)

        # Trim padding (0) to recover actual sequences
        p_trim = p[p != 0]
        a_trim = a[a != 0]
        b_trim = b[b != 0]

        # Concatenate: P + SEP + A + SEP + B
        # We use a list for construction then convert to array
        sep = np.array([self.sep_token_idx], dtype=np.int32)
        combined = np.concatenate([p_trim, sep, a_trim, sep, b_trim])

        # Pad/Truncate to MAX_SEQ_LEN
        max_len = self.config.MAX_SEQ_LEN
        if len(combined) > max_len:
            combined = combined[:max_len]
        else:
            padding = np.zeros(max_len - len(combined), dtype=np.int32)
            combined = np.concatenate([combined, padding])

        item = {
            "input_ids": torch.tensor(combined, dtype=torch.long),
            "lengths": torch.tensor(l, dtype=torch.float32),
        }

        if t is not None:
            item["target"] = torch.tensor(t, dtype=torch.float32)

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

            # Pad to individual buckets to allow robust saving/loading
            # We pad to generous lengths here, then trim and concat in Dataset
            p_pad = pad_sequences(p_seq, config.PROMPT_MAX_LEN)
            ra_pad = pad_sequences(ra_seq, config.RESP_MAX_LEN)
            rb_pad = pad_sequences(rb_seq, config.RESP_MAX_LEN)

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
        config=config,
        is_train=True,
    )

    val_dataset = ChatbotDataset(
        val_data["prompt"],
        val_data["response_a"],
        val_data["response_b"],
        val_data["lengths"],
        val_data["targets"],
        config=config,
        is_train=False,
    )

    test_dataset = ChatbotDataset(
        test_data["prompt"],
        test_data["response_a"],
        test_data["response_b"],
        test_data["lengths"],
        targets=None,
        config=config,
        is_train=False,
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
