import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.utils import load_data, set_seed


class NeuralDataset(Dataset):
    """
    PyTorch Dataset for the Heterogeneous Transformer.
    Input: Character-level sequence with context (Prev <SEP> Token <SEP> Next).
    Output: BPE-level sequence of normalized text.
    """

    def __init__(self, df, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

        # Pre-extract columns to lists for faster access in __getitem__
        self.before = self.df["before"].astype(str).tolist()
        self.after = self.df["after"].astype(str).tolist()
        self.prev_word = self.df["prev_word"].astype(str).tolist()
        self.next_word = self.df["next_word"].astype(str).tolist()

        self.sep_token = Config.SEP_TOKEN

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct source string: "prev <sep> token <sep> next"
        # The tokenizer will handle character-level encoding of this entire string
        src_text = f"{self.prev_word[idx]}{self.sep_token}{self.before[idx]}{self.sep_token}{self.next_word[idx]}"
        tgt_text = self.after[idx]

        # Encode
        src_ids = self.tokenizer.encode_source(src_text)
        tgt_ids = self.tokenizer.encode_target(tgt_text)

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(
            tgt_ids, dtype=torch.long
        )


def prepare_hfbb_data(load_cached_data=True):
    """
    Loads the full training data for the HFBB Engine.
    The HFBB Engine handles its own internal caching of N-gram maps,
    so this function primarily serves to load the raw dataframe.
    """
    print("DataFactory: Loading full training data for HFBB...")
    # We ignore load_cached_data here as load_data is fast and HFBB has its own cache logic
    df = load_data("train")
    return df


def _add_context_columns(df):
    """
    Adds prev_word and next_word columns to the dataframe.
    Assumes df is already sorted by sentence_id and token_id.
    """
    # Ensure sorting
    df = df.sort_values(["sentence_id", "token_id"])

    # Shift to get context
    # GroupBy is necessary to prevent context bleeding across sentences
    # Using fillna with PAD_TOKEN for boundary conditions
    df["prev_word"] = (
        df.groupby("sentence_id")["before"].shift(1).fillna(Config.PAD_TOKEN)
    )
    df["next_word"] = (
        df.groupby("sentence_id")["before"].shift(-1).fillna(Config.PAD_TOKEN)
    )
    return df


def _process_neural_data(split, load_cached_data):
    """
    Internal function to process data for the neural network:
    1. Load raw data
    2. Add context
    3. Filter for semiotic classes
    4. Upsample rare classes (if train)
    5. Cache result
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"processed_{split}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"DataFactory: Loading processed {split} data from cache...")
        return pd.read_parquet(cache_path)

    print(f"DataFactory: Processing {split} data from scratch...")

    # 1. Load Raw Data
    df = load_data(split)

    # 2. Add Context (Must be done BEFORE filtering to preserve context words)
    print(f"DataFactory: Generating context for {split}...")
    df = _add_context_columns(df)

    # 3. Filter for Semiotic Classes
    # We only train the neural net on the "hard" cases (Semiotic/Ambiguous)
    print(f"DataFactory: Filtering {split} for semiotic classes...")
    initial_len = len(df)
    df = df[df["class"].isin(Config.SEMIOTIC_CLASSES)].copy()
    print(f"  Filtered {split}: {initial_len} -> {len(df)} tokens")

    # 4. Upsample (Only for training set)
    if split == "train":
        print("DataFactory: Upsampling rare classes...")
        dfs_to_concat = [df]

        for cls in Config.UPSAMPLE_CLASSES:
            cls_df = df[df["class"] == cls]
            count = len(cls_df)

            if count > 0 and count < Config.UPSAMPLE_TARGET_COUNT:
                # Sample with replacement to fill the gap
                n_samples = Config.UPSAMPLE_TARGET_COUNT - count
                upsampled = cls_df.sample(
                    n=n_samples, replace=True, random_state=Config.SEED
                )
                dfs_to_concat.append(upsampled)
                print(f"  Upsampled {cls}: +{n_samples} samples")

        df = pd.concat(dfs_to_concat, ignore_index=True)
        # Shuffle after upsampling
        df = df.sample(frac=1.0, random_state=Config.SEED).reset_index(drop=True)
        print(f"  Final Training Size: {len(df)}")

    # 5. Save to Cache
    print(f"DataFactory: Saving {split} to cache...")
    Config.setup_directories()
    df.to_parquet(cache_path, index=False)

    return df


def get_dataloaders(tokenizer, batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Creates DataLoaders for the Neural Network.

    Args:
        tokenizer: Instance of HeterogeneousTokenizer.
        batch_size: Batch size for training/validation.
        load_cached_data: Whether to use cached processed dataframes.

    Returns:
        train_loader, val_loader
    """
    set_seed()

    # Process Data
    train_df = _process_neural_data("train", load_cached_data)
    val_df = _process_neural_data("val", load_cached_data)

    # Create Datasets
    train_dataset = NeuralDataset(train_df, tokenizer)
    val_dataset = NeuralDataset(val_df, tokenizer)

    # Collate Function for Padding
    pad_id = tokenizer.pad_id

    def collate_fn(batch):
        src_list, tgt_list = zip(*batch)

        # Pad sequences
        src_padded = pad_sequence(src_list, batch_first=True, padding_value=pad_id)
        tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=pad_id)

        return src_padded, tgt_padded

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader
