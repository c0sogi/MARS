import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Optional, Tuple, Dict, List, Union
from library.config import Config
from library.utils import seed_everything


class ChatbotDataset(Dataset):
    """
    Dataset class for Siamese DeBERTa-v3-Large preference model.
    Handles tokenization, scalar feature extraction, and symmetric augmentation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        mode: str = "train",
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the data.
            tokenizer (PreTrainedTokenizerBase): HuggingFace tokenizer.
            max_length (int): Maximum sequence length for tokenization.
            mode (str): 'train', 'val', or 'test'. Controls augmentation and target handling.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

        # Pre-extract columns to avoid overhead in __getitem__
        self.ids = self.df["id"].values
        self.prompts = self.df["prompt"].fillna("").values.astype(str)
        self.responses_a = self.df["response_a"].fillna("").values.astype(str)
        self.responses_b = self.df["response_b"].fillna("").values.astype(str)

        if self.mode != "test":
            self.targets = self.df[
                ["winner_model_a", "winner_model_b", "winner_tie"]
            ].values.astype(np.float32)
        else:
            self.targets = None

    def __len__(self) -> int:
        # Double the dataset size for training via symmetric augmentation
        if self.mode == "train":
            return 2 * len(self.df)
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        original_len = len(self.df)

        # Determine if this is a swapped sample (only for training)
        is_swapped = (self.mode == "train") and (idx >= original_len)
        real_idx = idx % original_len

        prompt = self.prompts[real_idx]
        resp_a = self.responses_a[real_idx]
        resp_b = self.responses_b[real_idx]

        # Apply swap if necessary
        if is_swapped:
            # Swap responses
            txt_a = resp_b
            txt_b = resp_a

            # Swap targets: [Win_A, Win_B, Tie] -> [Win_B, Win_A, Tie]
            if self.targets is not None:
                orig_target = self.targets[real_idx]
                target = np.array(
                    [orig_target[1], orig_target[0], orig_target[2]], dtype=np.float32
                )
        else:
            txt_a = resp_a
            txt_b = resp_b
            if self.targets is not None:
                target = self.targets[real_idx]

        # Tokenize Branch A: [CLS] Prompt [SEP] Response A [SEP]
        # Truncation strategy: "only_second" ensures Prompt is preserved, Response is truncated
        enc_a = self.tokenizer(
            prompt,
            txt_a,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # Dynamic padding in collate_fn
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        # Tokenize Branch B: [CLS] Prompt [SEP] Response B [SEP]
        enc_b = self.tokenizer(
            prompt,
            txt_b,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        # Extract Scalar Features (Log-transformed lengths of tokenized sequences)
        # We use sequence_ids to distinguish prompt (0) from response (1)
        # sequence_ids format: [None, 0, 0, ..., None, 1, 1, ..., None]

        def get_lengths(encoding):
            seq_ids = encoding.sequence_ids()
            # Filter out None (special tokens)
            p_len = sum(1 for x in seq_ids if x == 0)
            r_len = sum(1 for x in seq_ids if x == 1)
            return p_len, r_len

        p_len_a, r_len_a = get_lengths(enc_a)
        p_len_b, r_len_b = get_lengths(enc_b)

        # Note: p_len_a should be roughly equal to p_len_b, but truncation might affect it slightly
        # if max_length is extremely short (unlikely with 512). We average or take one.
        p_len = max(p_len_a, p_len_b)

        # Log-transform: log(length + 1)
        scalars = np.array(
            [np.log1p(p_len), np.log1p(r_len_a), np.log1p(r_len_b)], dtype=np.float32
        )

        # Construct output dictionary
        item = {
            "input_ids_a": enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "input_ids_b": enc_b["input_ids"],
            "attention_mask_b": enc_b["attention_mask"],
            "scalars": scalars,
            "id": self.ids[real_idx],
        }

        if self.targets is not None:
            item["labels"] = target

        return item


class CollateFn:
    """
    Custom collate function to handle dynamic padding for Siamese inputs.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict]):
        # Extract lists
        input_ids_a = [item["input_ids_a"] for item in batch]
        attention_mask_a = [item["attention_mask_a"] for item in batch]
        input_ids_b = [item["input_ids_b"] for item in batch]
        attention_mask_b = [item["attention_mask_b"] for item in batch]
        scalars = [item["scalars"] for item in batch]
        ids = [item["id"] for item in batch]

        # Dynamic Padding
        def pad_seqs(seqs, pad_val):
            max_len = max(len(s) for s in seqs)
            padded = []
            for s in seqs:
                pad_len = max_len - len(s)
                padded.append(s + [pad_val] * pad_len)
            return torch.tensor(padded, dtype=torch.long)

        batch_input_ids_a = pad_seqs(input_ids_a, self.pad_token_id)
        batch_mask_a = pad_seqs(attention_mask_a, 0)
        batch_input_ids_b = pad_seqs(input_ids_b, self.pad_token_id)
        batch_mask_b = pad_seqs(attention_mask_b, 0)

        batch_scalars = torch.tensor(np.array(scalars), dtype=torch.float32)

        out = {
            "input_ids_a": batch_input_ids_a,
            "attention_mask_a": batch_mask_a,
            "input_ids_b": batch_input_ids_b,
            "attention_mask_b": batch_mask_b,
            "scalars": batch_scalars,
            "ids": ids,
        }

        if "labels" in batch[0]:
            labels = [item["labels"] for item in batch]
            out["labels"] = torch.tensor(np.array(labels), dtype=torch.float32)

        return out


def load_data(
    load_cached_data: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads train, validation, and test data.
    Implements caching using Parquet format to speed up subsequent runs.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_data.parquet")
    val_cache = os.path.join(cache_dir, "val_data.parquet")
    test_cache = os.path.join(cache_dir, "test_data.parquet")

    # Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print("Loading data from cache...")
            try:
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)
                test_df = pd.read_parquet(test_cache)
                return train_df, val_df, test_df
            except Exception as e:
                print(f"Failed to load cache: {e}. Reloading from source.")

    # Load from source metadata
    print("Loading data from metadata CSVs...")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Save to cache
    print("Saving data to cache...")
    try:
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return train_df, val_df, test_df


def get_dataloaders(
    debug: bool = False,
    batch_size: int = Config.TRAIN_BATCH_SIZE,
    val_batch_size: int = Config.VALID_BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader, PreTrainedTokenizerBase]:
    """
    Prepares DataLoaders for training, validation, and testing.

    Args:
        debug (bool): If True, subsets data for quick debugging.
        batch_size (int): Batch size for training.
        val_batch_size (int): Batch size for validation/testing.
        num_workers (int): Number of worker processes.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, PreTrainedTokenizerBase]:
        (train_loader, val_loader, test_loader, tokenizer)
    """
    seed_everything(Config.SEED)

    # Load Data
    train_df, val_df, test_df = load_data(load_cached_data=Config.LOAD_CACHED_DATA)

    if debug:
        print(f"Debug Mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Create Datasets
    train_dataset = ChatbotDataset(
        train_df, tokenizer, max_length=Config.MAX_LENGTH, mode="train"
    )

    val_dataset = ChatbotDataset(
        val_df, tokenizer, max_length=Config.MAX_LENGTH, mode="val"
    )

    test_dataset = ChatbotDataset(
        test_df, tokenizer, max_length=Config.MAX_LENGTH, mode="test"
    )

    # Collate Function
    collate_fn = CollateFn(tokenizer)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, tokenizer
