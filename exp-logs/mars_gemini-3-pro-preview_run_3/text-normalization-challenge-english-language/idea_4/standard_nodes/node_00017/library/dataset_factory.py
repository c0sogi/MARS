import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Tuple, Any

from library.config import Config
from library.text_utils import CharTokenizer, is_hard_token, get_context_window
from library.retrieval_system import SimilarityIndex


class RAGDataset(Dataset):
    """
    Dataset class for the Retrieval-Augmented Character Transformer.
    Constructs inputs of the form:
    [Context] <SEP> [Retrieved_Raw] <SEP> [Retrieved_Norm] <SEP> [Target_Raw]
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: CharTokenizer,
        max_len: int = 128,
        mode: str = "train",
    ):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

        # Validate required columns
        required_cols = ["before", "retrieved_source", "retrieved_target", "context"]
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"DataFrame missing required column: {col}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 1. Construct Input String
        # Format: context <sep> ret_src <sep> ret_tgt <sep> target
        context = str(row["context"])
        ret_src = str(row["retrieved_source"])
        ret_tgt = str(row["retrieved_target"])
        target_raw = str(row["before"])

        # Helper to get IDs without special tokens
        def get_ids(s):
            return self.tokenizer.encode(s, add_special_tokens=False)

        sep_id = self.tokenizer.sep_token_id

        # Construct sequence manually to insert <sep> correctly
        input_ids = (
            get_ids(context)
            + [sep_id]
            + get_ids(ret_src)
            + [sep_id]
            + get_ids(ret_tgt)
            + [sep_id]
            + get_ids(target_raw)
        )

        # Add SOS/EOS to the full sequence
        input_ids = (
            [self.tokenizer.sos_token_id] + input_ids + [self.tokenizer.eos_token_id]
        )

        # Truncate if necessary
        # We truncate from the left (context) if needed to preserve the target at the end
        if len(input_ids) > self.max_len:
            input_ids = input_ids[-self.max_len :]
            # Ensure it starts with SOS if we cut it off
            input_ids[0] = self.tokenizer.sos_token_id

        # 2. Construct Target (if training/val)
        target_ids = []
        if self.mode in ["train", "val"]:
            target_text = str(row["after"])
            target_ids = self.tokenizer.encode(target_text, add_special_tokens=True)
            # Truncate target
            if len(target_ids) > self.max_len:
                target_ids = target_ids[: self.max_len]
                target_ids[-1] = self.tokenizer.eos_token_id

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": (
                torch.tensor(target_ids, dtype=torch.long)
                if len(target_ids) > 0
                else torch.tensor([], dtype=torch.long)
            ),
            "raw_text": target_raw,
            "id": row.get("id", ""),
        }


class RAGCollator:
    """
    Collate function to pad sequences in a batch.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        target_ids = [item["target_ids"] for item in batch]
        raw_texts = [item["raw_text"] for item in batch]
        ids = [item["id"] for item in batch]

        # Pad Inputs
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )

        # Pad Targets
        if len(target_ids) > 0 and len(target_ids[0]) > 0:
            target_ids_padded = torch.nn.utils.rnn.pad_sequence(
                target_ids, batch_first=True, padding_value=self.pad_token_id
            )
        else:
            target_ids_padded = None

        # Create Attention Mask (1 for real tokens, 0 for pad)
        attention_mask = (input_ids_padded != self.pad_token_id).long()

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "target_ids": target_ids_padded,
            "raw_texts": raw_texts,
            "ids": ids,
        }


def process_data(
    df: pd.DataFrame, index: SimilarityIndex, mode: str, k: int = 1
) -> pd.DataFrame:
    """
    Enriches the dataframe with context and retrieval results.
    """
    print(f"Processing {len(df)} samples for mode '{mode}'...")

    # 1. Get Context
    # Using apply with the helper function
    # Note: get_context_window needs random access to the full DF,
    # so we iterate by index.
    contexts = []
    # Reset index to ensure alignment with get_context_window logic
    df_reset = df.reset_index(drop=True)

    for idx in range(len(df_reset)):
        contexts.append(get_context_window(df_reset, idx))
    df_reset["context"] = contexts

    # 2. Retrieval
    queries = df_reset["before"].astype(str).tolist()

    # For training, we retrieve k+1 to handle potential self-retrieval
    retrieve_k = k + 1 if mode == "train" else k

    print("Running batch retrieval...")
    retrieval_results = index.retrieve_batch(queries, k=retrieve_k)

    ret_sources = []
    ret_targets = []

    for i, res_list in enumerate(retrieval_results):
        if not res_list:
            ret_sources.append("")
            ret_targets.append("")
            continue

        selected = res_list[0]

        if mode == "train":
            # Check for self-retrieval (exact string match)
            # If the first result is the query itself, take the second one.
            query_text = queries[i]
            if selected["source"] == query_text and len(res_list) > 1:
                selected = res_list[1]
            elif selected["source"] == query_text:
                # If only 1 result and it's self, we keep it.
                # Ideally we want distinct examples, but for rare tokens this might happen.
                pass

        ret_sources.append(selected["source"])
        ret_targets.append(selected["target"])

    df_reset["retrieved_source"] = ret_sources
    df_reset["retrieved_target"] = ret_targets

    return df_reset


def create_dataloaders(
    load_cached_data: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, CharTokenizer]:
    """
    Main function to prepare data and create DataLoaders.

    Args:
        load_cached_data: Whether to use cached parquet files in working dir.

    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """
    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)

    # 1. Initialize Tokenizer
    tokenizer = CharTokenizer()
    if os.path.exists(Config.TOKENIZER_PATH) and load_cached_data:
        tokenizer.load(Config.TOKENIZER_PATH)
    else:
        print("Training tokenizer...")
        # Load raw train data to get vocab
        if not os.path.exists(Config.TRAIN_DATA_PATH):
            raise FileNotFoundError(f"Training data missing: {Config.TRAIN_DATA_PATH}")

        df_train_raw = pd.read_parquet(Config.TRAIN_DATA_PATH)
        texts = (
            df_train_raw["before"].astype(str).tolist()
            + df_train_raw["after"].astype(str).tolist()
        )
        tokenizer.train(texts)
        tokenizer.save(Config.TOKENIZER_PATH)

    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # 2. Initialize Retrieval Index
    sim_index = SimilarityIndex()
    sim_index.build_index(load_cached_data=load_cached_data)

    # 3. Define Cache Paths
    cache_train = os.path.join(Config.PROCESSED_DIR, "train_rag.parquet")
    cache_val = os.path.join(Config.PROCESSED_DIR, "val_rag.parquet")
    cache_test = os.path.join(Config.PROCESSED_DIR, "test_rag.parquet")

    # --- Helper to load or process ---
    def get_df(path_raw, path_cache, mode):
        if load_cached_data and os.path.exists(path_cache):
            print(f"Loading cached {mode} data from {path_cache}")
            return pd.read_parquet(path_cache)

        if not os.path.exists(path_raw):
            # If test file is missing (e.g. in some environments), return empty DF
            if mode == "test":
                print(f"Warning: Test file {path_raw} not found.")
                return pd.DataFrame(
                    columns=[
                        "before",
                        "context",
                        "retrieved_source",
                        "retrieved_target",
                        "id",
                    ]
                )
            raise FileNotFoundError(f"Data file not found: {path_raw}")

        print(f"Processing {mode} data from scratch...")
        df = pd.read_parquet(path_raw)

        # Subsample for debugging
        if Config.MAX_TRAIN_SAMPLES and mode == "train":
            df = df.head(Config.MAX_TRAIN_SAMPLES).copy()

        # Filter Logic
        if mode in ["train", "val"]:
            # Apply "Hard" token filter
            mask = df.apply(lambda x: is_hard_token(x["class"], x["before"]), axis=1)
            df = df[mask].copy()
            print(f"Filtered {mode} set to {len(df)} hard tokens.")
        elif mode == "test":
            # For test, we filter out purely alphabetic tokens to match the "Gate" logic
            # This ensures the neural model only predicts on what it's supposed to handle.
            mask = ~df["before"].astype(str).apply(str.isalpha)
            df = df[mask].copy()
            print(f"Filtered {mode} set to {len(df)} non-alpha tokens.")

        if len(df) == 0:
            print(f"Warning: {mode} dataframe is empty after filtering.")
            # Return empty with correct columns to avoid errors
            return pd.DataFrame(
                columns=list(df.columns)
                + ["context", "retrieved_source", "retrieved_target"]
            )

        # Process (Context + Retrieval)
        df_processed = process_data(df, sim_index, mode, k=Config.RETRIEVAL_K)

        # Save to cache
        df_processed.to_parquet(path_cache, index=False)
        return df_processed

    # 4. Load DataFrames
    df_train = get_df(Config.TRAIN_DATA_PATH, cache_train, "train")
    df_val = get_df(Config.VAL_DATA_PATH, cache_val, "val")
    df_test = get_df(Config.TEST_DATA_PATH, cache_test, "test")

    # 5. Create Datasets
    train_dataset = RAGDataset(df_train, tokenizer, Config.MAX_SEQ_LEN, "train")
    val_dataset = RAGDataset(df_val, tokenizer, Config.MAX_SEQ_LEN, "val")
    test_dataset = RAGDataset(df_test, tokenizer, Config.MAX_SEQ_LEN, "test")

    # 6. Create Collator
    collator = RAGCollator(tokenizer.pad_token_id)

    # 7. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, tokenizer
