import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything

# Prevent tokenizer parallelism issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def process_dataframe(df, tokenizer, cat_maps, max_len, is_test=False):
    """
    Process a dataframe into numpy arrays suitable for TensorDataset.
    """
    # 1. Text Preprocessing
    # Ensure strings
    titles = df["question_title"].fillna("").astype(str).tolist()
    bodies = df["question_body"].fillna("").astype(str).tolist()
    answers = df["answer"].fillna("").astype(str).tolist()

    # 2. Tokenization
    # Stream 1: Question (Title + Body)
    q_enc = tokenizer(
        titles,
        bodies,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_token_type_ids=True,
        return_tensors="np",
    )

    # Stream 2: Answer
    a_enc = tokenizer(
        answers,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_token_type_ids=True,
        return_tensors="np",
    )

    # 3. Generate Segment Masks for Question
    # token_type_ids: 0 for Title (and CLS), 1 for Body (and SEP)
    # attention_mask: 1 for tokens, 0 for padding
    q_input_ids = q_enc["input_ids"]
    q_token_type_ids = q_enc["token_type_ids"]
    q_attention_mask = q_enc["attention_mask"]

    # Title mask: Segment 0 and real token
    q_title_mask = (q_token_type_ids == 0) & (q_attention_mask == 1)
    # Body mask: Segment 1 and real token
    q_body_mask = (q_token_type_ids == 1) & (q_attention_mask == 1)

    # Answer arrays
    a_input_ids = a_enc["input_ids"]
    a_token_type_ids = a_enc["token_type_ids"]
    a_attention_mask = a_enc["attention_mask"]

    # 4. Categorical Features
    cat_indices = []
    for col in Config.CAT_COLS:
        mapping = cat_maps[col]
        # Map values, default to 0 if unknown (though we fit on all data)
        indices = df[col].astype(str).map(mapping).fillna(0).astype(np.int64).values
        cat_indices.append(indices)

    # Shape: (N, Num_Cats)
    cat_feats = np.stack(cat_indices, axis=1)

    # 5. Targets
    if not is_test:
        targets = df[Config.TARGET_COLS].values.astype(np.float32)
    else:
        # Dummy targets for test set
        targets = np.zeros((len(df), Config.NUM_TARGETS), dtype=np.float32)

    # 6. QA IDs
    qa_ids = df["qa_id"].values.astype(np.int64)

    return {
        "qa_ids": qa_ids,
        "q_input_ids": q_input_ids,
        "q_attention_mask": q_attention_mask,
        "q_token_type_ids": q_token_type_ids,
        "q_title_mask": q_title_mask.astype(np.int8),  # Save space
        "q_body_mask": q_body_mask.astype(np.int8),
        "a_input_ids": a_input_ids,
        "a_attention_mask": a_attention_mask,
        "a_token_type_ids": a_token_type_ids,
        "cat_feats": cat_feats,
        "targets": targets,
    }


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Main function to load data, process it (or load cache), and return DataLoaders.
    """
    seed_everything()

    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Cache file paths
    train_cache = os.path.join(cache_dir, "train_data.npz")
    val_cache = os.path.join(cache_dir, "val_data.npz")
    test_cache = os.path.join(cache_dir, "test_data.npz")
    meta_cache = os.path.join(cache_dir, "meta_data.npz")

    # If debug, we force re-processing on a subset and do not save to main cache
    if debug:
        load_cached_data = False
        print(f"DEBUG mode: Processing {Config.DEBUG_SAMPLE_SIZE} rows per split...")

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(meta_cache)
    )

    if load_cached_data and cache_exists:
        print("Loading cached data from .npz files...")
        train_data = np.load(train_cache)
        val_data = np.load(val_cache)
        test_data = np.load(test_cache)
        meta_data = np.load(meta_cache)
        cat_dims = meta_data["cat_dims"].tolist()
    else:
        print("Processing data from scratch...")
        # Load Raw Data
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        if debug:
            train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
            val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
            test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

        # Build Categorical Mappings (Fit on all data to handle all categories)
        cat_maps = {}
        cat_dims = []
        for col in Config.CAT_COLS:
            # Union of all unique values
            uniques = (
                set(train_df[col].astype(str).unique())
                | set(val_df[col].astype(str).unique())
                | set(test_df[col].astype(str).unique())
            )
            # Sort for determinism
            sorted_cats = sorted(list(uniques))
            mapping = {c: i for i, c in enumerate(sorted_cats)}
            cat_maps[col] = mapping
            cat_dims.append(len(sorted_cats))

        print(f"Categorical Dims: {cat_dims}")

        # Initialize Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

        # Process Splits
        print("Tokenizing Train...")
        train_data = process_dataframe(
            train_df, tokenizer, cat_maps, Config.MAX_LEN, is_test=False
        )
        print("Tokenizing Val...")
        val_data = process_dataframe(
            val_df, tokenizer, cat_maps, Config.MAX_LEN, is_test=False
        )
        print("Tokenizing Test...")
        test_data = process_dataframe(
            test_df, tokenizer, cat_maps, Config.MAX_LEN, is_test=True
        )

        # Save to Cache (only if not debug)
        if not debug:
            print("Saving data to cache...")
            np.savez_compressed(train_cache, **train_data)
            np.savez_compressed(val_cache, **val_data)
            np.savez_compressed(test_cache, **test_data)
            np.savez_compressed(meta_cache, cat_dims=np.array(cat_dims))

    # Helper to create TensorDataset
    def create_dataset(data_dict):
        # Convert numpy arrays to torch tensors
        # Order of tensors must match what the model expects/unpacks
        return TensorDataset(
            torch.tensor(data_dict["qa_ids"], dtype=torch.long),
            torch.tensor(data_dict["q_input_ids"], dtype=torch.long),
            torch.tensor(data_dict["q_attention_mask"], dtype=torch.long),
            torch.tensor(data_dict["q_token_type_ids"], dtype=torch.long),
            torch.tensor(
                data_dict["q_title_mask"], dtype=torch.float
            ),  # Float for pooling math
            torch.tensor(data_dict["q_body_mask"], dtype=torch.float),
            torch.tensor(data_dict["a_input_ids"], dtype=torch.long),
            torch.tensor(data_dict["a_attention_mask"], dtype=torch.long),
            torch.tensor(data_dict["a_token_type_ids"], dtype=torch.long),
            torch.tensor(data_dict["cat_feats"], dtype=torch.long),
            torch.tensor(data_dict["targets"], dtype=torch.float),
        )

    train_ds = create_dataset(train_data)
    val_ds = create_dataset(val_data)
    test_ds = create_dataset(test_data)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, cat_dims
