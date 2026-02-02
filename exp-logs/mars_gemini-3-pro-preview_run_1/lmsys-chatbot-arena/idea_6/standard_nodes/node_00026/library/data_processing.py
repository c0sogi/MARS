import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config

# ==========================================
# Feature Engineering
# ==========================================


def extract_scalar_features(df):
    """
    Computes relative scalar features between Response A and Response B.
    Explicitly excludes absolute values to prevent length bias overfitting.

    Features:
    1. Char Length Difference & Ratio
    2. Word Length Difference & Ratio
    3. Newline Count Difference & Ratio

    Args:
        df (pd.DataFrame): DataFrame containing 'response_a' and 'response_b'.

    Returns:
        np.ndarray: Array of shape (N, 6) containing the features.
    """
    # Fill NaNs with empty strings to ensure robust processing
    resp_a = df["response_a"].fillna("").astype(str)
    resp_b = df["response_b"].fillna("").astype(str)

    # 1. Character Counts
    len_a_char = resp_a.str.len()
    len_b_char = resp_b.str.len()

    # 2. Word Counts (simple whitespace split)
    len_a_word = resp_a.apply(lambda x: len(x.split()))
    len_b_word = resp_b.apply(lambda x: len(x.split()))

    # 3. Newline Counts
    count_a_newline = resp_a.apply(lambda x: x.count("\n"))
    count_b_newline = resp_b.apply(lambda x: x.count("\n"))

    # --- Compute Relative Features ---
    epsilon = 1e-6  # Avoid division by zero

    features = pd.DataFrame()

    # Differences (A - B)
    features["diff_char"] = len_a_char - len_b_char
    features["diff_word"] = len_a_word - len_b_word
    features["diff_newline"] = count_a_newline - count_b_newline

    # Ratios (A / B)
    features["ratio_char"] = len_a_char / (len_b_char + epsilon)
    features["ratio_word"] = len_a_word / (len_b_word + epsilon)
    features["ratio_newline"] = (count_a_newline + 1) / (count_b_newline + 1)

    return features.values.astype(np.float32)


# ==========================================
# Data Processing & Caching
# ==========================================


def process_and_cache_data(df, split_name, tokenizer, load_cached_data=True):
    """
    Tokenizes text and extracts scalar features.
    Implements caching using .npy files in the working directory.

    Args:
        df (pd.DataFrame): The dataframe to process.
        split_name (str): 'train', 'val', or 'test'.
        tokenizer: HuggingFace tokenizer.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing processed numpy arrays.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define filenames
    files = {
        "ids_a": os.path.join(cache_dir, f"{split_name}_ids_a.npy"),
        "mask_a": os.path.join(cache_dir, f"{split_name}_mask_a.npy"),
        "ids_b": os.path.join(cache_dir, f"{split_name}_ids_b.npy"),
        "mask_b": os.path.join(cache_dir, f"{split_name}_mask_b.npy"),
        "scalars": os.path.join(cache_dir, f"{split_name}_scalars.npy"),
        "labels": os.path.join(cache_dir, f"{split_name}_labels.npy"),
    }

    # Check if all files exist
    all_exist = all(os.path.exists(p) for p in files.values())

    if load_cached_data and all_exist:
        print(f"Loading cached data for {split_name}...")
        data = {k: np.load(v) for k, v in files.items() if os.path.exists(v)}
        # Handle test set which might not have labels
        if split_name == "test" and "labels" not in data:
            pass
        return data

    print(f"Processing data for {split_name}...")

    # --- Text Tokenization ---
    # We tokenize "Prompt + Response" for both A and B
    # DeBERTa tokenizer handles [CLS] prompt [SEP] response [SEP] automatically if passed as pairs,
    # but here we simply concatenate strings or rely on the tokenizer's pair handling.
    # Ideally: tokenizer(prompt, response) creates correct segment ids.

    prompts = df["prompt"].fillna("").astype(str).tolist()
    resps_a = df["response_a"].fillna("").astype(str).tolist()
    resps_b = df["response_b"].fillna("").astype(str).tolist()

    # Tokenize Branch A
    enc_a = tokenizer(
        prompts,
        resps_a,
        truncation=True,
        max_length=Config.MAX_LENGTH,
        padding=False,  # We pad dynamically in collate_fn to save memory
        return_attention_mask=True,
        return_token_type_ids=False,  # DeBERTa v3 doesn't use token_type_ids usually, or we ignore them
    )

    # Tokenize Branch B
    enc_b = tokenizer(
        prompts,
        resps_b,
        truncation=True,
        max_length=Config.MAX_LENGTH,
        padding=False,
        return_attention_mask=True,
        return_token_type_ids=False,
    )

    # Convert lists to object arrays of arrays (for variable length storage before batching)
    # or we can pad here. To save disk space, let's store as object arrays of numpy arrays
    # However, np.save with object arrays requires allow_pickle=True which is discouraged but sometimes necessary for jagged arrays.
    # Alternatively, we can just pad to max_length here for simplicity in caching,
    # but that wastes space.
    # Given the constraint "Do NOT use pickle", we must use fixed size arrays or save flattened with offsets.
    # For simplicity and speed given 220GB RAM, let's pad to MAX_LENGTH and save as standard numpy arrays.
    # This ensures .npy compatibility without pickle.

    def pad_sequences(sequences, max_len, pad_val):
        out = np.full((len(sequences), max_len), pad_val, dtype=np.int32)
        for i, seq in enumerate(sequences):
            l = min(len(seq), max_len)
            out[i, :l] = seq[:l]
        return out

    ids_a = pad_sequences(enc_a["input_ids"], Config.MAX_LENGTH, tokenizer.pad_token_id)
    mask_a = pad_sequences(enc_a["attention_mask"], Config.MAX_LENGTH, 0)
    ids_b = pad_sequences(enc_b["input_ids"], Config.MAX_LENGTH, tokenizer.pad_token_id)
    mask_b = pad_sequences(enc_b["attention_mask"], Config.MAX_LENGTH, 0)

    # --- Scalar Features ---
    scalars = extract_scalar_features(df)

    # --- Labels ---
    if split_name != "test":
        labels = df[Config.TARGET_COLS].values.astype(np.float32)
    else:
        # Dummy labels for test
        labels = np.zeros((len(df), 3), dtype=np.float32)

    # --- Save to Cache ---
    np.save(files["ids_a"], ids_a)
    np.save(files["mask_a"], mask_a)
    np.save(files["ids_b"], ids_b)
    np.save(files["mask_b"], mask_b)
    np.save(files["scalars"], scalars)
    np.save(files["labels"], labels)

    return {
        "ids_a": ids_a,
        "mask_a": mask_a,
        "ids_b": ids_b,
        "mask_b": mask_b,
        "scalars": scalars,
        "labels": labels,
    }


# ==========================================
# Dataset Class
# ==========================================


class ChatbotDataset(Dataset):
    def __init__(self, data_dict, scaler=None, is_test=False):
        """
        Args:
            data_dict (dict): Dictionary containing numpy arrays of features/labels.
            scaler (object): StandardScaler object fitted on training scalars.
            is_test (bool): Whether this is the test set.
        """
        self.ids_a = data_dict["ids_a"]
        self.mask_a = data_dict["mask_a"]
        self.ids_b = data_dict["ids_b"]
        self.mask_b = data_dict["mask_b"]
        self.scalars = data_dict["scalars"]
        self.labels = data_dict["labels"]
        self.is_test = is_test

        # Normalize scalars
        if scaler is not None:
            self.scalars = scaler.transform(self.scalars)

    def __len__(self):
        return len(self.ids_a)

    def __getitem__(self, idx):
        return {
            "input_ids_a": torch.tensor(self.ids_a[idx], dtype=torch.long),
            "attention_mask_a": torch.tensor(self.mask_a[idx], dtype=torch.long),
            "input_ids_b": torch.tensor(self.ids_b[idx], dtype=torch.long),
            "attention_mask_b": torch.tensor(self.mask_b[idx], dtype=torch.long),
            "scalar_features": torch.tensor(self.scalars[idx], dtype=torch.float),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


# ==========================================
# Collate Function
# ==========================================


class CollateFn:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        """
        Dynamic padding for the batch.
        The dataset returns fixed 512 length arrays, but we can trim them here
        to the max length in the specific batch to speed up attention.
        """
        # Stack everything first
        input_ids_a = torch.stack([item["input_ids_a"] for item in batch])
        attention_mask_a = torch.stack([item["attention_mask_a"] for item in batch])
        input_ids_b = torch.stack([item["input_ids_b"] for item in batch])
        attention_mask_b = torch.stack([item["attention_mask_b"] for item in batch])
        scalar_features = torch.stack([item["scalar_features"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])

        # Determine max length in this batch (ignoring padding)
        # We look for the last non-zero attention mask index
        # Combine masks to find global max length for the batch
        max_len_a = attention_mask_a.sum(dim=1).max().item()
        max_len_b = attention_mask_b.sum(dim=1).max().item()
        max_len = max(max_len_a, max_len_b)

        # Ensure at least some length
        max_len = max(max_len, 1)

        # Trim
        # Note: We assume padding is at the end
        input_ids_a = input_ids_a[:, :max_len]
        attention_mask_a = attention_mask_a[:, :max_len]
        input_ids_b = input_ids_b[:, :max_len]
        attention_mask_b = attention_mask_b[:, :max_len]

        return {
            "input_ids_a": input_ids_a,
            "attention_mask_a": attention_mask_a,
            "input_ids_b": input_ids_b,
            "attention_mask_b": attention_mask_b,
            "scalar_features": scalar_features,
            "labels": labels,
        }


# ==========================================
# Main Data Loading Interface
# ==========================================


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.

    Args:
        debug (bool): If True, subsamples the data for quick debugging.
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Tokenizer
    print(f"Loading tokenizer: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load Metadata
    print("Loading metadata CSVs...")
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    if debug:
        print(f"DEBUG MODE: Subsampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    # Process Data
    train_data = process_and_cache_data(df_train, "train", tokenizer, load_cached_data)
    val_data = process_and_cache_data(df_val, "val", tokenizer, load_cached_data)
    test_data = process_and_cache_data(df_test, "test", tokenizer, load_cached_data)

    # Fit Scaler on Training Scalars
    # We implement a simple StandardScaler manually to avoid sklearn dependency issues in Dataset
    # or just use sklearn's StandardScaler and pass it.
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(train_data["scalars"])

    # Create Datasets
    train_dataset = ChatbotDataset(train_data, scaler=scaler, is_test=False)
    val_dataset = ChatbotDataset(val_data, scaler=scaler, is_test=False)
    test_dataset = ChatbotDataset(test_data, scaler=scaler, is_test=True)

    # Collate Function
    collate_fn = CollateFn(pad_token_id=tokenizer.pad_token_id)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(
        f"DataLoaders Ready. Train: {len(train_loader)} batches, Val: {len(val_loader)} batches."
    )
    return train_loader, val_loader, test_loader
