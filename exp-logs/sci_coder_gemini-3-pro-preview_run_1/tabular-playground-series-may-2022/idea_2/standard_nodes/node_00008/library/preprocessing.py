import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from library.utils import seed_everything

# Configuration
CACHE_DIR = "./working/idea_3/"


def get_vocab(series):
    """
    Builds a vocabulary dictionary from a pandas Series of strings.
    Characters are mapped to integers starting from 1.
    """
    unique_chars = set()
    for s in series:
        unique_chars.update(list(s))
    sorted_chars = sorted(list(unique_chars))
    # 1-based index, 0 reserved for padding/unknown
    vocab = {c: i + 1 for i, c in enumerate(sorted_chars)}
    return vocab


def tokenize_series(series, vocab, max_len):
    """
    Tokenizes a Series of strings into a 2D numpy array of shape (N, max_len).
    """
    n = len(series)
    tokenized = np.zeros((n, max_len), dtype=np.int32)

    for i, s in enumerate(series):
        # Truncate if necessary
        chars = list(s)[:max_len]
        # Map chars to indices, default to 0 if not in vocab
        indices = [vocab.get(c, 0) for c in chars]
        tokenized[i, : len(indices)] = indices

    return tokenized


def process_data(load_cached_data=True, sample_size=None):
    """
    Main data processing function.

    Args:
        load_cached_data (bool): Whether to try loading data from cache.
        sample_size (int, optional): Number of rows to sample for debugging.
                                     If set, caching is disabled.

    Returns:
        Tuple of numpy arrays:
        (X_train_num, X_train_seq, y_train,
         X_val_num, X_val_seq, y_val,
         X_test_num, X_test_seq, ids_test)
    """
    seed_everything(42)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache file paths
    files = {
        "X_train_num": os.path.join(CACHE_DIR, "X_train_num.npy"),
        "X_train_seq": os.path.join(CACHE_DIR, "X_train_seq.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "X_val_num": os.path.join(CACHE_DIR, "X_val_num.npy"),
        "X_val_seq": os.path.join(CACHE_DIR, "X_val_seq.npy"),
        "y_val": os.path.join(CACHE_DIR, "y_val.npy"),
        "X_test_num": os.path.join(CACHE_DIR, "X_test_num.npy"),
        "X_test_seq": os.path.join(CACHE_DIR, "X_test_seq.npy"),
        "ids_test": os.path.join(CACHE_DIR, "ids_test.npy"),
    }

    # Check if we can load from cache
    # We only load from cache if sample_size is None (full data needed)
    use_cache = sample_size is None
    all_exist = all(os.path.exists(p) for p in files.values())

    if load_cached_data and use_cache and all_exist:
        print("Loading cached data from", CACHE_DIR)
        return (
            np.load(files["X_train_num"]),
            np.load(files["X_train_seq"]),
            np.load(files["y_train"]),
            np.load(files["X_val_num"]),
            np.load(files["X_val_seq"]),
            np.load(files["y_val"]),
            np.load(files["X_test_num"]),
            np.load(files["X_test_seq"]),
            np.load(files["ids_test"]),
        )

    print("Processing data from scratch...")

    # 1. Load Metadata
    train_path = "./metadata/train.csv"
    val_path = "./metadata/val.csv"
    test_path = "./metadata/test.csv"

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        raise FileNotFoundError("Metadata files not found in ./metadata/")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # 2. Apply Sampling (if debugging)
    if sample_size is not None:
        print(f"Debug Mode: Sampling {sample_size} rows per split.")
        train_df = train_df.iloc[:sample_size]
        val_df = val_df.iloc[:sample_size]
        test_df = test_df.iloc[:sample_size]

    # 3. Identify Columns
    seq_col = "f_27"
    ignore_cols = ["id", "target", "source_path", seq_col]
    # Identify numerical columns (all columns in train except ignored ones)
    num_cols = [c for c in train_df.columns if c not in ignore_cols]

    # 4. Extract Targets and IDs
    y_train = train_df["target"].values.astype(np.float32)
    y_val = val_df["target"].values.astype(np.float32)
    ids_test = test_df["id"].values.astype(np.int64)

    # 6. Prepare Numerical Data
    X_train_raw = train_df[num_cols].values.astype(np.float32)
    X_val_raw = val_df[num_cols].values.astype(np.float32)
    X_test_raw = test_df[num_cols].values.astype(np.float32)

    # 7. Standardization
    # Fit on Train, Transform all
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_raw)
    X_val_num = scaler.transform(X_val_raw)
    X_test_num = scaler.transform(X_test_raw)

    # 8. Sequence Tokenization
    # Build vocab from training data
    vocab = get_vocab(train_df[seq_col])
    # Determine max sequence length from training data
    max_len = train_df[seq_col].apply(len).max()

    X_train_seq = tokenize_series(train_df[seq_col], vocab, max_len)
    X_val_seq = tokenize_series(val_df[seq_col], vocab, max_len)
    X_test_seq = tokenize_series(test_df[seq_col], vocab, max_len)

    # 9. Save to Cache (only if not debugging)
    if use_cache:
        print("Saving processed data to cache...")
        np.save(files["X_train_num"], X_train_num)
        np.save(files["X_train_seq"], X_train_seq)
        np.save(files["y_train"], y_train)
        np.save(files["X_val_num"], X_val_num)
        np.save(files["X_val_seq"], X_val_seq)
        np.save(files["y_val"], y_val)
        np.save(files["X_test_num"], X_test_num)
        np.save(files["X_test_seq"], X_test_seq)
        np.save(files["ids_test"], ids_test)

    return (
        X_train_num,
        X_train_seq,
        y_train,
        X_val_num,
        X_val_seq,
        y_val,
        X_test_num,
        X_test_seq,
        ids_test,
    )
