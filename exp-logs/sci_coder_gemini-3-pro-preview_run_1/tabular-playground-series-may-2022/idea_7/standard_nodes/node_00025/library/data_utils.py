import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from library.config import Config


def add_engineered_features(df):
    """
    Adds engineered features to the dataframe.
    Specifically calculates the number of unique characters in 'f_27'.
    """
    # Ensure f_27 is string
    df[Config.SEQUENCE_COL] = df[Config.SEQUENCE_COL].astype(str)

    # Calculate unique characters count
    # We use a simple lambda to count unique chars in the string
    df["unique_characters"] = df[Config.SEQUENCE_COL].apply(lambda x: len(set(x)))

    return df


def tokenize_sequences(series, vocab=None):
    """
    Tokenizes the sequence column (f_27) into integer arrays.

    Args:
        series: Pandas Series containing the strings.
        vocab: Dictionary mapping char -> int. If None, it is built from the series.

    Returns:
        tokenized_data: np.ndarray of shape (N, SEQ_LEN)
        vocab: The vocabulary dictionary used.
    """
    # Ensure strings
    strings = series.astype(str).values

    # Build vocab if not provided
    if vocab is None:
        unique_chars = set()
        for s in strings:
            unique_chars.update(s)
        # Sort for determinism
        sorted_chars = sorted(list(unique_chars))
        # Start index at 1 (0 reserved for padding/masking if needed)
        vocab = {c: i for i, c in enumerate(sorted_chars, start=1)}

    # Tokenize
    seq_len = Config.SEQ_LEN
    num_samples = len(strings)
    tokenized_data = np.zeros((num_samples, seq_len), dtype=np.int32)

    for idx, s in enumerate(strings):
        # Truncate if longer than SEQ_LEN, though data analysis says fixed length
        chars = list(s)[:seq_len]
        for char_idx, c in enumerate(chars):
            if c in vocab:
                tokenized_data[idx, char_idx] = vocab[c]
            else:
                # Handle unknown chars if any (though vocab is built from train)
                # We can map to 0 or a specific UNK token. Here we leave as 0.
                pass

    return tokenized_data, vocab


def preprocess_pipeline(load_cached_data=True):
    """
    Main data processing pipeline.
    Loads data, performs feature engineering, scaling, and tokenization.
    Implements caching using .npy files.

    Args:
        load_cached_data: If True, attempts to load processed data from disk.

    Returns:
        data_dict: Dictionary containing processed numpy arrays.
        vocab_size: Integer representing the size of the vocabulary.
    """
    # Define cache file paths
    cache_dir = Config.CACHE_DIR
    files = {
        "X_num_train": os.path.join(cache_dir, "X_num_train.npy"),
        "X_seq_train": os.path.join(cache_dir, "X_seq_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "X_num_val": os.path.join(cache_dir, "X_num_val.npy"),
        "X_seq_val": os.path.join(cache_dir, "X_seq_val.npy"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "X_num_test": os.path.join(cache_dir, "X_num_test.npy"),
        "X_seq_test": os.path.join(cache_dir, "X_seq_test.npy"),
        "ids_test": os.path.join(cache_dir, "ids_test.npy"),
        "vocab_chars": os.path.join(
            cache_dir, "vocab_chars.npy"
        ),  # To reconstruct vocab
    }

    # 1. Try to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(f) for f in files.values())
        if all_exist:
            print("Loading data from cache...")
            data_dict = {}
            for key, path in files.items():
                if key != "vocab_chars":
                    data_dict[key] = np.load(path)

            # Reconstruct vocab size
            vocab_chars = np.load(files["vocab_chars"])
            # Vocab size is max index + 1 (indices start at 1)
            vocab_size = len(vocab_chars) + 1

            return data_dict, vocab_size
        else:
            print("Cache missing or incomplete. Reprocessing data...")
    else:
        print("Force reprocessing data...")

    # 2. Load Raw Data
    print("Loading metadata CSVs...")
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # 3. Feature Engineering
    print("Adding engineered features...")
    df_train = add_engineered_features(df_train)
    df_val = add_engineered_features(df_val)
    df_test = add_engineered_features(df_test)

    # 4. Numerical Preprocessing
    # Identify numerical columns dynamically
    # Exclude ID, Target, Source Path, and the Sequence column itself
    ignore = set(
        Config.IGNORE_COLS
        + [Config.ID_COL, Config.TARGET_COL, "source_path", Config.SEQUENCE_COL]
    )

    # Get all numeric columns
    numeric_candidates = df_train.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_candidates if c not in ignore]

    print(f"Selected {len(feature_cols)} numerical features: {feature_cols}")

    # Extract arrays
    X_num_train_raw = df_train[feature_cols].values
    X_num_val_raw = df_val[feature_cols].values
    X_num_test_raw = df_test[feature_cols].values

    # Scale
    print("Scaling numerical features...")
    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(X_num_train_raw).astype(np.float32)
    X_num_val = scaler.transform(X_num_val_raw).astype(np.float32)
    X_num_test = scaler.transform(X_num_test_raw).astype(np.float32)

    # 5. Sequence Preprocessing
    print("Tokenizing sequences...")
    # Build vocab from train only
    _, vocab = tokenize_sequences(df_train[Config.SEQUENCE_COL], vocab=None)

    # Tokenize all sets
    X_seq_train, _ = tokenize_sequences(df_train[Config.SEQUENCE_COL], vocab=vocab)
    X_seq_val, _ = tokenize_sequences(df_val[Config.SEQUENCE_COL], vocab=vocab)
    X_seq_test, _ = tokenize_sequences(df_test[Config.SEQUENCE_COL], vocab=vocab)

    # 6. Targets and IDs
    y_train = df_train[Config.TARGET_COL].values.astype(np.float32)
    y_val = df_val[Config.TARGET_COL].values.astype(np.float32)
    ids_test = df_test[Config.ID_COL].values.astype(np.int64)

    # 7. Save to Cache
    print(f"Saving processed data to {cache_dir}...")
    os.makedirs(cache_dir, exist_ok=True)

    np.save(files["X_num_train"], X_num_train)
    np.save(files["X_seq_train"], X_seq_train)
    np.save(files["y_train"], y_train)

    np.save(files["X_num_val"], X_num_val)
    np.save(files["X_seq_val"], X_seq_val)
    np.save(files["y_val"], y_val)

    np.save(files["X_num_test"], X_num_test)
    np.save(files["X_seq_test"], X_seq_test)
    np.save(files["ids_test"], ids_test)

    # Save vocab chars to reconstruct vocab/size later
    # Vocab is dict {char: int}. We save the chars in order of their int values (1..N)
    sorted_chars = sorted(vocab.keys(), key=lambda k: vocab[k])
    np.save(files["vocab_chars"], np.array(sorted_chars))

    # 8. Return
    data_dict = {
        "X_num_train": X_num_train,
        "X_seq_train": X_seq_train,
        "y_train": y_train,
        "X_num_val": X_num_val,
        "X_seq_val": X_seq_val,
        "y_val": y_val,
        "X_num_test": X_num_test,
        "X_seq_test": X_seq_test,
        "ids_test": ids_test,
    }

    vocab_size = len(vocab) + 1
    print("Data preprocessing complete.")

    return data_dict, vocab_size
