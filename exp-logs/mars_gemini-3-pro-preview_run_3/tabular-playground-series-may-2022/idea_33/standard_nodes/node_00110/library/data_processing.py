import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from library.utils import seed_everything


def get_unique_char_count(series):
    """
    Computes the number of unique characters in each string of the series.
    """
    return series.apply(lambda x: len(set(x)))


def decompose_f27(df):
    """
    Decomposes the 'f_27' column into 10 separate character columns p_0 to p_9.
    """
    # Vectorized string slicing is generally faster and cleaner in pandas
    for i in range(10):
        df[f"p_{i}"] = df["f_27"].str[i]
    return df


def preprocess_pipeline(train_df, val_df, test_df):
    """
    Performs feature engineering, transductive label encoding, and scaling.
    Returns processed NumPy arrays and vocab sizes.
    """
    # 1. Feature Engineering
    # We apply the same transformations to all dataframes
    for df in [train_df, val_df, test_df]:
        df["unique_char_count"] = get_unique_char_count(df["f_27"])
        # decompose_f27 modifies df in-place or we can reassign.
        # The function defined above modifies in-place but let's be safe.
        decompose_f27(df)

    # 2. Define Column Groups
    # Categorical: f_29, f_30, and the decomposed characters p_0...p_9
    cat_cols = ["f_29", "f_30"] + [f"p_{i}" for i in range(10)]

    # Continuous: f_00...f_28 (excluding f_27) + unique_char_count
    # Note: f_29 and f_30 are categorical. f_27 is the string source.
    # We select f_00 to f_28, skipping f_27.
    cont_cols = [f"f_{i:02d}" for i in range(29) if i != 27] + ["unique_char_count"]

    # 3. Transductive Label Encoding
    # Fit on Train + Val + Test to ensure the vocabulary covers all seen categories
    # We convert to string to handle any potential mixed types safely
    all_cat = pd.concat(
        [train_df[cat_cols], val_df[cat_cols], test_df[cat_cols]], axis=0
    ).astype(str)

    encoder = OrdinalEncoder(dtype=np.int64)
    encoder.fit(all_cat)

    X_cat_train = encoder.transform(train_df[cat_cols].astype(str))
    X_cat_val = encoder.transform(val_df[cat_cols].astype(str))
    X_cat_test = encoder.transform(test_df[cat_cols].astype(str))

    # Calculate vocabulary sizes for embedding layers
    vocab_sizes = [int(all_cat[col].nunique()) for col in cat_cols]

    # 4. Continuous Normalization
    # Fit StandardScaler ONLY on the training set to prevent data leakage
    scaler = StandardScaler()
    scaler.fit(train_df[cont_cols])

    X_cont_train = scaler.transform(train_df[cont_cols]).astype(np.float32)
    X_cont_val = scaler.transform(val_df[cont_cols]).astype(np.float32)
    X_cont_test = scaler.transform(test_df[cont_cols]).astype(np.float32)

    # 5. Extract Targets and IDs
    y_train = train_df["target"].values.astype(np.float32)
    y_val = val_df["target"].values.astype(np.float32)
    test_ids = test_df["id"].values.astype(np.int64)

    return (
        X_cat_train,
        X_cont_train,
        y_train,
        X_cat_val,
        X_cont_val,
        y_val,
        X_cat_test,
        X_cont_test,
        test_ids,
        vocab_sizes,
    )


def process_data(
    load_cached_data=True, base_dir="./metadata", cache_dir="./working/idea_33"
):
    """
    Orchestrates data loading, processing, and caching.
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "processed_data.npz")
    vocab_file = os.path.join(cache_dir, "vocab_sizes.npy")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_file) and os.path.exists(vocab_file):
        print(f"Loading cached data from {cache_dir}")
        try:
            data = np.load(cache_file)
            vocab_sizes = np.load(vocab_file)
            return (
                data["X_cat_train"],
                data["X_cont_train"],
                data["y_train"],
                data["X_cat_val"],
                data["X_cont_val"],
                data["y_val"],
                data["X_cat_test"],
                data["X_cont_test"],
                data["test_ids"],
                vocab_sizes.tolist(),
            )
        except Exception as e:
            print(f"Cache load failed ({e}). Reprocessing...")

    print("Processing data from scratch...")

    # 2. Load Metadata CSVs
    train_path = os.path.join(base_dir, "train.csv")
    val_path = os.path.join(base_dir, "val.csv")
    test_path = os.path.join(base_dir, "test.csv")

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(f"One or more metadata files not found in {base_dir}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # 3. Process Pipeline
    (
        X_cat_train,
        X_cont_train,
        y_train,
        X_cat_val,
        X_cont_val,
        y_val,
        X_cat_test,
        X_cont_test,
        test_ids,
        vocab_sizes,
    ) = preprocess_pipeline(train_df, val_df, test_df)

    # 4. Save to Cache
    np.savez(
        cache_file,
        X_cat_train=X_cat_train,
        X_cont_train=X_cont_train,
        y_train=y_train,
        X_cat_val=X_cat_val,
        X_cont_val=X_cont_val,
        y_val=y_val,
        X_cat_test=X_cat_test,
        X_cont_test=X_cont_test,
        test_ids=test_ids,
    )
    np.save(vocab_file, np.array(vocab_sizes))

    return (
        X_cat_train,
        X_cont_train,
        y_train,
        X_cat_val,
        X_cont_val,
        y_val,
        X_cat_test,
        X_cont_test,
        test_ids,
        vocab_sizes,
    )
