import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from library.config import Config


class DataProcessor:
    """
    Handles data loading, feature engineering, preprocessing, and caching
    for the Granular Unified Transformer experiment.
    """

    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.vocab = None

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies feature engineering logic.
        Specifically calculates 'unique_characters' from the sequence column f_27.
        """
        # Ensure sequence column is treated as string
        df[self.config.sequence_col] = df[self.config.sequence_col].astype(str)

        # Calculate count of unique characters in the string
        # This captures the diversity of the sequence, which is a strong signal
        df["unique_characters"] = df[self.config.sequence_col].apply(
            lambda x: len(set(x))
        )

        return df

    def build_vocab(self, series: pd.Series) -> dict:
        """
        Builds a character-to-integer vocabulary from the training sequences.
        """
        # Get all unique characters present in the column
        unique_chars = sorted(list(set("".join(series.unique()))))

        # Create mapping: Char -> Int
        # Reserve 0 for padding/unknown, so start index at 1
        vocab = {char: idx + 1 for idx, char in enumerate(unique_chars)}
        return vocab

    def tokenize_sequence(
        self, series: pd.Series, vocab: dict, max_len: int
    ) -> np.ndarray:
        """
        Converts a Series of strings into a 2D numpy array of integer sequences.
        """

        def encode(s):
            # Map chars to ints, use 0 for unknown
            indices = [vocab.get(c, 0) for c in s]
            # Pad or truncate to fixed length
            if len(indices) < max_len:
                indices += [0] * (max_len - len(indices))
            else:
                indices = indices[:max_len]
            return indices

        # Apply encoding to all rows
        sequences = [encode(s) for s in series]
        return np.array(sequences, dtype=np.int32)

    def process_data(self, load_cached_data: bool = True) -> dict:
        """
        Main pipeline execution.
        1. Checks for cached .npy files.
        2. If not found, loads metadata CSVs.
        3. Applies feature engineering and preprocessing.
        4. Saves processed data to cache.
        5. Returns dictionary of numpy arrays.
        """
        # Define paths for cached files
        cache_files = {
            "X_num_train": os.path.join(self.config.working_dir, "X_num_train.npy"),
            "X_seq_train": os.path.join(self.config.working_dir, "X_seq_train.npy"),
            "y_train": os.path.join(self.config.working_dir, "y_train.npy"),
            "X_num_val": os.path.join(self.config.working_dir, "X_num_val.npy"),
            "X_seq_val": os.path.join(self.config.working_dir, "X_seq_val.npy"),
            "y_val": os.path.join(self.config.working_dir, "y_val.npy"),
            "X_num_test": os.path.join(self.config.working_dir, "X_num_test.npy"),
            "X_seq_test": os.path.join(self.config.working_dir, "X_seq_test.npy"),
            "ids_test": os.path.join(self.config.working_dir, "ids_test.npy"),
        }

        # Attempt to load from cache
        if load_cached_data:
            all_exist = all(os.path.exists(p) for p in cache_files.values())
            if all_exist:
                print("Loading cached data from working directory...")
                data = {k: np.load(v) for k, v in cache_files.items()}
                return data

        print("Cache not found or reload forced. Processing data from scratch...")

        # Load raw data from metadata
        df_train = pd.read_csv(self.config.train_path)
        df_val = pd.read_csv(self.config.val_path)
        df_test = pd.read_csv(self.config.test_path)

        # Apply Debug Sampling if enabled
        if self.config.debug:
            print(f"Debug mode: Sampling first {self.config.max_samples} rows.")
            df_train = df_train.iloc[: self.config.max_samples].copy()
            df_val = df_val.iloc[: self.config.max_samples].copy()
            df_test = df_test.iloc[: self.config.max_samples].copy()

        # 1. Feature Engineering
        print("Engineering features...")
        df_train = self.engineer_features(df_train)
        df_val = self.engineer_features(df_val)
        df_test = self.engineer_features(df_test)

        # 2. Numerical Preprocessing
        print("Standardizing numerical features...")
        num_cols = self.config.numerical_features

        # Fit scaler only on training data
        self.scaler.fit(df_train[num_cols])

        # Transform all sets
        X_num_train = self.scaler.transform(df_train[num_cols]).astype(np.float32)
        X_num_val = self.scaler.transform(df_val[num_cols]).astype(np.float32)
        X_num_test = self.scaler.transform(df_test[num_cols]).astype(np.float32)

        # 3. Sequence Preprocessing
        print("Tokenizing sequences...")
        # Build vocab from training data
        self.vocab = self.build_vocab(df_train[self.config.sequence_col])

        # Tokenize
        X_seq_train = self.tokenize_sequence(
            df_train[self.config.sequence_col], self.vocab, self.config.sequence_len
        )
        X_seq_val = self.tokenize_sequence(
            df_val[self.config.sequence_col], self.vocab, self.config.sequence_len
        )
        X_seq_test = self.tokenize_sequence(
            df_test[self.config.sequence_col], self.vocab, self.config.sequence_len
        )

        # 4. Extract Targets and IDs
        y_train = df_train[self.config.target_col].values.astype(np.float32)
        y_val = df_val[self.config.target_col].values.astype(np.float32)
        ids_test = df_test[self.config.id_col].values.astype(np.int64)

        # 5. Save to Cache
        print("Saving processed data to cache...")
        os.makedirs(self.config.working_dir, exist_ok=True)

        np.save(cache_files["X_num_train"], X_num_train)
        np.save(cache_files["X_seq_train"], X_seq_train)
        np.save(cache_files["y_train"], y_train)

        np.save(cache_files["X_num_val"], X_num_val)
        np.save(cache_files["X_seq_val"], X_seq_val)
        np.save(cache_files["y_val"], y_val)

        np.save(cache_files["X_num_test"], X_num_test)
        np.save(cache_files["X_seq_test"], X_seq_test)
        np.save(cache_files["ids_test"], ids_test)

        # Return dictionary
        data = {
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

        return data
