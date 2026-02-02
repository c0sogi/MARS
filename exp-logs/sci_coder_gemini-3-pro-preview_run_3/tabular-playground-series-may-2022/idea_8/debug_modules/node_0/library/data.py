import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    CACHE_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    STRING_COL,
    DISCRETE_COLS,
    TARGET_COL,
    ID_COL,
)


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control task.
    Separates data into categorical and continuous tensors for the Dual-Stream model.
    """

    def __init__(self, df, cat_cols, cont_cols, is_test=False):
        self.ids = df[ID_COL].values
        # Categorical features -> LongTensor for Embeddings
        self.cat_data = torch.tensor(df[cat_cols].values, dtype=torch.long)
        # Continuous features -> FloatTensor for Dense Layers
        self.cont_data = torch.tensor(df[cont_cols].values, dtype=torch.float32)
        self.is_test = is_test

        if not is_test:
            # Target -> FloatTensor (for BCEWithLogitsLoss)
            self.targets = torch.tensor(
                df[TARGET_COL].values, dtype=torch.float32
            ).unsqueeze(1)
        else:
            self.targets = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item = {
            "id": self.ids[idx],
            "cat": self.cat_data[idx],
            "cont": self.cont_data[idx],
        }
        if not self.is_test:
            item["target"] = self.targets[idx]
        return item


class DataProcessor:
    """
    Handles data loading, feature engineering, preprocessing, and caching.
    Implements Transductive Vocabulary Alignment.
    """

    def __init__(self):
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # Cache file paths
        self.train_cache = os.path.join(self.cache_dir, "train_processed.parquet")
        self.val_cache = os.path.join(self.cache_dir, "val_processed.parquet")
        self.test_cache = os.path.join(self.cache_dir, "test_processed.parquet")
        self.vocab_cache = os.path.join(self.cache_dir, "vocab_sizes.npy")

    def _feature_engineering(self, df):
        """
        Applies feature engineering: f_27 decomposition and unique char count.
        """
        # Decompose f_27 (string) into 10 character columns
        # Vectorized split: convert series of strings to list of lists
        chars = np.array([list(s) for s in df[STRING_COL].values])
        char_cols = [f"char_{i}" for i in range(10)]
        df_chars = pd.DataFrame(chars, columns=char_cols, index=df.index)

        # Calculate unique character count
        unique_counts = df[STRING_COL].apply(lambda x: len(set(x)))

        # Add new features to dataframe
        df = pd.concat([df, df_chars], axis=1)
        df["unique_character_count"] = unique_counts

        return df, char_cols

    def process_data(self, load_cached_data=True):
        """
        Loads raw data, processes it, and caches the result.
        Or loads from cache if available.
        """
        # 1. Try to load from cache
        if (
            load_cached_data
            and os.path.exists(self.train_cache)
            and os.path.exists(self.val_cache)
            and os.path.exists(self.test_cache)
            and os.path.exists(self.vocab_cache)
        ):
            print("Loading cached processed data...")
            train_df = pd.read_parquet(self.train_cache)
            val_df = pd.read_parquet(self.val_cache)
            test_df = pd.read_parquet(self.test_cache)
            vocab_sizes = np.load(self.vocab_cache, allow_pickle=True).item()
            return train_df, val_df, test_df, vocab_sizes

        print("Processing data from scratch...")

        # 2. Load Raw Data
        train_df = pd.read_csv(TRAIN_PATH)
        val_df = pd.read_csv(VAL_PATH)
        test_df = pd.read_csv(TEST_PATH)

        # 3. Feature Engineering
        train_df, char_cols = self._feature_engineering(train_df)
        val_df, _ = self._feature_engineering(val_df)
        test_df, _ = self._feature_engineering(test_df)

        # Define Feature Groups
        # Categorical: decomposed chars + discrete columns (f_29, f_30)
        cat_cols = char_cols + DISCRETE_COLS

        # Continuous: f_00..f_28 (excluding f_27) + unique_character_count
        # Filter columns starting with 'f_' that are not string or discrete
        all_cols = train_df.columns
        cont_cols = [
            c
            for c in all_cols
            if c.startswith("f_") and c not in [STRING_COL] + DISCRETE_COLS
        ]
        cont_cols.append("unique_character_count")

        # 4. Transductive Vocabulary Alignment (Ordinal Encoding)
        # Fit on Train + Val + Test to handle all tokens
        encoder = OrdinalEncoder(
            dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1
        )

        all_cat_data = pd.concat(
            [train_df[cat_cols], val_df[cat_cols], test_df[cat_cols]], axis=0
        )
        encoder.fit(all_cat_data)

        train_df[cat_cols] = encoder.transform(train_df[cat_cols])
        val_df[cat_cols] = encoder.transform(val_df[cat_cols])
        test_df[cat_cols] = encoder.transform(test_df[cat_cols])

        # Compute Vocab Sizes for Embeddings
        vocab_sizes = {col: int(all_cat_data[col].nunique()) for col in cat_cols}

        # 5. Scaling Continuous Features
        # Fit only on Train to prevent leakage
        scaler = StandardScaler()
        scaler.fit(train_df[cont_cols])

        train_df[cont_cols] = scaler.transform(train_df[cont_cols])
        val_df[cont_cols] = scaler.transform(val_df[cont_cols])
        test_df[cont_cols] = scaler.transform(test_df[cont_cols])

        # 6. Save to Cache
        print(f"Saving processed data to {self.cache_dir}...")
        train_df.to_parquet(self.train_cache, index=False)
        val_df.to_parquet(self.val_cache, index=False)
        test_df.to_parquet(self.test_cache, index=False)
        np.save(self.vocab_cache, vocab_sizes)

        return train_df, val_df, test_df, vocab_sizes

    def get_dataloaders(self, load_cached_data=True, debug=False, max_samples=None):
        """
        Main entry point to get PyTorch DataLoaders.
        """
        train_df, val_df, test_df, vocab_sizes = self.process_data(load_cached_data)

        # Debugging: Subset data
        if debug and max_samples is not None:
            print(f"Debug mode: limiting to {max_samples} samples.")
            train_df = train_df.iloc[:max_samples]
            val_df = val_df.iloc[:max_samples]
            test_df = test_df.iloc[:max_samples]

        # Identify columns for Dataset creation
        # Re-derive column lists based on vocab_sizes keys and remaining columns
        cat_cols = list(vocab_sizes.keys())

        # Exclude metadata and categorical columns to find continuous ones
        exclude_cols = set(cat_cols + [ID_COL, TARGET_COL, STRING_COL, "source_path"])
        cont_cols = [c for c in train_df.columns if c not in exclude_cols]
        cont_cols = sorted(cont_cols)  # Ensure deterministic order

        # Create Datasets
        train_ds = ManufacturingDataset(train_df, cat_cols, cont_cols)
        val_ds = ManufacturingDataset(val_df, cat_cols, cont_cols)
        test_ds = ManufacturingDataset(test_df, cat_cols, cont_cols, is_test=True)

        # Create DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader, vocab_sizes
