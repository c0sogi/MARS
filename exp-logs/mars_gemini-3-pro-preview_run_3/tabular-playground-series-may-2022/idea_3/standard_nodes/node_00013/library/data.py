import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from library.config import Config
from library.utils import seed_everything


def preprocess_features(load_cached_data=True):
    """
    Loads raw data, performs feature engineering, scaling, and encoding.
    Handles caching to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        tuple: (train_data, val_data, test_data, metadata)
               Each *_data is a dictionary containing 'cont', 'cat', 'target' (if avail).
               metadata contains 'cat_cardinalities' and 'cont_dim'.
    """
    seed_everything(Config.SEED)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    train_cache = os.path.join(Config.WORKING_DIR, "train_processed.parquet")
    val_cache = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
    test_cache = os.path.join(Config.WORKING_DIR, "test_processed.parquet")
    meta_cache = os.path.join(Config.WORKING_DIR, "metadata.npy")

    # Check if cache exists
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(meta_cache)
    ):
        print("Loading processed data from cache...")
        df_train = pd.read_parquet(train_cache)
        df_val = pd.read_parquet(val_cache)
        df_test = pd.read_parquet(test_cache)
        metadata = np.load(meta_cache, allow_pickle=True).item()

    else:
        print("Processing data from scratch...")
        # Load raw data based on metadata paths
        df_train = pd.read_csv(Config.TRAIN_PATH)
        df_val = pd.read_csv(Config.VAL_PATH)
        df_test = pd.read_csv(Config.TEST_PATH)

        # ---------------------------------------------------------
        # Feature Engineering
        # ---------------------------------------------------------
        def engineer_features(df):
            # 1. Decompose f_27 into characters
            # Assuming fixed length 10 based on Config.F27_SEQ_LEN
            chars = np.array([list(s) for s in df[Config.STRING_COL].values])
            for i in range(Config.F27_SEQ_LEN):
                df[f"{Config.STRING_COL}_{i}"] = chars[:, i]

            # 2. Unique character count
            df[Config.COUNT_COL] = (
                df[Config.STRING_COL].apply(lambda x: len(set(x))).astype(np.float32)
            )
            return df

        print("Engineering features...")
        df_train = engineer_features(df_train)
        df_val = engineer_features(df_val)
        df_test = engineer_features(df_test)

        # ---------------------------------------------------------
        # Column Definitions
        # ---------------------------------------------------------
        # Continuous: Original continuous cols + new count col
        cont_cols = Config.CONTINUOUS_COLS + [Config.COUNT_COL]

        # Categorical: Decomposed f_27 chars + discrete cols
        f27_cols = [f"{Config.STRING_COL}_{i}" for i in range(Config.F27_SEQ_LEN)]
        cat_cols = f27_cols + Config.DISCRETE_COLS

        # ---------------------------------------------------------
        # Normalization (Continuous)
        # ---------------------------------------------------------
        print("Normalizing continuous features...")
        scaler = StandardScaler()
        df_train[cont_cols] = scaler.fit_transform(
            df_train[cont_cols].astype(np.float32)
        )
        df_val[cont_cols] = scaler.transform(df_val[cont_cols].astype(np.float32))
        df_test[cont_cols] = scaler.transform(df_test[cont_cols].astype(np.float32))

        # ---------------------------------------------------------
        # Encoding (Categorical)
        # ---------------------------------------------------------
        print("Encoding categorical features...")
        # We use OrdinalEncoder. Handle unknown by marking as -1, then shifting everything +1.
        # This reserves index 0 for unknown/padding.
        encoder = OrdinalEncoder(
            dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1
        )

        # Fit on Train
        df_train[cat_cols] = encoder.fit_transform(df_train[cat_cols].astype(str))

        # Transform Val and Test
        df_val[cat_cols] = encoder.transform(df_val[cat_cols].astype(str))
        df_test[cat_cols] = encoder.transform(df_test[cat_cols].astype(str))

        # Shift indices by +1 so 0 is "unknown" and valid classes start at 1
        df_train[cat_cols] = df_train[cat_cols] + 1
        df_val[cat_cols] = df_val[cat_cols] + 1
        df_test[cat_cols] = df_test[cat_cols] + 1

        # Calculate cardinalities (max index + 1) for embedding layers
        # We look at the max value across all datasets to be safe, or just train
        # Since we shifted, max index is roughly num_unique.
        # We need a list of cardinalities corresponding to cat_cols order.
        cat_cardinalities = []
        for col in cat_cols:
            max_val = max(df_train[col].max(), df_val[col].max(), df_test[col].max())
            cat_cardinalities.append(int(max_val) + 1)

        metadata = {
            "cat_cols": cat_cols,
            "cont_cols": cont_cols,
            "cat_cardinalities": cat_cardinalities,
            "cont_dim": len(cont_cols),
        }

        # ---------------------------------------------------------
        # Caching
        # ---------------------------------------------------------
        print("Saving to cache...")
        df_train.to_parquet(train_cache, index=False)
        df_val.to_parquet(val_cache, index=False)
        df_test.to_parquet(test_cache, index=False)
        np.save(meta_cache, metadata)

    # Helper to extract numpy arrays
    def extract_data(df, meta):
        cont = df[meta["cont_cols"]].values.astype(np.float32)
        cat = df[meta["cat_cols"]].values.astype(np.int64)
        target = (
            df["target"].values.astype(np.float32) if "target" in df.columns else None
        )
        ids = df["id"].values if "id" in df.columns else None
        return {"cont": cont, "cat": cat, "target": target, "id": ids}

    train_data = extract_data(df_train, metadata)
    val_data = extract_data(df_val, metadata)
    test_data = extract_data(df_test, metadata)

    return train_data, val_data, test_data, metadata


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for Manufacturing Control Data.
    Supports 'supervised' mode for classification and 'pretrain' mode for DAE training.
    Implements Swap Noise Augmentation for the pretraining phase.
    """

    def __init__(self, cont_data, cat_data, targets=None, mode="supervised"):
        """
        Args:
            cont_data (np.ndarray): Normalized continuous features.
            cat_data (np.ndarray): Encoded categorical features.
            targets (np.ndarray, optional): Binary targets.
            mode (str): 'supervised' or 'pretrain'.
        """
        self.cont_data = torch.from_numpy(cont_data).float()
        self.cat_data = torch.from_numpy(cat_data).long()
        self.targets = (
            torch.from_numpy(targets).float() if targets is not None else None
        )
        self.mode = mode
        self.swap_prob = Config.SWAP_NOISE_PROBA

    def __len__(self):
        return len(self.cont_data)

    def __getitem__(self, idx):
        # Get clean sample
        cont = self.cont_data[idx]
        cat = self.cat_data[idx]

        if self.mode == "pretrain":
            # -----------------------------------------------------
            # Swap Noise Augmentation
            # -----------------------------------------------------
            # For each feature, with prob SWAP_NOISE_PROBA, replace value
            # with a value from a random row in the dataset.

            # 1. Continuous Swap
            # Create mask: 1 if we should swap, 0 otherwise
            mask_cont = torch.rand(cont.shape) < self.swap_prob

            if mask_cont.any():
                # We need to pick random rows for the masked features
                # Efficiently: Generate random indices for the number of swaps needed
                n_swaps = mask_cont.sum()
                rand_row_indices = torch.randint(0, len(self), (n_swaps,))

                # Get the column indices that need swapping
                swap_col_indices = torch.where(mask_cont)[0]

                # Extract noise values: self.cont_data[rows, cols]
                noise_vals = self.cont_data[rand_row_indices, swap_col_indices]

                # Apply noise
                cont_noisy = cont.clone()
                cont_noisy[mask_cont] = noise_vals
            else:
                cont_noisy = cont.clone()

            # 2. Categorical Swap
            mask_cat = torch.rand(cat.shape) < self.swap_prob

            if mask_cat.any():
                n_swaps = mask_cat.sum()
                rand_row_indices = torch.randint(0, len(self), (n_swaps,))
                swap_col_indices = torch.where(mask_cat)[0]

                noise_vals = self.cat_data[rand_row_indices, swap_col_indices]

                cat_noisy = cat.clone()
                cat_noisy[mask_cat] = noise_vals
            else:
                cat_noisy = cat.clone()

            # Return (Input, Target) where Target is the clean reconstruction
            return (cont_noisy, cat_noisy), (cont, cat)

        else:
            # Supervised Mode
            if self.targets is not None:
                return (cont, cat), self.targets[idx]
            else:
                return (cont, cat)
