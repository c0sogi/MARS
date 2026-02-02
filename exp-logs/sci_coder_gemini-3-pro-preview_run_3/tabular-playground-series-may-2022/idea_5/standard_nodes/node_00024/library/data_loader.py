import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from library.config import Config


class ManufacturingDataset(Dataset):
    def __init__(self, x_cont, x_cat, y=None):
        self.x_cont = torch.FloatTensor(x_cont)
        self.x_cat = torch.LongTensor(x_cat)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.x_cont)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x_cont[idx], self.x_cat[idx], self.y[idx]
        return self.x_cont[idx], self.x_cat[idx]


def process_f27(df):
    """
    Decomposes the 'f_27' string column into 10 character columns
    and computes the unique character count.
    """
    f27_series = df["f_27"].astype(str)

    # Decompose string into list of characters
    # Since all strings are length 10, this creates a list of lists
    chars_list = f27_series.apply(list).tolist()
    char_cols = [f"f_27_{i}" for i in range(10)]

    # Create DataFrame for characters
    df_features = pd.DataFrame(chars_list, columns=char_cols)

    # Compute unique character count
    df_features["unique_char_count"] = f27_series.apply(lambda x: len(set(x))).values

    return df_features


class GlobalPreprocessor:
    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

        self.cache_files = {
            "train_cont": os.path.join(self.working_dir, "train_cont.npy"),
            "train_cat": os.path.join(self.working_dir, "train_cat.npy"),
            "train_y": os.path.join(self.working_dir, "train_y.npy"),
            "val_cont": os.path.join(self.working_dir, "val_cont.npy"),
            "val_cat": os.path.join(self.working_dir, "val_cat.npy"),
            "val_y": os.path.join(self.working_dir, "val_y.npy"),
            "test_cont": os.path.join(self.working_dir, "test_cont.npy"),
            "test_cat": os.path.join(self.working_dir, "test_cat.npy"),
            "test_ids": os.path.join(self.working_dir, "test_ids.npy"),
            "meta": os.path.join(self.working_dir, "meta.npy"),
        }

    def process_data(self, load_cached_data=True):
        # 1. Check Cache
        if load_cached_data and all(
            os.path.exists(p) for p in self.cache_files.values()
        ):
            data = {k: np.load(v) for k, v in self.cache_files.items() if k != "meta"}
            meta = np.load(self.cache_files["meta"], allow_pickle=True).item()
            return data, meta

        # 2. Load Metadata CSVs
        df_train = pd.read_csv(Config.TRAIN_PATH)
        df_val = pd.read_csv(Config.VAL_PATH)
        df_test = pd.read_csv(Config.TEST_PATH)

        # Extract Targets and IDs
        train_y = df_train["target"].values
        val_y = df_val["target"].values
        test_ids = df_test["id"].values

        # Mark splits for later separation
        df_train["split"] = "train"
        df_val["split"] = "val"
        df_test["split"] = "test"

        # Concatenate for global processing
        # Drop non-feature columns (target, id, source_path)
        cols_to_drop = ["target", "id", "source_path"]

        full_df = pd.concat(
            [
                df_train.drop(
                    columns=[c for c in cols_to_drop if c in df_train.columns]
                ),
                df_val.drop(columns=[c for c in cols_to_drop if c in df_val.columns]),
                df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns]),
            ],
            axis=0,
            ignore_index=True,
        )

        # 3. Feature Engineering (f_27)
        f27_features = process_f27(full_df)

        # Replace f_27 with engineered features
        full_df = full_df.drop(columns=["f_27"])
        full_df = pd.concat([full_df, f27_features], axis=1)

        # 4. Define Feature Groups
        # Categorical: f_29, f_30, and decomposed f_27 chars
        cat_cols = [f"f_27_{i}" for i in range(10)] + ["f_29", "f_30"]

        # Continuous: All other columns except 'split'
        cont_cols = [c for c in full_df.columns if c not in cat_cols + ["split"]]

        # 5. Global Vocabulary Alignment (Ordinal Encoding)
        # Fit on ALL data to handle all tokens
        enc = OrdinalEncoder(dtype=np.int64)
        full_df[cat_cols] = enc.fit_transform(full_df[cat_cols].astype(str))

        # Calculate cardinalities for embedding layers
        cat_cardinalities = [int(full_df[col].max() + 1) for col in cat_cols]

        # 6. Normalization (StandardScaler)
        # Fit ONLY on Training data to prevent leakage
        scaler = StandardScaler()
        train_mask = full_df["split"] == "train"
        scaler.fit(full_df.loc[train_mask, cont_cols])
        full_df[cont_cols] = scaler.transform(full_df[cont_cols])

        # 7. Split back into Train, Val, Test
        x_train = full_df[full_df["split"] == "train"]
        x_val = full_df[full_df["split"] == "val"]
        x_test = full_df[full_df["split"] == "test"]

        # Convert to numpy arrays
        train_cont = x_train[cont_cols].values.astype(np.float32)
        train_cat = x_train[cat_cols].values.astype(np.int64)

        val_cont = x_val[cont_cols].values.astype(np.float32)
        val_cat = x_val[cat_cols].values.astype(np.int64)

        test_cont = x_test[cont_cols].values.astype(np.float32)
        test_cat = x_test[cat_cols].values.astype(np.int64)

        # 8. Save to Cache
        np.save(self.cache_files["train_cont"], train_cont)
        np.save(self.cache_files["train_cat"], train_cat)
        np.save(self.cache_files["train_y"], train_y)
        np.save(self.cache_files["val_cont"], val_cont)
        np.save(self.cache_files["val_cat"], val_cat)
        np.save(self.cache_files["val_y"], val_y)
        np.save(self.cache_files["test_cont"], test_cont)
        np.save(self.cache_files["test_cat"], test_cat)
        np.save(self.cache_files["test_ids"], test_ids)

        meta = {"cat_cardinalities": cat_cardinalities, "num_cont": len(cont_cols)}
        np.save(self.cache_files["meta"], meta)

        data = {
            "train_cont": train_cont,
            "train_cat": train_cat,
            "train_y": train_y,
            "val_cont": val_cont,
            "val_cat": val_cat,
            "val_y": val_y,
            "test_cont": test_cont,
            "test_cat": test_cat,
            "test_ids": test_ids,
        }

        return data, meta
