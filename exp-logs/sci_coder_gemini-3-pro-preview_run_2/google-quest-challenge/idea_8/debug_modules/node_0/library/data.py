import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything


class QADataset(Dataset):
    """
    Dataset class for Siamese DeBERTa with Granular Co-Attention.
    Handles Question (Title+Body) and Answer streams separately.
    """

    def __init__(self, data_dict, is_test=False):
        self.q_input_ids = data_dict["q_input_ids"]
        self.q_attention_mask = data_dict["q_attention_mask"]
        self.q_token_type_ids = data_dict[
            "q_token_type_ids"
        ]  # Segment mask for Title vs Body

        self.a_input_ids = data_dict["a_input_ids"]
        self.a_attention_mask = data_dict["a_attention_mask"]

        self.cats = data_dict["cats"]
        self.is_test = is_test

        if not self.is_test:
            self.targets = data_dict["targets"]

    def __len__(self):
        return len(self.q_input_ids)

    def __getitem__(self, idx):
        item = {
            "q_input_ids": torch.tensor(self.q_input_ids[idx], dtype=torch.long),
            "q_attention_mask": torch.tensor(
                self.q_attention_mask[idx], dtype=torch.long
            ),
            "q_token_type_ids": torch.tensor(
                self.q_token_type_ids[idx], dtype=torch.long
            ),
            "a_input_ids": torch.tensor(self.a_input_ids[idx], dtype=torch.long),
            "a_attention_mask": torch.tensor(
                self.a_attention_mask[idx], dtype=torch.long
            ),
            "cats": torch.tensor(self.cats[idx], dtype=torch.long),
        }

        if not self.is_test:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float)

        return item


def _tokenize_data(df, tokenizer, max_len_q, max_len_a):
    """
    Helper to tokenize dataframe columns.
    """
    # Question Stream: [CLS] Title [SEP] Body [SEP]
    # We use text_pair to automatically handle [SEP] insertion and token_type_ids generation.
    # token_type_ids will be 0 for Title and 1 for Body.
    q_encodings = tokenizer(
        df["question_title"].astype(str).tolist(),
        df["question_body"].astype(str).tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_len_q,
        return_token_type_ids=True,
        return_attention_mask=True,
    )

    # Answer Stream: [CLS] Answer [SEP]
    a_encodings = tokenizer(
        df["answer"].astype(str).tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_len_a,
        return_token_type_ids=False,  # Not needed for single sequence
        return_attention_mask=True,
    )

    return {
        "q_input_ids": np.array(q_encodings["input_ids"], dtype=np.int32),
        "q_attention_mask": np.array(q_encodings["attention_mask"], dtype=np.int8),
        "q_token_type_ids": np.array(q_encodings["token_type_ids"], dtype=np.int8),
        "a_input_ids": np.array(a_encodings["input_ids"], dtype=np.int32),
        "a_attention_mask": np.array(a_encodings["attention_mask"], dtype=np.int8),
    }


def _encode_categoricals(df, cat_cols, cat_maps):
    """
    Helper to encode categorical columns using pre-computed maps.
    """
    encoded_cats = []
    for col in cat_cols:
        # Map values to indices, default to 0 if unknown (though maps cover all train/val/test)
        mapping = cat_maps[col]
        encoded = df[col].astype(str).map(mapping).fillna(0).astype(np.int32).values
        encoded_cats.append(encoded)

    return np.stack(encoded_cats, axis=1)


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Main function to load data, process it (with caching), and return DataLoaders.
    """
    seed_everything(Config.seed)

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # Define cache filenames
    suffix = "_debug" if debug else ""
    cache_files = {
        "train": os.path.join(Config.working_dir, f"train_data{suffix}.npz"),
        "val": os.path.join(Config.working_dir, f"val_data{suffix}.npz"),
        "test": os.path.join(Config.working_dir, f"test_data{suffix}.npz"),
        "meta": os.path.join(
            Config.working_dir, f"meta_data{suffix}.npz"
        ),  # To store cat mappings if needed
    }

    # Check if we can load from cache
    all_cached = all(
        os.path.exists(f) for f in cache_files.values() if f != cache_files["meta"]
    )

    if load_cached_data and all_cached:
        print(f"Loading cached data from {Config.working_dir}...")
        train_data = np.load(cache_files["train"])
        val_data = np.load(cache_files["val"])
        test_data = np.load(cache_files["test"])

        # Convert npz files back to dicts
        train_dict = {k: train_data[k] for k in train_data.files}
        val_dict = {k: val_data[k] for k in val_data.files}
        test_dict = {k: test_data[k] for k in test_data.files}

    else:
        print("Processing data from scratch...")

        # 1. Load Dataframes
        df_train = pd.read_csv(Config.train_path)
        df_val = pd.read_csv(Config.val_path)
        df_test = pd.read_csv(Config.test_path)

        # Handle Debug Mode
        if debug:
            print(f"Debug mode: subsetting to {Config.debug_sample_size} rows.")
            df_train = df_train.iloc[: Config.debug_sample_size]
            df_val = df_val.iloc[: Config.debug_sample_size]
            df_test = df_test.iloc[: Config.debug_sample_size]

        # 2. Build Categorical Mappings (Global)
        # We combine all splits to ensure we capture all categories
        cat_maps = {}
        for col in Config.cat_cols:
            all_vals = pd.concat(
                [
                    df_train[col].astype(str),
                    df_val[col].astype(str),
                    df_test[col].astype(str),
                ]
            ).unique()
            all_vals = sorted(all_vals)  # Deterministic order
            cat_maps[col] = {val: i for i, val in enumerate(all_vals)}

        # 3. Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

        # 4. Process Splits
        def process_split(df, is_test=False):
            # Tokenize
            data_dict = _tokenize_data(
                df, tokenizer, Config.max_len_q, Config.max_len_a
            )

            # Categoricals
            data_dict["cats"] = _encode_categoricals(df, Config.cat_cols, cat_maps)

            # Targets
            if not is_test:
                data_dict["targets"] = df[Config.target_cols].values.astype(np.float32)
            else:
                # Placeholder for test set structure consistency
                data_dict["targets"] = np.zeros(
                    (len(df), len(Config.target_cols)), dtype=np.float32
                )

            return data_dict

        print("Tokenizing Train...")
        train_dict = process_split(df_train)
        print("Tokenizing Val...")
        val_dict = process_split(df_val)
        print("Tokenizing Test...")
        test_dict = process_split(df_test, is_test=True)

        # 5. Save to Cache
        print(f"Saving processed data to {Config.working_dir}...")
        np.savez(cache_files["train"], **train_dict)
        np.savez(cache_files["val"], **val_dict)
        np.savez(cache_files["test"], **test_dict)
        # We don't strictly need to save meta for this pipeline as mappings are implicit in the encoded integers,
        # but good practice if we needed to decode later.
        np.savez(cache_files["meta"], dummy=np.array([0]))

    # Create Datasets
    train_dataset = QADataset(train_dict, is_test=False)
    val_dataset = QADataset(val_dict, is_test=False)
    test_dataset = QADataset(test_dict, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
