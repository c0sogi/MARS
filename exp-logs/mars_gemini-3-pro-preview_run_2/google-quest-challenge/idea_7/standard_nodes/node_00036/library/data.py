import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from library.config import Config
from library.utils import seed_everything


class QADataset(Dataset):
    def __init__(
        self,
        q_input_ids,
        q_attention_mask,
        q_segment_ids,
        a_input_ids,
        a_attention_mask,
        cats,
        labels=None,
    ):
        self.q_input_ids = q_input_ids
        self.q_attention_mask = q_attention_mask
        self.q_segment_ids = q_segment_ids
        self.a_input_ids = a_input_ids
        self.a_attention_mask = a_attention_mask
        self.cats = cats
        self.labels = labels

    def __len__(self):
        return len(self.q_input_ids)

    def __getitem__(self, idx):
        item = {
            "q_input_ids": torch.tensor(self.q_input_ids[idx], dtype=torch.long),
            "q_attention_mask": torch.tensor(
                self.q_attention_mask[idx], dtype=torch.long
            ),
            "q_segment_ids": torch.tensor(self.q_segment_ids[idx], dtype=torch.long),
            "a_input_ids": torch.tensor(self.a_input_ids[idx], dtype=torch.long),
            "a_attention_mask": torch.tensor(
                self.a_attention_mask[idx], dtype=torch.long
            ),
            "cats": torch.tensor(self.cats[idx], dtype=torch.long),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


def generate_segment_ids(input_ids, sep_token_id):
    """
    Generates segment_ids for Granular Pooling.
    0: Special/Pad
    1: Title Tokens
    2: Body Tokens
    """
    segment_ids = np.zeros_like(input_ids, dtype=np.int8)

    for i, seq in enumerate(input_ids):
        sep_indices = np.where(seq == sep_token_id)[0]
        if len(sep_indices) >= 1:
            first_sep = sep_indices[0]
            # Title: indices from 1 to first_sep - 1 (excluding CLS and SEP)
            if first_sep > 1:
                segment_ids[i, 1:first_sep] = 1

            # Body: indices from first_sep + 1 to second SEP (or end of valid sequence)
            start_body = first_sep + 1
            end_body = (
                sep_indices[1]
                if len(sep_indices) > 1
                else np.where(seq == 0)[0][0] if 0 in seq else len(seq) - 1
            )

            # Adjust if second SEP is the boundary
            if len(sep_indices) > 1:
                end_body = sep_indices[1]

            if end_body > start_body:
                segment_ids[i, start_body:end_body] = 2

    return segment_ids


def process_data(df, tokenizer, fit_encoders=None):
    # --- Stream 1: Question (Title + Body) ---
    titles = df["question_title"].fillna("").astype(str).tolist()
    bodies = df["question_body"].fillna("").astype(str).tolist()

    q_enc = tokenizer(
        titles,
        bodies,
        max_length=Config.MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
        return_token_type_ids=False,
    )

    q_input_ids = q_enc["input_ids"]
    q_attention_mask = q_enc["attention_mask"]
    q_segment_ids = generate_segment_ids(q_input_ids, tokenizer.sep_token_id)

    # --- Stream 2: Answer ---
    answers = df["answer"].fillna("").astype(str).tolist()
    a_enc = tokenizer(
        answers,
        max_length=Config.MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
        return_token_type_ids=False,
    )
    a_input_ids = a_enc["input_ids"]
    a_attention_mask = a_enc["attention_mask"]

    # --- Categorical Features ---
    cat_data = []
    encoders = {}

    for col in Config.CAT_COLS:
        # Fill missing with a placeholder
        vals = df[col].fillna("unknown").astype(str).values

        if fit_encoders is not None:
            le = fit_encoders[col]
            # Handle unseen labels by mapping to the first class (usually safe fallback or 'unknown' if present)
            known_classes = set(le.classes_)
            # Map unknown values to the first class in the encoder
            vals = np.array([x if x in known_classes else le.classes_[0] for x in vals])
            encoded = le.transform(vals)
        else:
            le = LabelEncoder()
            encoded = le.fit_transform(vals)
            encoders[col] = le

        cat_data.append(encoded)

    cats = np.stack(cat_data, axis=1)  # Shape: (N, num_cats)

    # --- Targets ---
    labels = None
    # Check if target columns exist (they won't in test set)
    if all(c in df.columns for c in Config.TARGET_COLS):
        labels = df[Config.TARGET_COLS].values.astype(np.float32)

    return {
        "q_input_ids": q_input_ids,
        "q_attention_mask": q_attention_mask,
        "q_segment_ids": q_segment_ids,
        "a_input_ids": a_input_ids,
        "a_attention_mask": a_attention_mask,
        "cats": cats,
        "labels": labels,
        "encoders": encoders if fit_encoders is None else None,
    }


def get_data_loaders(load_cached_data=True):
    seed_everything(Config.SEED)

    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    splits = ["train", "val", "test"]
    data_dict = {}

    # Define required files for cache check
    def get_filenames(split):
        base = [
            f"{split}_q_input_ids.npy",
            f"{split}_q_attention_mask.npy",
            f"{split}_q_segment_ids.npy",
            f"{split}_a_input_ids.npy",
            f"{split}_a_attention_mask.npy",
            f"{split}_cats.npy",
        ]
        if split != "test":
            base.append(f"{split}_labels.npy")
        return [os.path.join(cache_dir, f) for f in base]

    # Check cache existence
    cache_exists = True
    if load_cached_data:
        for split in splits:
            for f in get_filenames(split):
                if not os.path.exists(f):
                    cache_exists = False
                    break
    else:
        cache_exists = False

    if cache_exists:
        print("Loading data from cache...")
        for split in splits:
            prefix = os.path.join(cache_dir, f"{split}_")
            data_dict[split] = {
                "q_input_ids": np.load(f"{prefix}q_input_ids.npy"),
                "q_attention_mask": np.load(f"{prefix}q_attention_mask.npy"),
                "q_segment_ids": np.load(f"{prefix}q_segment_ids.npy"),
                "a_input_ids": np.load(f"{prefix}a_input_ids.npy"),
                "a_attention_mask": np.load(f"{prefix}a_attention_mask.npy"),
                "cats": np.load(f"{prefix}cats.npy"),
                "labels": np.load(f"{prefix}labels.npy") if split != "test" else None,
            }
    else:
        print("Processing data from scratch...")
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

        # Load DataFrames
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Process Train
        train_out = process_data(train_df, tokenizer, fit_encoders=None)
        encoders = train_out["encoders"]

        # Process Val & Test using trained encoders
        val_out = process_data(val_df, tokenizer, fit_encoders=encoders)
        test_out = process_data(test_df, tokenizer, fit_encoders=encoders)

        data_dict = {"train": train_out, "val": val_out, "test": test_out}

        # Save to cache
        for split, out in data_dict.items():
            prefix = os.path.join(cache_dir, f"{split}_")
            np.save(f"{prefix}q_input_ids.npy", out["q_input_ids"])
            np.save(f"{prefix}q_attention_mask.npy", out["q_attention_mask"])
            np.save(f"{prefix}q_segment_ids.npy", out["q_segment_ids"])
            np.save(f"{prefix}a_input_ids.npy", out["a_input_ids"])
            np.save(f"{prefix}a_attention_mask.npy", out["a_attention_mask"])
            np.save(f"{prefix}cats.npy", out["cats"])
            if out["labels"] is not None:
                np.save(f"{prefix}labels.npy", out["labels"])

    # Initialize Datasets
    train_dataset = QADataset(
        data_dict["train"]["q_input_ids"],
        data_dict["train"]["q_attention_mask"],
        data_dict["train"]["q_segment_ids"],
        data_dict["train"]["a_input_ids"],
        data_dict["train"]["a_attention_mask"],
        data_dict["train"]["cats"],
        data_dict["train"]["labels"],
    )

    val_dataset = QADataset(
        data_dict["val"]["q_input_ids"],
        data_dict["val"]["q_attention_mask"],
        data_dict["val"]["q_segment_ids"],
        data_dict["val"]["a_input_ids"],
        data_dict["val"]["a_attention_mask"],
        data_dict["val"]["cats"],
        data_dict["val"]["labels"],
    )

    test_dataset = QADataset(
        data_dict["test"]["q_input_ids"],
        data_dict["test"]["q_attention_mask"],
        data_dict["test"]["q_segment_ids"],
        data_dict["test"]["a_input_ids"],
        data_dict["test"]["a_attention_mask"],
        data_dict["test"]["cats"],
        None,
    )

    # Initialize Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
