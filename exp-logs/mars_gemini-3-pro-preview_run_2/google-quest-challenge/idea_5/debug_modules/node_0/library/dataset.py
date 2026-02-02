import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from library.config import Config


def save_flattened_array(data_list, path_prefix):
    """
    Flattens a list of variable-length arrays and saves them along with offsets.
    """
    # Calculate offsets
    lengths = [len(x) for x in data_list]
    offsets = np.cumsum([0] + lengths)

    # Flatten
    if len(data_list) > 0:
        flat_data = np.concatenate(data_list)
    else:
        flat_data = np.array([])

    np.save(f"{path_prefix}_flat.npy", flat_data)
    np.save(f"{path_prefix}_offsets.npy", offsets)


def load_flattened_array(path_prefix):
    """
    Loads flattened data and offsets.
    """
    flat_data = np.load(f"{path_prefix}_flat.npy")
    offsets = np.load(f"{path_prefix}_offsets.npy")
    return flat_data, offsets


def preprocess_and_cache(split_name, load_cached_data=True):
    """
    Tokenizes data for both views, handles caching using flattened numpy arrays.
    """
    cache_dir = Config.working_dir
    os.makedirs(cache_dir, exist_ok=True)

    # Define file paths
    prefix = os.path.join(cache_dir, split_name)
    files_to_check = [
        f"{prefix}_v1_ids_flat.npy",
        f"{prefix}_v1_ids_offsets.npy",
        f"{prefix}_v1_mask_flat.npy",
        f"{prefix}_v1_mask_offsets.npy",
        f"{prefix}_v2_ids_flat.npy",
        f"{prefix}_v2_ids_offsets.npy",
        f"{prefix}_v2_mask_flat.npy",
        f"{prefix}_v2_mask_offsets.npy",
        f"{prefix}_v2_type_flat.npy",
        f"{prefix}_v2_type_offsets.npy",
    ]

    has_labels = split_name in ["train", "val"]
    if has_labels:
        files_to_check.append(f"{prefix}_labels.npy")

    # Check cache
    cache_exists = all(os.path.exists(f) for f in files_to_check)

    if load_cached_data and cache_exists:
        print(f"Loading cached data for {split_name} from {cache_dir}...")
        v1_ids, v1_offsets = load_flattened_array(f"{prefix}_v1_ids")
        v1_mask, _ = load_flattened_array(f"{prefix}_v1_mask")

        v2_ids, v2_offsets = load_flattened_array(f"{prefix}_v2_ids")
        v2_mask, _ = load_flattened_array(f"{prefix}_v2_mask")
        v2_type, _ = load_flattened_array(f"{prefix}_v2_type")

        labels = np.load(f"{prefix}_labels.npy") if has_labels else None

        return {
            "v1_ids": v1_ids,
            "v1_offsets": v1_offsets,
            "v1_mask": v1_mask,
            "v2_ids": v2_ids,
            "v2_offsets": v2_offsets,
            "v2_mask": v2_mask,
            "v2_type": v2_type,
            "labels": labels,
        }

    # Process from scratch
    print(f"Processing data for {split_name}...")

    # Load CSV
    if split_name == "train":
        df = pd.read_csv(Config.train_path)
    elif split_name == "val":
        df = pd.read_csv(Config.val_path)
    else:
        df = pd.read_csv(Config.test_path)

    if Config.debug:
        df = df.head(100).copy()
        print("Debug mode: Sampled 100 rows.")

    # Prepare text
    df["question_title"] = df["question_title"].fillna("").astype(str)
    df["question_body"] = df["question_body"].fillna("").astype(str)
    df["answer"] = df["answer"].fillna("").astype(str)

    q_text = (df["question_title"] + " " + df["question_body"]).tolist()
    a_text = df["answer"].tolist()

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # View 1: Intrinsic (Question only)
    print("Tokenizing View 1...")
    v1_enc = tokenizer(
        q_text,
        truncation=True,
        max_length=Config.max_len,
        padding=False,  # Dynamic padding later
        add_special_tokens=True,
    )

    # View 2: Relational (Question + Answer)
    print("Tokenizing View 2...")
    v2_enc = tokenizer(
        q_text,
        a_text,
        truncation=True,
        max_length=Config.max_len,
        padding=False,
        add_special_tokens=True,
    )

    # Save View 1
    save_flattened_array(
        [np.array(x, dtype=np.int32) for x in v1_enc["input_ids"]], f"{prefix}_v1_ids"
    )
    save_flattened_array(
        [np.array(x, dtype=np.int8) for x in v1_enc["attention_mask"]],
        f"{prefix}_v1_mask",
    )

    # Save View 2
    save_flattened_array(
        [np.array(x, dtype=np.int32) for x in v2_enc["input_ids"]], f"{prefix}_v2_ids"
    )
    save_flattened_array(
        [np.array(x, dtype=np.int8) for x in v2_enc["attention_mask"]],
        f"{prefix}_v2_mask",
    )
    # DeBERTa tokenizer produces token_type_ids (0 for Q, 1 for A) when pairs are passed
    if "token_type_ids" in v2_enc:
        save_flattened_array(
            [np.array(x, dtype=np.int8) for x in v2_enc["token_type_ids"]],
            f"{prefix}_v2_type",
        )
    else:
        # Fallback if tokenizer doesn't return type ids (unlikely for DeBERTa-v3-base in pair mode)
        # Create dummy type ids (all 0) - this would break the logic, but assuming correct tokenizer usage
        print("Warning: token_type_ids not found. Creating zeros.")
        save_flattened_array(
            [np.zeros(len(x), dtype=np.int8) for x in v2_enc["input_ids"]],
            f"{prefix}_v2_type",
        )

    # Save Labels
    labels = None
    if has_labels:
        labels = df[Config.target_cols].values.astype(np.float32)
        np.save(f"{prefix}_labels.npy", labels)

    # Reload to return consistent format
    return load_and_preprocess(split_name, load_cached_data=True)


class QuestDataset(Dataset):
    def __init__(self, data_dict):
        self.v1_ids = data_dict["v1_ids"]
        self.v1_offsets = data_dict["v1_offsets"]
        self.v1_mask = data_dict["v1_mask"]

        self.v2_ids = data_dict["v2_ids"]
        self.v2_offsets = data_dict["v2_offsets"]
        self.v2_mask = data_dict["v2_mask"]
        self.v2_type = data_dict["v2_type"]

        self.labels = data_dict["labels"]
        self.length = len(self.v1_offsets) - 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # View 1
        start_v1, end_v1 = self.v1_offsets[idx], self.v1_offsets[idx + 1]
        v1_input_ids = self.v1_ids[start_v1:end_v1]
        v1_attention_mask = self.v1_mask[start_v1:end_v1]

        # View 2
        start_v2, end_v2 = self.v2_offsets[idx], self.v2_offsets[idx + 1]
        v2_input_ids = self.v2_ids[start_v2:end_v2]
        v2_attention_mask = self.v2_mask[start_v2:end_v2]
        v2_token_type_ids = self.v2_type[start_v2:end_v2]

        item = {
            "view1_input_ids": torch.tensor(v1_input_ids, dtype=torch.long),
            "view1_attention_mask": torch.tensor(v1_attention_mask, dtype=torch.long),
            "view2_input_ids": torch.tensor(v2_input_ids, dtype=torch.long),
            "view2_attention_mask": torch.tensor(v2_attention_mask, dtype=torch.long),
            "view2_token_type_ids": torch.tensor(v2_token_type_ids, dtype=torch.long),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


class Collate:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        # Extract lists
        v1_ids = [item["view1_input_ids"] for item in batch]
        v1_mask = [item["view1_attention_mask"] for item in batch]
        v2_ids = [item["view2_input_ids"] for item in batch]
        v2_mask = [item["view2_attention_mask"] for item in batch]
        v2_type = [item["view2_token_type_ids"] for item in batch]

        # Pad sequences
        # input_ids pad with pad_token_id
        v1_ids_padded = pad_sequence(
            v1_ids, batch_first=True, padding_value=self.pad_token_id
        )
        v2_ids_padded = pad_sequence(
            v2_ids, batch_first=True, padding_value=self.pad_token_id
        )

        # masks pad with 0
        v1_mask_padded = pad_sequence(v1_mask, batch_first=True, padding_value=0)
        v2_mask_padded = pad_sequence(v2_mask, batch_first=True, padding_value=0)

        # token_type_ids pad with 0 (usually safe as long as we use attention mask)
        v2_type_padded = pad_sequence(v2_type, batch_first=True, padding_value=0)

        # Generate View 2 Masks (Q vs A)
        # Q: type 0 and not padding
        # A: type 1 and not padding
        v2_q_mask = (v2_type_padded == 0) & (v2_mask_padded == 1)
        v2_a_mask = (v2_type_padded == 1) & (v2_mask_padded == 1)

        batch_out = {
            "view1_input_ids": v1_ids_padded,
            "view1_attention_mask": v1_mask_padded,
            "view2_input_ids": v2_ids_padded,
            "view2_attention_mask": v2_mask_padded,
            "view2_q_mask": v2_q_mask.long(),
            "view2_a_mask": v2_a_mask.long(),
        }

        if "labels" in batch[0]:
            labels = torch.stack([item["labels"] for item in batch])
            batch_out["labels"] = labels

        return batch_out


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders for Train, Val, and Test.
    """
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    collate_fn = Collate(tokenizer)

    # Train
    train_data = preprocess_and_cache("train", load_cached_data)
    train_dataset = QuestDataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Val
    val_data = preprocess_and_cache("val", load_cached_data)
    val_dataset = QuestDataset(val_data)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Test
    test_data = preprocess_and_cache("test", load_cached_data)
    test_dataset = QuestDataset(test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
