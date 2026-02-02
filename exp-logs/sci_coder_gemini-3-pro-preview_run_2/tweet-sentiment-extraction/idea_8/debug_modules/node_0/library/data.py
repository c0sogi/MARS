import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, Sampler
from library.config import Config
from library.utils import seed_everything

# Disable tokenizer parallelism to prevent deadlocks in DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SmartBatchingCollate:
    """
    Collate function that performs dynamic padding.
    It pads the batch to the maximum sequence length found in that specific batch,
    rather than the global max length, to optimize compute.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        # batch is a list of dicts from TweetDataset.__getitem__

        # Calculate actual lengths of sequences in this batch (ignoring padding)
        # input_ids are already padded to MAX_LEN in Dataset, so we count non-pad tokens
        lengths = [torch.sum(x["input_ids"] != self.pad_token_id).item() for x in batch]
        max_len = max(lengths)

        output = {}

        # Tensors that need to be stacked and truncated
        tensor_keys_1d = ["input_ids", "attention_mask", "token_type_ids"]

        for key in batch[0].keys():
            if key in tensor_keys_1d:
                # Stack (B, MAX_LEN) -> Slice (B, max_len)
                tensors = [x[key] for x in batch]
                stacked = torch.stack(tensors)
                output[key] = stacked[:, :max_len]

            elif key == "offsets":
                # Stack (B, MAX_LEN, 2) -> Slice (B, max_len, 2)
                tensors = [x[key] for x in batch]
                stacked = torch.stack(tensors)
                output[key] = stacked[:, :max_len, :]

            else:
                # Pass through other keys (start_idx, end_idx, text, etc.) as lists
                output[key] = [x[key] for x in batch]

        # Convert scalar targets to tensors if they exist
        if "start_idx" in output:
            output["start_idx"] = torch.tensor(output["start_idx"], dtype=torch.long)
        if "end_idx" in output:
            output["end_idx"] = torch.tensor(output["end_idx"], dtype=torch.long)

        return output


class SmartBatchSampler(Sampler):
    """
    Sampler that groups samples of similar lengths together.
    1. Sorts the dataset by length.
    2. Chunks into batches.
    3. Shuffles the batches to ensure randomness across epochs.
    """

    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Use pre-calculated lengths from the dataset
        if hasattr(data_source, "lengths"):
            self.lengths = data_source.lengths
        else:
            # Fallback
            self.lengths = [len(x["input_ids"]) for x in data_source]

    def __iter__(self):
        indices = np.arange(len(self.data_source))

        # Sort indices by sequence length
        sorted_indices = indices[np.argsort(self.lengths)]

        # Create batches
        batches = [
            sorted_indices[i : i + self.batch_size]
            for i in range(0, len(sorted_indices), self.batch_size)
        ]

        # Shuffle the batches
        if self.shuffle:
            np.random.shuffle(batches)

        # Flatten back to a list of indices for the DataLoader
        final_indices = []
        for batch in batches:
            final_indices.extend(batch)

        return iter(final_indices)

    def __len__(self):
        return len(self.data_source) // self.batch_size


class TweetDataset(Dataset):
    def __init__(self, df, data_dict):
        self.df = df
        self.input_ids = data_dict["input_ids"]
        self.attention_mask = data_dict["attention_mask"]
        self.token_type_ids = data_dict["token_type_ids"]
        self.offsets = data_dict["offsets"]
        self.lengths = data_dict["lengths"]

        # Targets (only for train/val)
        self.start_idx = data_dict.get("start_idx")
        self.end_idx = data_dict.get("end_idx")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "token_type_ids": torch.tensor(self.token_type_ids[idx], dtype=torch.long),
            "offsets": torch.tensor(self.offsets[idx], dtype=torch.long),
            "text": str(self.df.iloc[idx]["text"]),
            "textID": str(self.df.iloc[idx]["textID"]),
            "sentiment": str(self.df.iloc[idx]["sentiment"]),
        }

        if self.start_idx is not None:
            item["start_idx"] = self.start_idx[idx]
            item["end_idx"] = self.end_idx[idx]
            item["selected_text"] = str(self.df.iloc[idx]["selected_text"])

        return item


def process_data(df, tokenizer, max_len, is_test=False):
    """
    Tokenizes data, extracts targets, and prepares numpy arrays for caching.
    """
    input_ids = []
    attention_mask = []
    token_type_ids = []
    offsets = []
    start_indices = []
    end_indices = []
    lengths = []

    for idx, row in df.iterrows():
        text = str(row["text"])
        sentiment = str(row["sentiment"])

        # Tokenize: [CLS] sentiment [SEP] text [SEP]
        # DeBERTa v3 handles this structure via text_pair
        encoded = tokenizer(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
        )

        input_ids.append(encoded["input_ids"])
        attention_mask.append(encoded["attention_mask"])
        token_type_ids.append(encoded["token_type_ids"])
        offsets.append(encoded["offset_mapping"])

        # Calculate actual length (sum of attention mask) for smart batching
        lengths.append(sum(encoded["attention_mask"]))

        if not is_test:
            selected_text = str(row["selected_text"])

            # Find character start/end of selected_text in text
            start_char = text.find(selected_text)
            if start_char == -1:
                # Fallback: try finding stripped version
                start_char = text.find(selected_text.strip())

            if start_char == -1:
                # Fallback: use full text if matching fails
                start_char = 0
                end_char = len(text)
            else:
                end_char = start_char + len(selected_text)

            # Map character positions to token indices
            sequence_ids = encoded.sequence_ids()
            offset_mapping = encoded["offset_mapping"]

            # Identify tokens belonging to the 'text' part (sequence_id == 1)
            text_token_indices = [
                i for i, seq_id in enumerate(sequence_ids) if seq_id == 1
            ]

            if not text_token_indices:
                start_indices.append(0)
                end_indices.append(0)
                continue

            s_idx = text_token_indices[0]
            e_idx = text_token_indices[-1]

            found_start = False

            for i in text_token_indices:
                token_start, token_end = offset_mapping[i]

                # If this token overlaps with the start of the selection
                if (
                    not found_start
                    and token_start <= start_char
                    and token_end > start_char
                ):
                    s_idx = i
                    found_start = True

                # If this token is inside the selection or contains the end
                if token_start < end_char:
                    e_idx = i

            start_indices.append(s_idx)
            end_indices.append(e_idx)

    data = {
        "input_ids": np.array(input_ids),
        "attention_mask": np.array(attention_mask),
        "token_type_ids": np.array(token_type_ids),
        "offsets": np.array(offsets),
        "lengths": np.array(lengths),
    }

    if not is_test:
        data["start_idx"] = np.array(start_indices)
        data["end_idx"] = np.array(end_indices)

    return data


def get_loaders(load_cached_data=True):
    seed_everything(Config.SEED)

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load Metadata DataFrames
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    val_df = pd.read_csv(Config.VAL_META_PATH)
    test_df = pd.read_csv(Config.TEST_META_PATH)

    if Config.DEBUG:
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        test_df = test_df.head(50)

    # Define Cache Paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "cached_train_v8.npz")
    val_cache = os.path.join(cache_dir, "cached_val_v8.npz")
    test_cache = os.path.join(cache_dir, "cached_test_v8.npz")

    # Helper to load or process data
    def get_data(df, cache_path, is_test=False):
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            return dict(data)
        else:
            print(f"Processing data for {cache_path}...")
            data = process_data(df, tokenizer, Config.MAX_LEN, is_test)
            np.savez(cache_path, **data)
            return data

    # Load Data
    train_data = get_data(train_df, train_cache, is_test=False)
    val_data = get_data(val_df, val_cache, is_test=False)
    test_data = get_data(test_df, test_cache, is_test=True)

    # Create Datasets
    train_dataset = TweetDataset(train_df, train_data)
    val_dataset = TweetDataset(val_df, val_data)
    test_dataset = TweetDataset(test_df, test_data)

    # Collate Function for Dynamic Padding
    collate_fn = SmartBatchingCollate(tokenizer)

    # Create DataLoaders
    # Train: Use SmartBatchSampler to group by length
    train_sampler = SmartBatchSampler(
        train_dataset, Config.TRAIN_BATCH_SIZE, shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Val/Test: Sequential sampling is fine; dynamic padding still applies via collate
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
