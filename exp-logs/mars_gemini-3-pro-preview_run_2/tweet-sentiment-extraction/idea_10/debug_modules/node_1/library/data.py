import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
from library.config import Config
from library.utils import seed_everything


class TweetDataset(Dataset):
    """
    Dataset class for Tweet Sentiment Extraction.
    Returns tokenized inputs, attention masks, and target indices.
    """

    def __init__(
        self,
        input_ids,
        attention_masks,
        token_type_ids,
        start_labels,
        end_labels,
        offsets,
        orig_texts,
        sentiments,
        selected_texts=None,
    ):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.start_labels = start_labels
        self.end_labels = end_labels
        self.offsets = offsets
        self.orig_texts = orig_texts
        self.sentiments = sentiments
        self.selected_texts = selected_texts

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "token_type_ids": torch.tensor(self.token_type_ids[idx], dtype=torch.long),
            "start_labels": torch.tensor(self.start_labels[idx], dtype=torch.long),
            "end_labels": torch.tensor(self.end_labels[idx], dtype=torch.long),
            "offsets": torch.tensor(self.offsets[idx], dtype=torch.long),
            "orig_text": self.orig_texts[idx],
            "sentiment": self.sentiments[idx],
            "selected_text": (
                self.selected_texts[idx] if self.selected_texts is not None else ""
            ),
        }


class SmartBatchingCollate:
    """
    Collate function that dynamically pads the batch to the maximum length
    of sequences in that specific batch, optimizing computational efficiency.
    """

    def __call__(self, batch):
        # Extract sequences
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        token_type_ids = [item["token_type_ids"] for item in batch]
        start_labels = [item["start_labels"] for item in batch]
        end_labels = [item["end_labels"] for item in batch]
        offsets = [item["offsets"] for item in batch]

        # Determine max length in this batch
        max_len = max([len(x) for x in input_ids])

        # Helper to pad tensors
        def pad_tensor(tensor, length, padding_value=0):
            if len(tensor) >= length:
                return tensor[:length]
            pad_size = length - len(tensor)
            return torch.cat(
                [tensor, torch.tensor([padding_value] * pad_size, dtype=tensor.dtype)]
            )

        # Helper to pad offsets (which are 2D)
        def pad_offsets(tensor, length):
            if len(tensor) >= length:
                return tensor[:length]
            pad_size = length - len(tensor)
            # Pad with (0,0)
            padding = torch.zeros((pad_size, 2), dtype=tensor.dtype)
            return torch.cat([tensor, padding])

        # Apply padding
        input_ids_padded = torch.stack([pad_tensor(x, max_len, 0) for x in input_ids])
        attention_mask_padded = torch.stack(
            [pad_tensor(x, max_len, 0) for x in attention_mask]
        )
        token_type_ids_padded = torch.stack(
            [pad_tensor(x, max_len, 0) for x in token_type_ids]
        )

        # Labels are scalars, just stack them
        start_labels_stacked = torch.stack(start_labels)
        end_labels_stacked = torch.stack(end_labels)

        # Pad offsets
        offsets_padded = torch.stack([pad_offsets(x, max_len) for x in offsets])

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "token_type_ids": token_type_ids_padded,
            "start_labels": start_labels_stacked,
            "end_labels": end_labels_stacked,
            "offsets": offsets_padded,
            "orig_text": [item["orig_text"] for item in batch],
            "sentiment": [item["sentiment"] for item in batch],
            "selected_text": [item["selected_text"] for item in batch],
        }


class LengthSortedBatchSampler(Sampler):
    """
    Sampler that yields batches of indices sorted by sequence length.
    This minimizes padding within batches. Batches themselves are shuffled.
    """

    def __init__(self, dataset, batch_size, drop_last=False, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        # Access lengths directly from the dataset's internal storage for speed
        lengths = [len(x) for x in self.dataset.input_ids]
        indices = np.argsort(lengths)

        # Create batches
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i : i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)

        # Shuffle the order of batches, not the samples within batches
        if self.shuffle:
            np.random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def process_data(df, tokenizer, config, mode="train", load_cached_data=True):
    """
    Tokenizes data and finds targets. Implements caching to .npz files.
    """
    # Ensure cache directory exists
    os.makedirs(config.cache_dir, exist_ok=True)
    cache_file = os.path.join(config.cache_dir, f"cached_{mode}_{config.max_len}.npz")

    # Load from cache if available
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        data = np.load(cache_file, allow_pickle=True)
        return (
            data["input_ids"],
            data["attention_masks"],
            data["token_type_ids"],
            data["start_labels"],
            data["end_labels"],
            data["offsets"],
            data["orig_texts"],
            data["sentiments"],
            data["selected_texts"],
        )

    print(f"Processing {mode} data...")

    input_ids_list = []
    attention_masks_list = []
    token_type_ids_list = []
    start_labels_list = []
    end_labels_list = []
    offsets_list = []
    orig_texts_list = []
    sentiments_list = []
    selected_texts_list = []

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Basic cleaning: collapse whitespace
        text = str(row["text"])
        text_clean = " ".join(text.split())
        sentiment = str(row["sentiment"])

        orig_texts_list.append(text_clean)
        sentiments_list.append(sentiment)

        selected_text = (
            str(row["selected_text"])
            if "selected_text" in row and not pd.isna(row["selected_text"])
            else None
        )
        selected_texts_list.append(selected_text if selected_text is not None else "")

        # Tokenize sentiment (Context)
        sent_tokens = tokenizer.encode(sentiment, add_special_tokens=False)

        # Tokenize text with offsets
        # Use clean text for tokenization
        encoded_text = tokenizer.encode_plus(
            text_clean, add_special_tokens=False, return_offsets_mapping=True
        )
        text_ids = encoded_text["input_ids"]
        text_offsets = encoded_text["offset_mapping"]

        # Construct full sequence: [CLS] sentiment [SEP] text [SEP]
        input_ids = [cls_id] + sent_tokens + [sep_id] + text_ids + [sep_id]
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        # Construct Offsets: (0,0) for special tokens
        prefix_len = 1 + len(sent_tokens) + 1
        offsets = [(0, 0)] * prefix_len + text_offsets + [(0, 0)]

        # Truncate if necessary
        if len(input_ids) > config.max_len:
            input_ids = input_ids[: config.max_len]
            attention_mask = attention_mask[: config.max_len]
            token_type_ids = token_type_ids[: config.max_len]
            offsets = offsets[: config.max_len]

        # Find Targets (Start/End Indices)
        start_idx = 0
        end_idx = 0

        if selected_text is not None:
            # We must find the selected_text within the text we tokenized.
            # Since we cleaned the text, we should ideally find the cleaned selected_text.
            # However, selected_text in metadata is already cleaned/aligned usually.
            # We search for the substring.

            # Use raw text for finding index to be robust against cleaning diffs,
            # but we tokenized text_clean. So we must search in text_clean.
            selected_text_clean = " ".join(selected_text.split())

            idx_start = text_clean.find(selected_text_clean)

            if idx_start != -1:
                idx_end = idx_start + len(selected_text_clean)

                # Find tokens that overlap with this character span
                tokens_indices = []
                for i, (o_start, o_end) in enumerate(offsets):
                    if i < prefix_len:
                        continue  # Skip prefix
                    if o_start == o_end:
                        continue  # Skip specials

                    # Calculate overlap
                    intersect_start = max(idx_start, o_start)
                    intersect_end = min(idx_end, o_end)
                    overlap = max(0, intersect_end - intersect_start)

                    if overlap > 0:
                        tokens_indices.append(i)

                if tokens_indices:
                    start_idx = tokens_indices[0]
                    end_idx = tokens_indices[-1]

        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)
        token_type_ids_list.append(token_type_ids)
        start_labels_list.append(start_idx)
        end_labels_list.append(end_idx)
        offsets_list.append(offsets)

    # Convert to object arrays to handle variable lengths
    data_dict = {
        "input_ids": np.array(input_ids_list, dtype=object),
        "attention_masks": np.array(attention_masks_list, dtype=object),
        "token_type_ids": np.array(token_type_ids_list, dtype=object),
        "start_labels": np.array(start_labels_list),
        "end_labels": np.array(end_labels_list),
        "offsets": np.array(offsets_list, dtype=object),
        "orig_texts": np.array(orig_texts_list, dtype=object),
        "sentiments": np.array(sentiments_list, dtype=object),
        "selected_texts": np.array(selected_texts_list, dtype=object),
    }

    # Save to cache
    np.savez(cache_file, **data_dict)
    print(f"Saved processed data to {cache_file}")

    return (
        data_dict["input_ids"],
        data_dict["attention_masks"],
        data_dict["token_type_ids"],
        data_dict["start_labels"],
        data_dict["end_labels"],
        data_dict["offsets"],
        data_dict["orig_texts"],
        data_dict["sentiments"],
        data_dict["selected_texts"],
    )


def get_data_loaders(config):
    """
    Main entry point to get DataLoaders.
    Loads metadata, processes/loads data, and returns dataloaders with smart batching.
    """
    # Load Metadata
    train_df = pd.read_csv(config.train_path)
    val_df = pd.read_csv(config.val_path)
    test_df = pd.read_csv(config.test_path)

    # Handle missing column in test
    test_df["selected_text"] = ""

    # Debug mode
    if config.debug:
        train_df = train_df.head(config.debug_subset_size)
        val_df = val_df.head(config.debug_subset_size)
        test_df = test_df.head(config.debug_subset_size)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Process Data
    train_data = process_data(train_df, tokenizer, config, mode="train")
    val_data = process_data(val_df, tokenizer, config, mode="val")
    test_data = process_data(test_df, tokenizer, config, mode="test")

    # Create Datasets
    train_dataset = TweetDataset(*train_data)
    val_dataset = TweetDataset(*val_data)
    test_dataset = TweetDataset(*test_data)

    # Create Samplers/Loaders

    # Train: Sort by length + shuffle batches (Smart Batching)
    train_sampler = LengthSortedBatchSampler(
        train_dataset, batch_size=config.train_batch_size, drop_last=True, shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=SmartBatchingCollate(),
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Val: Sort by length + no shuffle (Faster inference)
    val_sampler = LengthSortedBatchSampler(
        val_dataset, batch_size=config.valid_batch_size, drop_last=False, shuffle=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=SmartBatchingCollate(),
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Test: Sort by length + no shuffle
    test_sampler = LengthSortedBatchSampler(
        test_dataset, batch_size=config.valid_batch_size, drop_last=False, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        collate_fn=SmartBatchingCollate(),
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
