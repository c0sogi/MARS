import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
from library.config import Config
from library.utils import set_seed


class StackExchangeDataset(Dataset):
    """
    Custom Dataset for Stack Exchange Questions.
    Handles tokenization of Title + Body and retrieval of binary labels.
    """

    def __init__(self, df, tokenizer, max_length, labels=None):
        self.ids = df["Id"].values
        # Ensure text columns are strings and handle potential NaNs (though metadata should be clean)
        self.titles = df["Title"].fillna("").astype(str).values
        self.bodies = df["Body"].fillna("").astype(str).values
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Concatenate Title and Body
        text = self.titles[idx] + " " + self.bodies[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Flatten to remove batch dimension added by tokenizer
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "id": self.ids[idx],
        }

        # Add labels if available (Train/Val)
        if self.labels is not None:
            # Convert numpy bool/uint8 to float tensor for BCEWithLogitsLoss
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


def get_top_tags(df, num_labels):
    """
    Identifies the top `num_labels` most frequent tags in the dataframe.
    """
    print("Computing top tags from training data...")
    # Counter for all tags
    tag_counts = Counter()

    # Iterate in chunks to be memory safe, though 220GB is plenty
    # pandas str.split is efficient enough
    all_tags = df["Tags"].astype(str).str.split()

    for tags in all_tags:
        tag_counts.update(tags)

    # Get top K tags
    top_tags = [tag for tag, count in tag_counts.most_common(num_labels)]
    print(f"Selected {len(top_tags)} top tags.")
    return top_tags


def process_labels(df, classes, cache_path, load_cached_data):
    """
    Converts 'Tags' column to binary matrix using MultiLabelBinarizer.
    Uses caching to speed up subsequent runs.
    """
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached labels from {cache_path}...")
        return np.load(cache_path)

    print(f"Generating label matrix for {len(df)} samples...")
    # Initialize MLB with fixed classes (the top K tags)
    mlb = MultiLabelBinarizer(classes=classes, sparse_output=False)

    # Fit is not needed as we provided classes, but we need to call it to initialize internal state properly
    # or just call transform. MLB with classes provided behaves correctly on transform.
    # However, sklearn recommends fit first.
    mlb.fit([[]])

    # Transform
    # df['Tags'] is space-delimited string. Split into list of strings.
    tags_list = df["Tags"].astype(str).str.split()

    # Transform to binary matrix
    # Note: Tags not in 'classes' are silently ignored by MLB
    label_matrix = mlb.transform(tags_list)

    # Save as .npy (uint8 or bool to save space, but float required for torch later.
    # We save as uint8 to save disk space, cast to float in Dataset)
    label_matrix = label_matrix.astype(np.uint8)

    print(f"Saving labels to {cache_path}...")
    np.save(cache_path, label_matrix)

    return label_matrix


def get_dataloaders(load_cached_data=True):
    """
    Main function to load data and return PyTorch DataLoaders.
    """
    set_seed()

    # Define cache paths based on debug mode
    cache_prefix = "debug_" if Config.debug else ""
    tags_cache_path = os.path.join(Config.working_dir, f"{cache_prefix}tags.json")
    train_labels_path = os.path.join(
        Config.working_dir, f"{cache_prefix}train_labels.npy"
    )
    val_labels_path = os.path.join(Config.working_dir, f"{cache_prefix}val_labels.npy")

    # 1. Load Dataframes
    print("Loading Metadata CSVs...")
    df_train = pd.read_csv(Config.train_path, engine="c")
    df_val = pd.read_csv(Config.val_path, engine="c")
    df_test = pd.read_csv(Config.test_path, engine="c")

    # Handle Debug Mode
    if Config.debug:
        print(f"DEBUG MODE: Truncating datasets to {Config.debug_sample_size} samples.")
        df_train = df_train.head(Config.debug_sample_size).copy()
        df_val = df_val.head(Config.debug_sample_size).copy()
        df_test = df_test.head(Config.debug_sample_size).copy()

    # 2. Determine Tag Vocabulary
    # We need the tag list to define the output dimension and mapping
    if load_cached_data and os.path.exists(tags_cache_path):
        print(f"Loading tag vocabulary from {tags_cache_path}...")
        with open(tags_cache_path, "r") as f:
            tag_vocab = json.load(f)
    else:
        tag_vocab = get_top_tags(df_train, Config.num_labels)
        print(f"Saving tag vocabulary to {tags_cache_path}...")
        with open(tags_cache_path, "w") as f:
            json.dump(tag_vocab, f)

    # Also save to the main Config.tags_path if not in debug mode, for inference reference
    if not Config.debug:
        with open(Config.tags_path, "w") as f:
            json.dump(tag_vocab, f)

    # 3. Process Labels (Train/Val)
    train_labels = process_labels(
        df_train, tag_vocab, train_labels_path, load_cached_data
    )
    val_labels = process_labels(df_val, tag_vocab, val_labels_path, load_cached_data)

    # 4. Initialize Tokenizer
    print(f"Initializing Tokenizer: {Config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # 5. Create Datasets
    print("Creating Datasets...")
    train_dataset = StackExchangeDataset(
        df_train, tokenizer, Config.max_length, labels=train_labels
    )

    val_dataset = StackExchangeDataset(
        df_val, tokenizer, Config.max_length, labels=val_labels
    )

    test_dataset = StackExchangeDataset(
        df_test, tokenizer, Config.max_length, labels=None  # No labels for test
    )

    # 6. Create DataLoaders
    print("Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
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

    print("Data loading complete.")
    return train_loader, val_loader, test_loader
