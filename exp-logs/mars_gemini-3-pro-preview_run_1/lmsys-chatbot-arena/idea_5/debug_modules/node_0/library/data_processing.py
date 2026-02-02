import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything


class FeatureEngineer:
    """
    Computes strictly relative scalar features between two responses.
    """

    def __init__(self):
        self.epsilon = 1e-9

    def extract_features(self, df):
        """
        Computes differences and ratios for char count, word count, and newline count.
        Returns a numpy array of shape (N, 6).
        """
        # Ensure string type
        resp_a = df["response_a"].fillna("").astype(str)
        resp_b = df["response_b"].fillna("").astype(str)

        # 1. Character Counts
        len_char_a = resp_a.apply(len).values
        len_char_b = resp_b.apply(len).values

        # 2. Word Counts
        len_word_a = resp_a.apply(lambda x: len(x.split())).values
        len_word_b = resp_b.apply(lambda x: len(x.split())).values

        # 3. Newline Counts
        len_nl_a = resp_a.apply(lambda x: x.count("\n")).values
        len_nl_b = resp_b.apply(lambda x: x.count("\n")).values

        # Compute Relative Features
        features = []

        # Char features
        features.append(len_char_a - len_char_b)  # Diff
        features.append(len_char_a / (len_char_b + self.epsilon))  # Ratio

        # Word features
        features.append(len_word_a - len_word_b)
        features.append(len_word_a / (len_word_b + self.epsilon))

        # Newline features
        features.append(len_nl_a - len_nl_b)
        features.append(len_nl_a / (len_nl_b + self.epsilon))

        # Stack: Shape (6, N) -> Transpose to (N, 6)
        return np.stack(features, axis=1).astype(np.float32)


class ChatbotDataset(Dataset):
    """
    PyTorch Dataset for Siamese DeBERTa model.
    """

    def __init__(
        self,
        input_ids_a,
        attention_mask_a,
        input_ids_b,
        attention_mask_b,
        scalar_features,
        targets=None,
    ):
        self.input_ids_a = input_ids_a
        self.attention_mask_a = attention_mask_a
        self.input_ids_b = input_ids_b
        self.attention_mask_b = attention_mask_b
        self.scalar_features = scalar_features
        self.targets = targets

    def __len__(self):
        return len(self.input_ids_a)

    def __getitem__(self, idx):
        item = {
            "input_ids_a": torch.tensor(self.input_ids_a[idx], dtype=torch.long),
            "attention_mask_a": torch.tensor(
                self.attention_mask_a[idx], dtype=torch.long
            ),
            "input_ids_b": torch.tensor(self.input_ids_b[idx], dtype=torch.long),
            "attention_mask_b": torch.tensor(
                self.attention_mask_b[idx], dtype=torch.long
            ),
            "scalar_features": torch.tensor(
                self.scalar_features[idx], dtype=torch.float
            ),
        }

        if self.targets is not None:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float)

        return item


def prepare_arrays(df, tokenizer, split_name, load_cached_data=True):
    """
    Processes dataframe into numpy arrays for the dataset, with caching.
    """
    # Define cache filenames
    cache_files = {
        "ids_a": os.path.join(Config.cache_dir, f"{split_name}_ids_a.npy"),
        "mask_a": os.path.join(Config.cache_dir, f"{split_name}_mask_a.npy"),
        "ids_b": os.path.join(Config.cache_dir, f"{split_name}_ids_b.npy"),
        "mask_b": os.path.join(Config.cache_dir, f"{split_name}_mask_b.npy"),
        "features": os.path.join(Config.cache_dir, f"{split_name}_features.npy"),
        "targets": os.path.join(Config.cache_dir, f"{split_name}_targets.npy"),
    }

    has_targets = "winner_model_a" in df.columns

    # Check if cache exists
    cache_exists = all(
        os.path.exists(f)
        for k, f in cache_files.items()
        if (k != "targets" or has_targets)
    )

    if load_cached_data and cache_exists:
        print(f"Loading cached data for {split_name}...")
        ids_a = np.load(cache_files["ids_a"])
        mask_a = np.load(cache_files["mask_a"])
        ids_b = np.load(cache_files["ids_b"])
        mask_b = np.load(cache_files["mask_b"])
        features = np.load(cache_files["features"])
        targets = np.load(cache_files["targets"]) if has_targets else None
        return ids_a, mask_a, ids_b, mask_b, features, targets

    print(f"Processing data for {split_name}...")

    # 1. Feature Engineering
    fe = FeatureEngineer()
    features = fe.extract_features(df)

    # 2. Tokenization
    # We tokenize (Prompt, Response A) and (Prompt, Response B)
    prompts = df["prompt"].fillna("").astype(str).tolist()
    resps_a = df["response_a"].fillna("").astype(str).tolist()
    resps_b = df["response_b"].fillna("").astype(str).tolist()

    # Tokenize A
    enc_a = tokenizer(
        prompts,
        resps_a,
        truncation=True,
        max_length=Config.max_length,
        padding="max_length",
        return_tensors="np",
        return_token_type_ids=False,
    )
    ids_a = enc_a["input_ids"]
    mask_a = enc_a["attention_mask"]

    # Tokenize B
    enc_b = tokenizer(
        prompts,
        resps_b,
        truncation=True,
        max_length=Config.max_length,
        padding="max_length",
        return_tensors="np",
        return_token_type_ids=False,
    )
    ids_b = enc_b["input_ids"]
    mask_b = enc_b["attention_mask"]

    # 3. Targets
    targets = None
    if has_targets:
        # Columns: winner_model_a, winner_model_b, winner_tie
        # We keep them as probabilities (float)
        target_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
        targets = df[target_cols].values.astype(np.float32)

    # 4. Save to Cache
    np.save(cache_files["ids_a"], ids_a)
    np.save(cache_files["mask_a"], mask_a)
    np.save(cache_files["ids_b"], ids_b)
    np.save(cache_files["mask_b"], mask_b)
    np.save(cache_files["features"], features)
    if targets is not None:
        np.save(cache_files["targets"], targets)

    return ids_a, mask_a, ids_b, mask_b, features, targets


def get_dataloaders(load_cached_data=True):
    """
    Main function to load data, process it, and return DataLoaders.
    """
    seed_everything(Config.seed)

    # Ensure directories exist
    Config.setup()

    # Load Metadata
    train_df = pd.read_csv(Config.train_path)
    val_df = pd.read_csv(Config.val_path)
    test_df = pd.read_csv(Config.test_path)

    # Debug Mode: Subsample
    if Config.debug:
        print("DEBUG MODE: Subsampling data...")
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        test_df = test_df.head(50)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Process Data
    # Train
    t_ids_a, t_mask_a, t_ids_b, t_mask_b, t_feats, t_targets = prepare_arrays(
        train_df,
        tokenizer,
        "train" if not Config.debug else "train_debug",
        load_cached_data,
    )
    # Val
    v_ids_a, v_mask_a, v_ids_b, v_mask_b, v_feats, v_targets = prepare_arrays(
        val_df, tokenizer, "val" if not Config.debug else "val_debug", load_cached_data
    )
    # Test
    te_ids_a, te_mask_a, te_ids_b, te_mask_b, te_feats, _ = prepare_arrays(
        test_df,
        tokenizer,
        "test" if not Config.debug else "test_debug",
        load_cached_data,
    )

    # Create Datasets
    train_dataset = ChatbotDataset(
        t_ids_a, t_mask_a, t_ids_b, t_mask_b, t_feats, t_targets
    )
    val_dataset = ChatbotDataset(
        v_ids_a, v_mask_a, v_ids_b, v_mask_b, v_feats, v_targets
    )
    test_dataset = ChatbotDataset(
        te_ids_a, te_mask_a, te_ids_b, te_mask_b, te_feats, None
    )

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
