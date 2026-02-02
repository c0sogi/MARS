import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import seed_everything


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the manufacturing data.
    Stores preprocessed numerical and sequence features.
    """

    def __init__(self, X_num, X_seq, y=None, ids=None):
        self.X_num = torch.FloatTensor(X_num)
        self.X_seq = torch.LongTensor(X_seq)
        self.y = torch.FloatTensor(y) if y is not None else None
        self.ids = ids

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        sample = {
            "num_features": self.X_num[idx],
            "seq_features": self.X_seq[idx],
        }

        if self.y is not None:
            sample["target"] = self.y[idx]
        else:
            # Placeholder for test set (will be ignored by loss function)
            sample["target"] = torch.tensor(-1.0)

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        return sample


class DenoisingCollator:
    """
    Collator that implements On-the-Fly Masking for the Denoising Objective.
    Generates boolean masks for numerical and sequence inputs.
    """

    def __init__(self, mask_prob, vocab, mask_token_id):
        self.mask_prob = mask_prob
        self.vocab = vocab
        self.mask_token_id = mask_token_id

    def __call__(self, batch):
        # Stack inputs into tensors
        num_features = torch.stack([item["num_features"] for item in batch])
        seq_features = torch.stack([item["seq_features"] for item in batch])
        targets = torch.stack([item["target"] for item in batch])
        ids = [item.get("id") for item in batch]

        # Determine if we are in training mode (masking enabled)
        if self.mask_prob > 0:
            # 1. Numerical Masking
            # Create boolean mask: True means "mask this value"
            mask_num = torch.rand_like(num_features) < self.mask_prob

            # The model will use this boolean mask to replace the numerical embedding
            # with a learnable [MASK] embedding. We pass the original values as input
            # (or we could zero them, but the mask dictates the replacement).
            num_input = num_features.clone()

            # 2. Sequence Masking
            # Create boolean mask
            mask_seq = torch.rand(seq_features.shape) < self.mask_prob

            # Replace masked tokens with [MASK] token ID
            seq_input = seq_features.clone()
            seq_input[mask_seq] = self.mask_token_id

        else:
            # Inference/Validation mode: No masking
            mask_num = torch.zeros_like(num_features, dtype=torch.bool)
            mask_seq = torch.zeros(seq_features.shape, dtype=torch.bool)
            num_input = num_features
            seq_input = seq_features

        return {
            "num_features": num_input,  # Input to model (B, N_num)
            "seq_features": seq_input,  # Input to model (B, N_seq)
            "mask_num": mask_num,  # Boolean mask (B, N_num)
            "mask_seq": mask_seq,  # Boolean mask (B, N_seq)
            "target_num": num_features,  # Reconstruction Target (Original values)
            "target_seq": seq_features,  # Reconstruction Target (Original tokens)
            "target_cls": targets,  # Classification Target
            "ids": ids,
        }


class Preprocessor:
    """
    Handles feature extraction, tokenization, and standardization.
    """

    def __init__(self, cache_dir="./working/idea_8"):
        self.cache_dir = cache_dir
        self.scaler = StandardScaler()
        self.vocab = {"[PAD]": 0, "[MASK]": 1}
        self.num_cols = None

        os.makedirs(self.cache_dir, exist_ok=True)

    def _extract_features(self, df, is_train=True):
        # 1. Feature Engineering: Unique Characters Count
        # This captures the complexity of the sequence in f_27
        unique_chars = df["f_27"].apply(lambda x: len(set(x))).values.reshape(-1, 1)

        # 2. Identify Numerical Columns dynamically
        if self.num_cols is None:
            exclude = ["id", "target", "source_path", "f_27"]
            # Select all columns that are not excluded
            candidates = [c for c in df.columns if c not in exclude]
            # Further filter to ensure they are numeric
            self.num_cols = (
                df[candidates].select_dtypes(include=[np.number]).columns.tolist()
            )

        X_num_raw = df[self.num_cols].values

        # Append the engineered feature to the numerical matrix
        X_num = np.hstack([X_num_raw, unique_chars])

        # 3. Process Sequence (f_27)
        sequences = df["f_27"].values

        if is_train:
            # Build Vocabulary from training data
            all_chars = set()
            for s in sequences:
                all_chars.update(s)

            # Sort for determinism
            sorted_chars = sorted(list(all_chars))
            for char in sorted_chars:
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)

            # Save vocab for consistency
            with open(os.path.join(self.cache_dir, "vocab.json"), "w") as f:
                json.dump(self.vocab, f)

        # Tokenize sequences
        X_seq = []
        pad_token = self.vocab["[PAD]"]
        for s in sequences:
            tokens = [self.vocab.get(c, pad_token) for c in s]
            X_seq.append(tokens)

        X_seq = np.array(X_seq, dtype=np.int32)

        return X_num, X_seq

    def fit_transform(self, df_train):
        X_num, X_seq = self._extract_features(df_train, is_train=True)

        # Standardize numerical features
        self.scaler.fit(X_num)
        X_num_scaled = self.scaler.transform(X_num)

        return X_num_scaled, X_seq

    def transform(self, df):
        # Ensure vocab is loaded if we are just transforming (e.g. inference)
        vocab_path = os.path.join(self.cache_dir, "vocab.json")
        if len(self.vocab) == 2 and os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                self.vocab = json.load(f)

        X_num, X_seq = self._extract_features(df, is_train=False)
        X_num_scaled = self.scaler.transform(X_num)
        return X_num_scaled, X_seq


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, load_cached_data=True, debug=Config.DEBUG
):
    """
    Main interface to load data, preprocess (with caching), and return DataLoaders.
    """
    seed_everything(Config.SEED)
    cache_dir = "./working/idea_8"
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file names
    files = {
        "train": ["X_num_train.npy", "X_seq_train.npy", "y_train.npy", "ids_train.npy"],
        "val": ["X_num_val.npy", "X_seq_val.npy", "y_val.npy", "ids_val.npy"],
        "test": ["X_num_test.npy", "X_seq_test.npy", "ids_test.npy"],
    }

    # Check if all cache files exist
    cache_exists = True
    for split, flist in files.items():
        for f in flist:
            if not os.path.exists(os.path.join(cache_dir, f)):
                cache_exists = False
                break
    if not os.path.exists(os.path.join(cache_dir, "vocab.json")):
        cache_exists = False

    data = {}

    # Load from cache if available and requested
    if load_cached_data and cache_exists and not debug:
        print(f"Loading cached data from {cache_dir} ...")

        # Helper to load
        def load_npy(name):
            return np.load(os.path.join(cache_dir, name))

        data["X_num_train"] = load_npy("X_num_train.npy")
        data["X_seq_train"] = load_npy("X_seq_train.npy")
        data["y_train"] = load_npy("y_train.npy")
        data["ids_train"] = load_npy("ids_train.npy")

        data["X_num_val"] = load_npy("X_num_val.npy")
        data["X_seq_val"] = load_npy("X_seq_val.npy")
        data["y_val"] = load_npy("y_val.npy")
        data["ids_val"] = load_npy("ids_val.npy")

        data["X_num_test"] = load_npy("X_num_test.npy")
        data["X_seq_test"] = load_npy("X_seq_test.npy")
        data["ids_test"] = load_npy("ids_test.npy")

        with open(os.path.join(cache_dir, "vocab.json"), "r") as f:
            vocab = json.load(f)

    else:
        print("Processing data from scratch...")
        # Load Metadata CSVs
        df_train = pd.read_csv(Config.TRAIN_PATH)
        df_val = pd.read_csv(Config.VAL_PATH)
        df_test = pd.read_csv(Config.TEST_PATH)

        # Debug sampling
        if debug:
            print("Debug mode: Sampling data...")
            df_train = df_train.sample(n=5000, random_state=Config.SEED).reset_index(
                drop=True
            )
            df_val = df_val.sample(n=1000, random_state=Config.SEED).reset_index(
                drop=True
            )
            df_test = df_test.sample(n=1000, random_state=Config.SEED).reset_index(
                drop=True
            )

        preprocessor = Preprocessor(cache_dir=cache_dir)

        # Fit & Transform Train
        X_num_train, X_seq_train = preprocessor.fit_transform(df_train)
        y_train = df_train["target"].values.astype(np.float32)
        ids_train = df_train["id"].values

        # Transform Val
        X_num_val, X_seq_val = preprocessor.transform(df_val)
        y_val = df_val["target"].values.astype(np.float32)
        ids_val = df_val["id"].values

        # Transform Test
        X_num_test, X_seq_test = preprocessor.transform(df_test)
        ids_test = df_test["id"].values

        vocab = preprocessor.vocab

        # Save to cache (skip if debug to avoid corrupting cache)
        if not debug:
            print(f"Saving processed data to {cache_dir} ...")
            np.save(os.path.join(cache_dir, "X_num_train.npy"), X_num_train)
            np.save(os.path.join(cache_dir, "X_seq_train.npy"), X_seq_train)
            np.save(os.path.join(cache_dir, "y_train.npy"), y_train)
            np.save(os.path.join(cache_dir, "ids_train.npy"), ids_train)

            np.save(os.path.join(cache_dir, "X_num_val.npy"), X_num_val)
            np.save(os.path.join(cache_dir, "X_seq_val.npy"), X_seq_val)
            np.save(os.path.join(cache_dir, "y_val.npy"), y_val)
            np.save(os.path.join(cache_dir, "ids_val.npy"), ids_val)

            np.save(os.path.join(cache_dir, "X_num_test.npy"), X_num_test)
            np.save(os.path.join(cache_dir, "X_seq_test.npy"), X_seq_test)
            np.save(os.path.join(cache_dir, "ids_test.npy"), ids_test)

        data = {
            "X_num_train": X_num_train,
            "X_seq_train": X_seq_train,
            "y_train": y_train,
            "ids_train": ids_train,
            "X_num_val": X_num_val,
            "X_seq_val": X_seq_val,
            "y_val": y_val,
            "ids_val": ids_val,
            "X_num_test": X_num_test,
            "X_seq_test": X_seq_test,
            "ids_test": ids_test,
        }

    # Instantiate Datasets
    train_dataset = ManufacturingDataset(
        data["X_num_train"], data["X_seq_train"], data["y_train"], data["ids_train"]
    )
    val_dataset = ManufacturingDataset(
        data["X_num_val"], data["X_seq_val"], data["y_val"], data["ids_val"]
    )
    test_dataset = ManufacturingDataset(
        data["X_num_test"], data["X_seq_test"], None, data["ids_test"]
    )

    # Instantiate Collators
    # Train: Apply masking
    train_collator = DenoisingCollator(
        mask_prob=Config.MASK_PROB, vocab=vocab, mask_token_id=vocab["[MASK]"]
    )

    # Val/Test: No masking
    eval_collator = DenoisingCollator(
        mask_prob=0.0, vocab=vocab, mask_token_id=vocab["[MASK]"]
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=train_collator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=eval_collator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=eval_collator,
    )

    return train_loader, val_loader, test_loader, vocab
