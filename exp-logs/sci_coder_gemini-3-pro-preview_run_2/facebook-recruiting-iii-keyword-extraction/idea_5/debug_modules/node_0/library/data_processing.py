import os
import json
import numpy as np
import pandas as pd
import scipy.sparse
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

from library.config import Config
from library.utils import clean_text, Timer, set_seed


class TextVectorizer:
    """
    Wrapper for TF-IDF Vectorization.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=Config.VOCAB_SIZE,
            ngram_range=Config.NGRAM_RANGE,
            min_df=Config.MIN_DF,
            max_df=Config.MAX_DF,
            stop_words="english",
            sublinear_tf=True,
            dtype=np.float32,
        )

    def fit(self, texts):
        print("Fitting TextVectorizer...")
        self.vectorizer.fit(texts)

    def transform(self, texts):
        print("Transforming text to TF-IDF features...")
        return self.vectorizer.transform(texts)


class TagEncoder:
    """
    Wrapper for Multi-Label Binarization with Top-K filtering.
    """

    def __init__(self):
        self.classes_ = None
        self.mlb = None

    def fit(self, tags_series):
        """
        Identify top K tags and fit the binarizer.
        """
        print(f"Fitting TagEncoder on top {Config.NUM_TAGS} tags...")

        # Count all tags
        all_tags = [tag for row in tags_series for tag in row.split()]
        counts = Counter(all_tags)

        # Select top K
        top_tags = [tag for tag, _ in counts.most_common(Config.NUM_TAGS)]
        self.classes_ = np.array(top_tags)

        # Initialize MLB with fixed classes
        self.mlb = MultiLabelBinarizer(classes=self.classes_, sparse_output=True)
        # We call fit on a dummy list just to initialize internal structures if needed,
        # but passing classes in __init__ is usually sufficient for transform.
        self.mlb.fit([[]])

    def transform(self, tags_series):
        """
        Convert space-delimited tag strings to sparse binary matrix.
        Filters out tags not in the top K.
        """
        print("Transforming tags to multi-hot vectors...")
        # Split strings into lists
        tags_list = [row.split() for row in tags_series]

        # Transform
        # MLB ignores unknown classes if they are not in `classes` provided in __init__?
        # Actually MLB behavior with fixed classes: it strictly adheres to them.
        return self.mlb.transform(tags_list)

    def inverse_transform(self, binary_matrix):
        """
        Convert binary matrix back to tag strings.
        """
        if scipy.sparse.issparse(binary_matrix):
            binary_matrix = binary_matrix.toarray()
        return self.mlb.inverse_transform(binary_matrix)


class SparseDataset(Dataset):
    """
    Custom Dataset for Sparse Matrices (Scipy CSR).
    Densifies data on-the-fly to save memory.
    """

    def __init__(self, features, targets=None):
        self.features = features
        self.targets = targets
        self.n_samples = features.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Convert sparse row to dense tensor
        # features is CSR, so indexing returns a 1xDim CSR matrix
        x = torch.tensor(self.features[idx].toarray(), dtype=torch.float32).squeeze(0)

        if self.targets is not None:
            # targets is also CSR
            y = torch.tensor(self.targets[idx].toarray(), dtype=torch.float32).squeeze(
                0
            )
            return x, y
        else:
            # Return dummy target for test set
            return x, torch.zeros(Config.OUTPUT_DIM, dtype=torch.float32)


def prepare_loaders(load_cached_data=True):
    """
    Main function to prepare DataLoaders.
    Handles caching of processed features to avoid re-computation.
    """
    set_seed(Config.SEED)

    # Define cache paths
    cache_dir = Config.WORKING_DIR
    paths = {
        "X_train": os.path.join(cache_dir, "X_train.npz"),
        "y_train": os.path.join(cache_dir, "y_train.npz"),
        "X_val": os.path.join(cache_dir, "X_val.npz"),
        "y_val": os.path.join(cache_dir, "y_val.npz"),
        "X_test": os.path.join(cache_dir, "X_test.npz"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
        "tag_classes": os.path.join(cache_dir, "tag_classes.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in paths.values())

    if load_cached_data and cache_exists:
        print("Loading cached data from disk...")
        with Timer("Load Cache"):
            X_train = scipy.sparse.load_npz(paths["X_train"])
            y_train = scipy.sparse.load_npz(paths["y_train"])
            X_val = scipy.sparse.load_npz(paths["X_val"])
            y_val = scipy.sparse.load_npz(paths["y_val"])
            X_test = scipy.sparse.load_npz(paths["X_test"])
            test_ids = np.load(paths["test_ids"])
            tag_classes = np.load(paths["tag_classes"])

            # Reconstruct TagEncoder
            encoder = TagEncoder()
            encoder.classes_ = tag_classes
            encoder.mlb = MultiLabelBinarizer(classes=tag_classes, sparse_output=True)
            encoder.mlb.fit([[]])

    else:
        print("Processing data from scratch...")

        # Load Raw Data
        with Timer("Load CSVs"):
            train_df = pd.read_csv(Config.TRAIN_PATH, engine="c")
            val_df = pd.read_csv(Config.VAL_PATH, engine="c")
            test_df = pd.read_csv(Config.TEST_PATH, engine="c")

            if Config.DEBUG:
                print(f"Debug Mode: Subsampling {Config.DEBUG_SIZE} rows.")
                train_df = train_df.iloc[: Config.DEBUG_SIZE]
                val_df = val_df.iloc[: min(len(val_df), Config.DEBUG_SIZE)]
                test_df = test_df.iloc[: min(len(test_df), Config.DEBUG_SIZE)]

        # Text Preprocessing
        with Timer("Text Cleaning"):
            # Combine Title and Body
            # We use a simple lambda for cleaning to avoid overhead of pandas apply if possible,
            # but apply is necessary for regex.
            print("Cleaning Train...")
            train_text = (train_df["Title"] + " " + train_df["Body"]).apply(clean_text)
            print("Cleaning Val...")
            val_text = (val_df["Title"] + " " + val_df["Body"]).apply(clean_text)
            print("Cleaning Test...")
            test_text = (test_df["Title"] + " " + test_df["Body"]).apply(clean_text)

        # Vectorization
        with Timer("Vectorization"):
            vectorizer = TextVectorizer()
            vectorizer.fit(train_text)
            X_train = vectorizer.transform(train_text)
            X_val = vectorizer.transform(val_text)
            X_test = vectorizer.transform(test_text)

        # Target Encoding
        with Timer("Target Encoding"):
            encoder = TagEncoder()
            encoder.fit(train_df["Tags"].astype(str))
            y_train = encoder.transform(train_df["Tags"].astype(str))
            y_val = encoder.transform(val_df["Tags"].astype(str))

            tag_classes = encoder.classes_
            test_ids = test_df["Id"].values

        # Save to Cache
        with Timer("Save Cache"):
            scipy.sparse.save_npz(paths["X_train"], X_train)
            scipy.sparse.save_npz(paths["y_train"], y_train)
            scipy.sparse.save_npz(paths["X_val"], X_val)
            scipy.sparse.save_npz(paths["y_val"], y_val)
            scipy.sparse.save_npz(paths["X_test"], X_test)
            np.save(paths["test_ids"], test_ids)
            np.save(paths["tag_classes"], tag_classes)

        # Clean up memory
        del train_df, val_df, test_df, train_text, val_text, test_text

    # Create Datasets
    print("Creating Datasets...")
    train_dataset = SparseDataset(X_train, y_train)
    val_dataset = SparseDataset(X_val, y_val)
    test_dataset = SparseDataset(X_test, targets=None)

    # Create DataLoaders
    # num_workers > 0 with sparse matrices can be tricky/slow due to pickling overhead.
    # We'll use num_workers=Config.NUM_WORKERS but if it hangs, 0 is safer.
    # Given the simple __getitem__, 4 workers should be fine if RAM allows.
    print("Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, encoder, test_ids
