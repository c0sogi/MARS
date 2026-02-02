import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import scipy.sparse
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import clean_text, set_seed


class TagEncoder:
    """
    Encodes and decodes tags for multi-label classification.
    Keeps top `max_tags` most frequent tags.
    """

    def __init__(self, max_tags=5000):
        self.max_tags = max_tags
        self.tag_to_idx = {}
        self.idx_to_tag = {}
        self.classes_ = []

    def fit(self, tags_series):
        """
        Fits the encoder on a series of space-delimited tag strings.
        """
        counts = Counter()
        for tags in tags_series:
            if isinstance(tags, str):
                counts.update(tags.split())

        # Select top N tags
        most_common = counts.most_common(self.max_tags)
        self.classes_ = [tag for tag, _ in most_common]
        self.tag_to_idx = {tag: i for i, tag in enumerate(self.classes_)}
        self.idx_to_tag = {i: tag for i, tag in enumerate(self.classes_)}
        print(
            f"TagEncoder fitted. Retained {len(self.classes_)} tags out of {len(counts)} unique tags."
        )

    def transform(self, tags_series):
        """
        Transforms tag strings into a sparse multi-hot binary matrix.
        """
        rows = []
        cols = []
        data = []

        for i, tags in enumerate(tags_series):
            if isinstance(tags, str):
                for tag in tags.split():
                    if tag in self.tag_to_idx:
                        rows.append(i)
                        cols.append(self.tag_to_idx[tag])
                        data.append(1)

        # Create CSR matrix
        matrix = scipy.sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(tags_series), len(self.classes_)),
            dtype=np.float32,
        )
        return matrix

    def inverse_transform(self, probs, threshold=0.5):
        """
        Converts probability matrix/tensor to list of space-delimited tag strings.
        """
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()

        preds = []
        for row in probs:
            indices = np.where(row > threshold)[0]
            tags = [self.idx_to_tag[idx] for idx in indices]
            preds.append(" ".join(tags))
        return preds

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"tag_to_idx": self.tag_to_idx, "classes_": self.classes_}, f)

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
            self.tag_to_idx = data["tag_to_idx"]
            self.classes_ = data["classes_"]
            self.idx_to_tag = {int(i): tag for tag, i in self.tag_to_idx.items()}


class TextPreprocessor:
    """
    Handles text processing for both Deep (Sequence) and Wide (TF-IDF) components.
    """

    def __init__(self, vocab_size=50000, max_len=200, tfidf_max_features=150000):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tfidf_max_features = tfidf_max_features

        # Deep component: Vocab mapping
        self.vocab = {"<PAD>": 0, "<UNK>": 1}

        # Wide component: TF-IDF Vectorizer
        self.tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=(1, 2),
            dtype=np.float32,
            sublinear_tf=True,
            min_df=2,
            token_pattern=r"\b\w+\b",
        )

    def _tokenize(self, text):
        # Simple regex tokenizer
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, texts):
        """
        Fits Tokenizer and TF-IDF on the provided texts.
        """
        print("Fitting Tokenizer (Deep Component)...")
        word_counts = Counter()
        # Using a simple loop; for 4M rows this is acceptable given the constraints
        for text in texts:
            word_counts.update(self._tokenize(text))

        # Keep top vocab
        most_common = word_counts.most_common(self.vocab_size - 2)
        for word, _ in most_common:
            self.vocab[word] = len(self.vocab)
        print(f"Tokenizer fitted. Vocab size: {len(self.vocab)}")

        print("Fitting TF-IDF (Wide Component)...")
        self.tfidf.fit(texts)
        print(f"TF-IDF fitted. Number of features: {len(self.tfidf.vocabulary_)}")

    def transform_deep(self, texts):
        """
        Converts texts to padded integer sequences.
        Returns: numpy array of shape (N, max_len)
        """
        print("Transforming Deep Features...")
        sequences = []
        for text in texts:
            tokens = self._tokenize(text)
            # Map to indices, use UNK (1) if missing
            seq = [self.vocab.get(t, 1) for t in tokens[: self.max_len]]

            # Pad with PAD (0)
            if len(seq) < self.max_len:
                seq += [0] * (self.max_len - len(seq))
            else:
                seq = seq[: self.max_len]
            sequences.append(seq)

        return np.array(sequences, dtype=np.int32)

    def transform_wide(self, texts):
        """
        Converts texts to sparse TF-IDF matrix.
        Returns: scipy.sparse.csr_matrix
        """
        print("Transforming Wide Features...")
        return self.tfidf.transform(texts)

    def save(self, vocab_path, tfidf_path):
        with open(vocab_path, "w") as f:
            json.dump(self.vocab, f)
        joblib.dump(self.tfidf, tfidf_path)

    def load(self, vocab_path, tfidf_path):
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
        self.tfidf = joblib.load(tfidf_path)


class StackExchangeDataset(Dataset):
    """
    PyTorch Dataset for Hybrid Wide-and-Deep model.
    """

    def __init__(self, deep_data, wide_data, labels=None):
        self.deep_data = deep_data  # numpy array (N, max_len)
        self.wide_data = wide_data  # sparse matrix (N, num_features)
        self.labels = labels  # sparse matrix (N, num_classes) or None

    def __len__(self):
        return len(self.deep_data)

    def __getitem__(self, idx):
        # Deep Input
        deep_item = torch.tensor(self.deep_data[idx], dtype=torch.long)

        # Wide Input
        # Convert specific sparse row to dense tensor on-the-fly
        # wide_data[idx] returns a 1xF CSR matrix
        wide_row = self.wide_data[idx].toarray().flatten()
        wide_item = torch.tensor(wide_row, dtype=torch.float)

        # Label
        if self.labels is not None:
            label_row = self.labels[idx].toarray().flatten()
            label_item = torch.tensor(label_row, dtype=torch.float)
            return {"deep": deep_item, "wide": wide_item, "label": label_item}
        else:
            return {"deep": deep_item, "wide": wide_item}


def prepare_data(load_cached_data=True, debug=False):
    """
    Orchestrates data loading, cleaning, processing, and saving.
    """
    set_seed(Config.SEED)

    # Check if cache exists
    required_files = [
        Config.TRAIN_WIDE_PATH,
        Config.TRAIN_DEEP_PATH,
        Config.TRAIN_LABELS_PATH,
        Config.VAL_WIDE_PATH,
        Config.VAL_DEEP_PATH,
        Config.VAL_LABELS_PATH,
        Config.TEST_WIDE_PATH,
        Config.TEST_DEEP_PATH,
        Config.TEST_IDS_PATH,
        Config.TAG_ENCODER_PATH,
        Config.VOCAB_PATH,
        Config.TFIDF_VECTORIZER_PATH,
    ]

    cache_exists = all(os.path.exists(f) for f in required_files)

    if load_cached_data and cache_exists:
        # Validate cached artifact dimensions before use (Cite debug_lesson_2)
        # Ensure cached test data matches the full test set size, even in debug mode
        cached_test_ids = np.load(Config.TEST_IDS_PATH)
        test_meta_len = len(pd.read_csv(Config.TEST_METADATA_PATH, usecols=["Id"]))

        if len(cached_test_ids) == test_meta_len:
            print("Loading cached data found. Skipping processing.")
            return
        else:
            print(
                f"Cached test size ({len(cached_test_ids)}) mismatch with metadata ({test_meta_len}). Reprocessing..."
            )

    print("Processing data from scratch...")

    # Load Metadata
    print("Loading metadata CSVs...")
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if debug:
        print(f"DEBUG MODE: Subsampling {Config.DEBUG_SIZE} rows for Train/Val.")
        train_df = train_df.head(Config.DEBUG_SIZE)
        val_df = val_df.head(Config.DEBUG_SIZE)
        # Test set must remain full size for valid submission

    # Preprocess Text (Title + Body)
    print("Cleaning and combining text...")

    # Helper to combine and clean
    def process_text_col(df):
        # Combine Title and Body
        raw_text = df["Title"].astype(str) + " " + df["Body"].astype(str)
        # Apply cleaning
        return raw_text.apply(clean_text)

    train_text = process_text_col(train_df)
    val_text = process_text_col(val_df)
    test_text = process_text_col(test_df)

    # 1. Tag Encoding
    print("Encoding Tags...")
    tag_encoder = TagEncoder(max_tags=Config.NUM_CLASSES)
    tag_encoder.fit(train_df["Tags"])

    train_labels = tag_encoder.transform(train_df["Tags"])
    val_labels = tag_encoder.transform(val_df["Tags"])

    # Save Tag Encoder
    tag_encoder.save(Config.TAG_ENCODER_PATH)

    # 2. Text Processing
    print("Processing Text Features...")
    preprocessor = TextPreprocessor(
        vocab_size=Config.VOCAB_SIZE,
        max_len=Config.MAX_LEN,
        tfidf_max_features=Config.TFIDF_MAX_FEATURES,
    )

    # Fit on Train
    preprocessor.fit(train_text)

    # Transform All
    train_deep = preprocessor.transform_deep(train_text)
    train_wide = preprocessor.transform_wide(train_text)

    val_deep = preprocessor.transform_deep(val_text)
    val_wide = preprocessor.transform_wide(val_text)

    test_deep = preprocessor.transform_deep(test_text)
    test_wide = preprocessor.transform_wide(test_text)

    # Save Preprocessor
    preprocessor.save(Config.VOCAB_PATH, Config.TFIDF_VECTORIZER_PATH)

    # 3. Save Processed Data
    print("Saving processed data to disk...")
    # Train
    np.save(Config.TRAIN_DEEP_PATH, train_deep)
    scipy.sparse.save_npz(Config.TRAIN_WIDE_PATH, train_wide)
    scipy.sparse.save_npz(Config.TRAIN_LABELS_PATH, train_labels)

    # Val
    np.save(Config.VAL_DEEP_PATH, val_deep)
    scipy.sparse.save_npz(Config.VAL_WIDE_PATH, val_wide)
    scipy.sparse.save_npz(Config.VAL_LABELS_PATH, val_labels)

    # Test
    np.save(Config.TEST_DEEP_PATH, test_deep)
    scipy.sparse.save_npz(Config.TEST_WIDE_PATH, test_wide)
    np.save(Config.TEST_IDS_PATH, test_df["Id"].values)

    print("Data processing complete.")


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=False
):
    """
    Loads processed data and returns PyTorch DataLoaders.
    """
    print("Loading processed data for DataLoaders...")

    # Load Train
    train_deep = np.load(Config.TRAIN_DEEP_PATH)
    train_wide = scipy.sparse.load_npz(Config.TRAIN_WIDE_PATH)
    train_labels = scipy.sparse.load_npz(Config.TRAIN_LABELS_PATH)

    # Load Val
    val_deep = np.load(Config.VAL_DEEP_PATH)
    val_wide = scipy.sparse.load_npz(Config.VAL_WIDE_PATH)
    val_labels = scipy.sparse.load_npz(Config.VAL_LABELS_PATH)

    # Load Test
    test_deep = np.load(Config.TEST_DEEP_PATH)
    test_wide = scipy.sparse.load_npz(Config.TEST_WIDE_PATH)
    # Test has no labels

    if debug:
        # Slice for debug
        limit = Config.DEBUG_SIZE
        train_deep = train_deep[:limit]
        train_wide = train_wide[:limit]
        train_labels = train_labels[:limit]
        val_deep = val_deep[:limit]
        val_wide = val_wide[:limit]
        val_labels = val_labels[:limit]
        # Do not slice test data; we need full predictions for submission

    print(
        f"Train Data Shapes: Deep {train_deep.shape}, Wide {train_wide.shape}, Labels {train_labels.shape}"
    )
    print(
        f"Val Data Shapes: Deep {val_deep.shape}, Wide {val_wide.shape}, Labels {val_labels.shape}"
    )

    # Create Datasets
    train_dataset = StackExchangeDataset(train_deep, train_wide, train_labels)
    val_dataset = StackExchangeDataset(val_deep, val_wide, val_labels)
    test_dataset = StackExchangeDataset(test_deep, test_wide, labels=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
