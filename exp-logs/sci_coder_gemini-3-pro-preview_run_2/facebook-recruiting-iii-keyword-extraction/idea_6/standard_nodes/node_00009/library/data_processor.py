import os
import re
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from library.config import (
    OUTPUT_DIR,
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    MAX_LEN_DEEP,
    VOCAB_SIZE_DEEP,
    VOCAB_SIZE_WIDE,
    NUM_TAGS,
    SEED,
)

# Compile regex patterns for cleaning
HTML_TAG_RE = re.compile(r"<[^>]+>")
NON_ALPHANUM_RE = re.compile(r"[^a-z0-9\s]")


def clean_text(text):
    """
    Removes HTML tags and non-alphanumeric characters from text.
    """
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove HTML
    text = HTML_TAG_RE.sub(" ", text)
    # Remove non-alphanumeric (keep spaces)
    text = NON_ALPHANUM_RE.sub(" ", text)
    # Collapse multiple spaces
    return " ".join(text.split())


class CustomTokenizer:
    """
    A simple frequency-based tokenizer for the Deep component.
    """

    def __init__(self, vocab_size, max_len):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word2idx = {}
        self.pad_token = 0
        self.unk_token = 1

    def fit(self, texts):
        """
        Builds vocabulary from a list/series of text strings.
        """
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        # Keep top (vocab_size - 2) words to reserve spots for PAD and UNK
        most_common = counter.most_common(self.vocab_size - 2)

        self.word2idx = {"<PAD>": self.pad_token, "<UNK>": self.unk_token}

        for i, (word, _) in enumerate(most_common):
            self.word2idx[word] = i + 2

    def transform(self, texts):
        """
        Converts texts to padded integer sequences.
        Returns a numpy array of shape (len(texts), max_len).
        """
        sequences = []
        for text in texts:
            words = text.split()
            # Truncate if too long
            if len(words) > self.max_len:
                words = words[: self.max_len]

            # Map to indices
            seq = [self.word2idx.get(w, self.unk_token) for w in words]

            # Pad if too short
            if len(seq) < self.max_len:
                seq = seq + [self.pad_token] * (self.max_len - len(seq))

            sequences.append(seq)

        return np.array(sequences, dtype=np.int32)


class TextPreprocessor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=VOCAB_SIZE_WIDE,
            stop_words="english",
            dtype=np.float32,
        )
        self.tokenizer = CustomTokenizer(
            vocab_size=VOCAB_SIZE_DEEP, max_len=MAX_LEN_DEEP
        )
        self.mlb = MultiLabelBinarizer(sparse_output=True)
        self.top_tags = None

    def fit(self, train_df):
        """
        Fits all vectorizers and encoders on the training data.
        """
        print("Preprocessing text...")
        # Combine Title and Body for text features
        text_data = (
            train_df["Title"].fillna("") + " " + train_df["Body"].fillna("")
        ).apply(clean_text)

        print(f"Fitting TF-IDF (Wide) on {len(text_data)} samples...")
        self.tfidf.fit(text_data)

        print(f"Fitting Tokenizer (Deep) on {len(text_data)} samples...")
        self.tokenizer.fit(text_data)

        print("Processing Tags...")
        # Handle Tags
        # Split tags into lists
        all_tags_lists = train_df["Tags"].fillna("").str.split()

        # Count tag frequencies to find top NUM_TAGS
        tag_counts = Counter()
        for tags in all_tags_lists:
            tag_counts.update(tags)

        self.top_tags = set([t for t, c in tag_counts.most_common(NUM_TAGS)])
        print(
            f"Selected {len(self.top_tags)} top tags from {len(tag_counts)} unique tags."
        )

        # Filter tags in the training set to only include top tags
        filtered_tags = all_tags_lists.apply(
            lambda tags: [t for t in tags if t in self.top_tags]
        )

        print("Fitting MultiLabelBinarizer...")
        self.mlb.fit(filtered_tags)

    def transform(self, df, is_test=False):
        """
        Transforms dataframe into features and targets.
        """
        print("Cleaning text for transformation...")
        text_data = (df["Title"].fillna("") + " " + df["Body"].fillna("")).apply(
            clean_text
        )

        print("Generating Wide features (TF-IDF)...")
        X_wide = self.tfidf.transform(text_data)

        print("Generating Deep features (Sequences)...")
        X_deep = self.tokenizer.transform(text_data)

        y = None
        if not is_test and "Tags" in df.columns:
            print("Generating Targets...")
            # Filter tags
            tags_lists = df["Tags"].fillna("").str.split()
            filtered_tags = tags_lists.apply(
                lambda tags: (
                    [t for t in tags if t in self.top_tags]
                    if isinstance(tags, list)
                    else []
                )
            )
            y = self.mlb.transform(filtered_tags)

        return X_wide, X_deep, y


def prepare_data(load_cached_data=True):
    """
    Main function to load, process, and cache data.
    """
    # Define cache file paths
    cache_files = {
        "X_wide_train": os.path.join(OUTPUT_DIR, "X_wide_train.npz"),
        "X_deep_train": os.path.join(OUTPUT_DIR, "X_deep_train.npy"),
        "y_train": os.path.join(OUTPUT_DIR, "y_train.npz"),
        "X_wide_val": os.path.join(OUTPUT_DIR, "X_wide_val.npz"),
        "X_deep_val": os.path.join(OUTPUT_DIR, "X_deep_val.npy"),
        "y_val": os.path.join(OUTPUT_DIR, "y_val.npz"),
        "X_wide_test": os.path.join(OUTPUT_DIR, "X_wide_test.npz"),
        "X_deep_test": os.path.join(OUTPUT_DIR, "X_deep_test.npy"),
        "test_ids": os.path.join(OUTPUT_DIR, "test_ids.npy"),
        "preprocessor": os.path.join(OUTPUT_DIR, "preprocessor.pkl"),
    }

    # Check if we can load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in cache_files.values())
        if all_exist:
            print("Loading cached data from", OUTPUT_DIR)
            X_wide_train = sparse.load_npz(cache_files["X_wide_train"])
            X_deep_train = np.load(cache_files["X_deep_train"])
            y_train = sparse.load_npz(cache_files["y_train"])

            X_wide_val = sparse.load_npz(cache_files["X_wide_val"])
            X_deep_val = np.load(cache_files["X_deep_val"])
            y_val = sparse.load_npz(cache_files["y_val"])

            X_wide_test = sparse.load_npz(cache_files["X_wide_test"])
            X_deep_test = np.load(cache_files["X_deep_test"])
            test_ids = np.load(cache_files["test_ids"])

            preprocessor = joblib.load(cache_files["preprocessor"])

            return (
                X_wide_train,
                X_deep_train,
                y_train,
                X_wide_val,
                X_deep_val,
                y_val,
                X_wide_test,
                X_deep_test,
                test_ids,
                preprocessor,
            )
        else:
            print("Cache incomplete or missing. Reprocessing data...")

    # Load raw data
    print("Loading metadata CSVs...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Initialize and fit preprocessor
    preprocessor = TextPreprocessor()
    preprocessor.fit(train_df)

    # Transform datasets
    print("Transforming Train set...")
    X_wide_train, X_deep_train, y_train = preprocessor.transform(train_df)

    print("Transforming Validation set...")
    X_wide_val, X_deep_val, y_val = preprocessor.transform(val_df)

    print("Transforming Test set...")
    X_wide_test, X_deep_test, _ = preprocessor.transform(test_df, is_test=True)

    test_ids = test_df["Id"].values

    # Save to cache
    print("Saving processed data to cache...")
    sparse.save_npz(cache_files["X_wide_train"], X_wide_train)
    np.save(cache_files["X_deep_train"], X_deep_train)
    sparse.save_npz(cache_files["y_train"], y_train)

    sparse.save_npz(cache_files["X_wide_val"], X_wide_val)
    np.save(cache_files["X_deep_val"], X_deep_val)
    sparse.save_npz(cache_files["y_val"], y_val)

    sparse.save_npz(cache_files["X_wide_test"], X_wide_test)
    np.save(cache_files["X_deep_test"], X_deep_test)
    np.save(cache_files["test_ids"], test_ids)

    joblib.dump(preprocessor, cache_files["preprocessor"])

    return (
        X_wide_train,
        X_deep_train,
        y_train,
        X_wide_val,
        X_deep_val,
        y_val,
        X_wide_test,
        X_deep_test,
        test_ids,
        preprocessor,
    )
