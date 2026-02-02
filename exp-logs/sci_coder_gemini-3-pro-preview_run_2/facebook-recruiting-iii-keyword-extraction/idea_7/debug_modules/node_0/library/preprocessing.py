import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from library.config import Config


class TextCleaner:
    """
    Handles text cleaning: HTML stripping, lowercase, and basic normalization.
    """

    @staticmethod
    def clean(text):
        if pd.isna(text):
            return ""
        # Convert to string
        text = str(text)
        # Lowercase
        text = text.lower()
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Remove special characters but keep alphanumeric and spaces
        # We keep some punctuation that might be relevant for code like +, #, .
        # But for general text CNN, simpler is often better.
        # Let's keep it relatively simple: replace non-alphanumeric (except specific code chars) with space
        # Actually, for the Wide component (TFIDF), keeping punctuation can be useful (e.g., c++).
        # For Deep component, the tokenizer will handle splitting.
        # We will just normalize whitespace here.
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def batch_clean(series):
        return series.apply(TextCleaner.clean)


class CustomTokenizer:
    """
    Tokenizer for the Deep Component (CNN).
    Maps words to integers based on frequency.
    """

    def __init__(self, vocab_size=Config.VOCAB_SIZE, max_len=Config.MAX_LEN):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word_index = {}
        self.index_word = {}
        self.oov_token = 1
        self.pad_token = 0

    def fit(self, texts):
        """
        Builds vocabulary from a list/series of texts.
        """
        print("Building vocabulary for Deep Component...")
        counter = Counter()
        for text in texts:
            # Simple whitespace tokenization for speed
            tokens = text.split()
            counter.update(tokens)

        # Select top most common words
        # We reserve 0 for padding and 1 for OOV, so we take vocab_size - 2
        most_common = counter.most_common(self.vocab_size - 2)

        self.word_index = {word: i + 2 for i, (word, _) in enumerate(most_common)}
        self.index_word = {i: word for word, i in self.word_index.items()}
        print(f"Vocabulary built. Size: {len(self.word_index)}")

    def transform(self, texts):
        """
        Converts texts to padded integer sequences.
        Returns a dense numpy array.
        """
        num_samples = len(texts)
        sequences = np.full((num_samples, self.max_len), self.pad_token, dtype=np.int32)

        for idx, text in enumerate(texts):
            tokens = text.split()
            # Convert to indices
            int_seq = [self.word_index.get(t, self.oov_token) for t in tokens]

            # Truncate and Pad
            # We truncate from the end if too long (keep beginning)
            trunc = int_seq[: self.max_len]
            sequences[idx, : len(trunc)] = trunc

        return sequences

    def save(self, path):
        with open(path, "w") as f:
            json.dump(
                {
                    "word_index": self.word_index,
                    "max_len": self.max_len,
                    "vocab_size": self.vocab_size,
                },
                f,
            )

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
            self.word_index = data["word_index"]
            self.max_len = data["max_len"]
            self.vocab_size = data["vocab_size"]
            self.index_word = {i: word for word, i in self.word_index.items()}


class TagEncoder:
    """
    Encodes target tags into multi-hot binary vectors.
    """

    def __init__(self, num_tags=Config.NUM_TAGS):
        self.num_tags = num_tags
        self.tag_to_index = {}
        self.index_to_tag = {}

    def fit(self, tags_series):
        """
        Identifies top K tags.
        tags_series: pandas Series of space-delimited tag strings.
        """
        print("Fitting TagEncoder...")
        counter = Counter()
        for tags in tags_series:
            if pd.isna(tags):
                continue
            t_list = tags.split()
            counter.update(t_list)

        most_common = counter.most_common(self.num_tags)
        self.tag_to_index = {tag: i for i, (tag, _) in enumerate(most_common)}
        self.index_to_tag = {i: tag for tag, i in self.tag_to_index.items()}
        print(f"TagEncoder fitted. Top {len(self.tag_to_index)} tags selected.")

    def transform(self, tags_series):
        """
        Returns a sparse matrix (CSR) of shape (N, num_tags).
        """
        n_samples = len(tags_series)
        indptr = [0]
        indices = []
        data = []

        for tags in tags_series:
            if pd.isna(tags):
                indptr.append(len(indices))
                continue

            t_list = tags.split()
            row_indices = []
            for t in t_list:
                if t in self.tag_to_index:
                    row_indices.append(self.tag_to_index[t])

            # Remove duplicates within a row if any, though tags usually unique per post
            row_indices = list(set(row_indices))
            indices.extend(row_indices)
            data.extend([1] * len(row_indices))
            indptr.append(len(indices))

        return sparse.csr_matrix(
            (data, indices, indptr), shape=(n_samples, self.num_tags), dtype=np.float32
        )

    def inverse_transform(self, binary_matrix):
        """
        Converts binary matrix back to list of tag strings.
        binary_matrix: (N, num_tags) numpy array or sparse matrix.
        """
        if sparse.issparse(binary_matrix):
            binary_matrix = binary_matrix.toarray()

        result = []
        for row in binary_matrix:
            tags = [
                self.index_to_tag[i] for i, val in enumerate(row) if val >= 0.5
            ]  # Threshold usually handled outside
            result.append(" ".join(tags))
        return result

    def save(self, path):
        joblib.dump(
            {"tag_to_index": self.tag_to_index, "num_tags": self.num_tags}, path
        )

    def load(self, path):
        data = joblib.load(path)
        self.tag_to_index = data["tag_to_index"]
        self.num_tags = data["num_tags"]
        self.index_to_tag = {i: tag for tag, i in self.tag_to_index.items()}


class Preprocessor:
    """
    Main class to orchestrate data loading, cleaning, and feature extraction.
    """

    def __init__(self):
        self.tokenizer = CustomTokenizer()
        self.tfidf = TfidfVectorizer(
            max_features=Config.TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            dtype=np.float32,
            token_pattern=r"(?u)\b\w+\b",  # Simple pattern
        )
        self.tag_encoder = TagEncoder()

    def _get_paths(self, split):
        base_dir = Config.WORKING_DIR
        if split == "train":
            prefix = Config.TRAIN_PROCESSED_DATA
        elif split == "val":
            prefix = Config.VAL_PROCESSED_DATA
        elif split == "test":
            prefix = Config.TEST_PROCESSED_DATA
        else:
            raise ValueError("Unknown split")

        # Ensure base directory exists (handled by Config.setup but good to be safe)
        os.makedirs(base_dir, exist_ok=True)

        # Define file paths
        # Wide: Sparse NPZ
        # Deep: Dense NPY
        # Targets: Sparse NPZ
        return {
            "X_wide": f"{prefix}_X_wide.npz",
            "X_deep": f"{prefix}_X_deep.npy",
            "y": f"{prefix}_y.npz",
        }

    def load_data(self, split="train", load_cached_data=True):
        """
        Loads processed data if available and requested.
        Otherwise, processes from scratch and caches it.
        """
        paths = self._get_paths(split)

        # Check cache
        if load_cached_data:
            files_exist = os.path.exists(paths["X_wide"]) and os.path.exists(
                paths["X_deep"]
            )
            if split != "test":
                files_exist = files_exist and os.path.exists(paths["y"])

            if files_exist:
                print(f"Loading cached data for {split}...")
                X_wide = sparse.load_npz(paths["X_wide"])
                X_deep = np.load(paths["X_deep"])
                y = sparse.load_npz(paths["y"]) if split != "test" else None

                # Load processors if not already loaded (for inference consistency)
                if split != "train":
                    self._load_processors()

                return X_wide, X_deep, y

        # Process from scratch
        print(f"Processing data for {split} from scratch...")

        # 1. Load Raw Metadata
        if split == "train":
            df = pd.read_csv(Config.TRAIN_PATH)
        elif split == "val":
            df = pd.read_csv(Config.VAL_PATH)
        elif split == "test":
            df = pd.read_csv(Config.TEST_PATH)

        # 2. Text Cleaning
        print("Cleaning text...")
        # Combine Title and Body
        text_series = df["Title"].fillna("") + " " + df["Body"].fillna("")
        text_series = TextCleaner.batch_clean(text_series)

        # 3. Fit Processors (if train) or Load (if val/test)
        if split == "train":
            # Fit Tokenizer
            self.tokenizer.fit(text_series)
            self.tokenizer.save(Config.TOKENIZER_PATH)

            # Fit TF-IDF
            print("Fitting TF-IDF...")
            self.tfidf.fit(text_series)
            joblib.dump(self.tfidf, Config.TFIDF_VECTORIZER_PATH)

            # Fit Tag Encoder
            self.tag_encoder.fit(df["Tags"].astype(str))
            self.tag_encoder.save(Config.TAG_ENCODER_PATH)
        else:
            self._load_processors()

        # 4. Transform Features
        print("Transforming Deep features...")
        X_deep = self.tokenizer.transform(text_series)

        print("Transforming Wide features...")
        X_wide = self.tfidf.transform(text_series)

        # 5. Transform Targets
        y = None
        if split != "test":
            print("Transforming Targets...")
            y = self.tag_encoder.transform(df["Tags"].astype(str))

        # 6. Cache Data
        print(f"Caching data to {Config.WORKING_DIR}...")
        sparse.save_npz(paths["X_wide"], X_wide)
        np.save(paths["X_deep"], X_deep)
        if y is not None:
            sparse.save_npz(paths["y"], y)

        return X_wide, X_deep, y

    def _load_processors(self):
        """Helper to load fitted processors from disk."""
        if os.path.exists(Config.TOKENIZER_PATH):
            self.tokenizer.load(Config.TOKENIZER_PATH)
        else:
            print("Warning: Tokenizer not found on disk.")

        if os.path.exists(Config.TFIDF_VECTORIZER_PATH):
            self.tfidf = joblib.load(Config.TFIDF_VECTORIZER_PATH)
        else:
            print("Warning: TF-IDF Vectorizer not found on disk.")

        if os.path.exists(Config.TAG_ENCODER_PATH):
            self.tag_encoder.load(Config.TAG_ENCODER_PATH)
        else:
            print("Warning: Tag Encoder not found on disk.")

    def get_test_ids(self):
        """Returns the IDs for the test set."""
        df = pd.read_csv(Config.TEST_PATH)
        return df["Id"].values

    def inverse_transform_tags(self, binary_matrix):
        return self.tag_encoder.inverse_transform(binary_matrix)
