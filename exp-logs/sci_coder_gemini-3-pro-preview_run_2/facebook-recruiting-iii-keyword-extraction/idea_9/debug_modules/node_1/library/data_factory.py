import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import scipy.sparse
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from library.config import Config
from library.utils import set_seed

# Set seed for reproducibility
set_seed(Config.SEED)


class TextPreprocessor:
    """
    Handles cleaning and normalization of Stack Exchange text data.
    """

    @staticmethod
    def clean_text(text):
        """
        Removes HTML tags and normalizes whitespace/casing.
        """
        if not isinstance(text, str):
            return ""

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Lowercase
        text = text.lower()

        return text

    @staticmethod
    def preprocess_dataframe(df):
        """
        Applies cleaning to Title and Body columns of a DataFrame.
        Returns a Series of combined Title + Body text.
        """
        # Ensure string type
        title = df["Title"].astype(str).fillna("")
        body = df["Body"].astype(str).fillna("")

        # Clean
        clean_title = title.apply(TextPreprocessor.clean_text)
        clean_body = body.apply(TextPreprocessor.clean_text)

        # Combine
        return clean_title + " " + clean_body


class TagEncoder:
    """
    Encodes and decodes tags using a Multi-Hot representation.
    Focuses on the top N most frequent tags.
    """

    def __init__(self, max_features=Config.NUM_TOP_TAGS):
        self.max_features = max_features
        self.tag_to_idx = {}
        self.idx_to_tag = {}
        self.classes_ = []

    def fit(self, tags_series):
        """
        Fits the encoder on a pandas Series of space-delimited tags.
        Selects top max_features tags.
        """
        print("Fitting TagEncoder...")
        tag_counts = Counter()

        # Iterate to count tags (memory efficient)
        for tags in tags_series:
            if not isinstance(tags, str):
                continue
            tag_counts.update(tags.split())

        # Select top tags
        most_common = tag_counts.most_common(self.max_features)
        self.classes_ = [tag for tag, count in most_common]

        # Build mappings
        self.tag_to_idx = {tag: i for i, tag in enumerate(self.classes_)}
        self.idx_to_tag = {i: tag for i, tag in enumerate(self.classes_)}

        print(f"TagEncoder fitted. Vocab size: {len(self.classes_)}")
        return self

    def transform(self, tags_series):
        """
        Transforms a Series of tags into a sparse binary matrix (CSR).
        """
        n_samples = len(tags_series)
        n_classes = len(self.classes_)

        rows = []
        cols = []

        for i, tags in enumerate(tags_series):
            if not isinstance(tags, str):
                continue

            for tag in tags.split():
                if tag in self.tag_to_idx:
                    rows.append(i)
                    cols.append(self.tag_to_idx[tag])

        data = np.ones(len(rows), dtype=np.int8)
        matrix = scipy.sparse.csr_matrix(
            (data, (rows, cols)), shape=(n_samples, n_classes)
        )
        return matrix

    def inverse_transform(self, binary_matrix):
        """
        Converts a binary matrix (or probabilities) back to a list of tag strings.
        For probability matrices, assumes thresholding has already been applied (values are 0 or 1).
        """
        # Ensure we have a dense array or sparse matrix
        if scipy.sparse.issparse(binary_matrix):
            binary_matrix = binary_matrix.toarray()

        result = []
        for row in binary_matrix:
            # Get indices where value is 1 (or > 0)
            indices = np.where(row > 0)[0]
            tags = [self.idx_to_tag[idx] for idx in indices]
            result.append(" ".join(tags))

        return result

    def save(self, filepath):
        """Saves the vocabulary to a JSON file."""
        with open(filepath, "w") as f:
            json.dump({"tag_to_idx": self.tag_to_idx, "classes": self.classes_}, f)
        print(f"TagEncoder saved to {filepath}")

    def load(self, filepath):
        """Loads the vocabulary from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
            self.tag_to_idx = data["tag_to_idx"]
            self.classes_ = data["classes"]
            self.idx_to_tag = {i: tag for i, tag in enumerate(self.classes_)}
        print(f"TagEncoder loaded from {filepath}")
        return self


class TfidfFeaturizer:
    """
    Wraps TfidfVectorizer to handle fitting, transforming, and caching.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=Config.TFIDF_MAX_FEATURES,
            ngram_range=Config.TFIDF_NGRAM_RANGE,
            stop_words="english",
            dtype=np.float32,
            sublinear_tf=True,
        )

    def fit(self, text_series):
        print("Fitting TfidfVectorizer on full corpus...")
        self.vectorizer.fit(text_series)
        print(
            f"TfidfVectorizer fitted. Features: {len(self.vectorizer.get_feature_names_out())}"
        )
        return self

    def transform(self, text_series):
        print("Transforming text to TF-IDF features...")
        return self.vectorizer.transform(text_series)

    def save(self, filepath):
        joblib.dump(self.vectorizer, filepath)
        print(f"TfidfVectorizer saved to {filepath}")

    def load(self, filepath):
        self.vectorizer = joblib.load(filepath)
        print(f"TfidfVectorizer loaded from {filepath}")
        return self


class DataFactory:
    """
    Orchestrates data loading, preprocessing, and caching for Wide and Deep models.
    """

    def __init__(self):
        self.tag_encoder = TagEncoder()
        self.tfidf_featurizer = TfidfFeaturizer()

    def _load_raw_df(self, split):
        """Helper to load raw data from metadata CSVs."""
        if split == "train":
            path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            path = Config.VAL_METADATA_PATH
        elif split == "test":
            path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        print(f"Loading raw {split} data from {path}...")
        return pd.read_csv(path, engine="c")

    def get_wide_features(self, split="train", load_cached_data=True):
        """
        Returns TF-IDF features (sparse matrix) for the specified split.
        Handles caching via .npz files.
        """
        # Determine cache path
        if split == "train":
            cache_path = Config.WIDE_TRAIN_FEATURES_PATH
        elif split == "val":
            cache_path = Config.WIDE_VAL_FEATURES_PATH
        elif split == "test":
            cache_path = Config.WIDE_TEST_FEATURES_PATH
        else:
            raise ValueError("Invalid split")

        # Try loading cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached Wide features for {split} from {cache_path}...")
            return scipy.sparse.load_npz(cache_path)

        # If not cached, compute
        print(f"Computing Wide features for {split}...")

        # Load raw data
        df = self._load_raw_df(split)
        text_data = TextPreprocessor.preprocess_dataframe(df)

        # Handle Vectorizer
        if split == "train":
            # Fit vectorizer on train
            self.tfidf_featurizer.fit(text_data)
            self.tfidf_featurizer.save(Config.TFIDF_VECTORIZER_PATH)
        else:
            # Load vectorizer for val/test if not already loaded
            if not hasattr(self.tfidf_featurizer.vectorizer, "vocabulary_"):
                if os.path.exists(Config.TFIDF_VECTORIZER_PATH):
                    self.tfidf_featurizer.load(Config.TFIDF_VECTORIZER_PATH)
                else:
                    raise FileNotFoundError(
                        "TfidfVectorizer not found. Fit on train first."
                    )

        # Transform
        features = self.tfidf_featurizer.transform(text_data)

        # Save cache
        print(f"Saving Wide features for {split} to {cache_path}...")
        scipy.sparse.save_npz(cache_path, features)

        return features

    def get_targets(self, split="train", load_cached_data=True):
        """
        Returns Multi-Hot encoded targets (sparse matrix) for train/val.
        Handles caching via .npz files.
        """
        if split == "test":
            raise ValueError("Test set has no targets.")

        if split == "train":
            cache_path = Config.TARGETS_TRAIN_PATH
        else:
            cache_path = Config.TARGETS_VAL_PATH

        # Try loading cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached targets for {split} from {cache_path}...")
            # Load encoder if needed
            if split == "val" and not self.tag_encoder.tag_to_idx:
                if os.path.exists(Config.TAG_ENCODER_PATH):
                    self.tag_encoder.load(Config.TAG_ENCODER_PATH)
            return scipy.sparse.load_npz(cache_path)

        # Compute
        print(f"Computing targets for {split}...")
        df = self._load_raw_df(split)
        tags = df["Tags"].astype(str).fillna("")

        if split == "train":
            self.tag_encoder.fit(tags)
            self.tag_encoder.save(Config.TAG_ENCODER_PATH)
        else:
            if not self.tag_encoder.tag_to_idx:
                if os.path.exists(Config.TAG_ENCODER_PATH):
                    self.tag_encoder.load(Config.TAG_ENCODER_PATH)
                else:
                    raise FileNotFoundError("TagEncoder not found. Fit on train first.")

        targets = self.tag_encoder.transform(tags)

        print(f"Saving targets for {split} to {cache_path}...")
        scipy.sparse.save_npz(cache_path, targets)

        return targets

    def get_deep_data(self, split="train", load_cached_data=True):
        """
        Returns a DataFrame for the Deep Model.
        For 'train', it performs stratified subsampling to Config.DEEP_SUBSET_SIZE.
        For 'val' and 'test', it returns the full set (or processed version).

        Returns:
            pd.DataFrame: Contains 'text' (cleaned) and 'tags' (raw string) columns.
                          Also includes 'Id' for tracking.
        """
        # We cache the processed dataframes as parquet
        cache_filename = f"deep_data_{split}.parquet"
        cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached Deep data for {split} from {cache_path}...")
            return pd.read_parquet(cache_path)

        print(f"Processing Deep data for {split}...")
        df = self._load_raw_df(split)

        # Subsampling for Train
        if split == "train":
            print(
                f"Subsampling train set for Deep Model to {Config.DEEP_SUBSET_SIZE} samples..."
            )
            # Since the original train.csv is already stratified, a random sample
            # is a reasonable approximation of a stratified subsample.
            # However, to be precise, we can just sample randomly given the large N.
            if len(df) > Config.DEEP_SUBSET_SIZE:
                df = df.sample(n=Config.DEEP_SUBSET_SIZE, random_state=Config.SEED)
            print(f"Subsampled shape: {df.shape}")

        # Preprocess text
        print("Preprocessing text for Deep Model...")
        df["text"] = TextPreprocessor.preprocess_dataframe(df)

        # Ensure tags exist (empty for test)
        if "Tags" not in df.columns:
            df["tags"] = ""
        else:
            df["tags"] = df["Tags"].astype(str).fillna("")

        # Select relevant columns
        out_df = df[["Id", "text", "tags"]].copy()

        # Save cache
        print(f"Saving Deep data for {split} to {cache_path}...")
        out_df.to_parquet(cache_path, index=False)

        return out_df

    def get_tag_encoder(self):
        """Returns the fitted TagEncoder instance."""
        if not self.tag_encoder.tag_to_idx:
            if os.path.exists(Config.TAG_ENCODER_PATH):
                self.tag_encoder.load(Config.TAG_ENCODER_PATH)
        return self.tag_encoder
