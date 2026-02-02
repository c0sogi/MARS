import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from library.config import Config


def load_data():
    """
    Loads the train, validation, and test datasets from the metadata directory.
    Applies subsetting if Config.DEBUG is True.
    """
    train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(Config.VAL_DATA_PATH)
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    if Config.DEBUG:
        # Use a small subset for debugging
        train_df = train_df.iloc[:200].reset_index(drop=True)
        val_df = val_df.iloc[:50].reset_index(drop=True)
        test_df = test_df.iloc[:50].reset_index(drop=True)
        print("DEBUG Mode: Loaded subset of data.")
    else:
        print(
            f"Loaded full data: Train {train_df.shape}, Val {val_df.shape}, Test {test_df.shape}"
        )

    return train_df, val_df, test_df


class MetaFeatureExtractor:
    """
    Extracts meta-features from text: character length, word count, and punctuation counts.
    """

    def __init__(self):
        # Specific punctuation marks mentioned in the idea: commas, semicolons, colons
        self.target_puncts = {",", ";", ":"}

    def _extract_single(self, text):
        text = str(text)
        char_len = len(text)
        words = text.split()
        word_count = len(words)
        punct_count = sum(1 for char in text if char in self.target_puncts)

        avg_word_len = char_len / word_count if word_count > 0 else 0

        return [char_len, word_count, punct_count, avg_word_len]

    def extract(self, texts):
        features = [self._extract_single(t) for t in texts]
        return np.array(features, dtype=np.float32)

    def get_features(self, df, dataset_name, load_cached_data=True):
        """
        Computes or loads cached meta-features.
        """
        suffix = "_debug" if Config.DEBUG else ""
        filename = f"meta_features_{dataset_name}{suffix}.npy"
        cache_path = os.path.join(Config.CACHE_DIR, filename)

        # Try loading cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                features = np.load(cache_path)
                # Verify consistency
                if len(features) == len(df):
                    print(f"Loaded meta-features for {dataset_name} from cache.")
                    return features
                else:
                    print(
                        f"Cache mismatch for {dataset_name} (Size {len(features)} vs {len(df)}). Recomputing..."
                    )
            except Exception as e:
                print(f"Error loading cache for {dataset_name}: {e}. Recomputing...")

        # Compute
        print(f"Computing meta-features for {dataset_name}...")
        features = self.extract(df["text"].tolist())

        # Save
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        np.save(cache_path, features)

        return features


class TextPreprocessor:
    """
    Handles TF-IDF vectorization for Stylometric Expert.
    """

    def __init__(self):
        self.word_vectorizer = TfidfVectorizer(**Config.TFIDF_WORD_PARAMS)
        self.char_vectorizer = TfidfVectorizer(**Config.TFIDF_CHAR_PARAMS)

    def get_tfidf_features(
        self, train_text, val_text, test_text, load_cached_data=True
    ):
        """
        Generates or loads TF-IDF features (Word + Char n-grams).
        """
        suffix = "_debug" if Config.DEBUG else ""
        train_path = os.path.join(Config.CACHE_DIR, f"tfidf_train{suffix}.npz")
        val_path = os.path.join(Config.CACHE_DIR, f"tfidf_val{suffix}.npz")
        test_path = os.path.join(Config.CACHE_DIR, f"tfidf_test{suffix}.npz")

        # Check if all cache files exist
        if (
            load_cached_data
            and os.path.exists(train_path)
            and os.path.exists(val_path)
            and os.path.exists(test_path)
        ):
            try:
                X_train = sparse.load_npz(train_path)
                X_val = sparse.load_npz(val_path)
                X_test = sparse.load_npz(test_path)

                # Verify dimensions
                if (
                    X_train.shape[0] == len(train_text)
                    and X_val.shape[0] == len(val_text)
                    and X_test.shape[0] == len(test_text)
                ):
                    print("Loaded TF-IDF features from cache.")
                    return X_train, X_val, X_test
                else:
                    print("Cached TF-IDF dimensions mismatch. Recomputing...")
            except Exception as e:
                print(f"Error loading TF-IDF cache: {e}. Recomputing...")

        print("Computing TF-IDF features...")

        # Fit on training data
        print("Fitting Word Vectorizer...")
        self.word_vectorizer.fit(train_text)
        print("Fitting Char Vectorizer...")
        self.char_vectorizer.fit(train_text)

        # Transform helper
        def transform(texts):
            w = self.word_vectorizer.transform(texts)
            c = self.char_vectorizer.transform(texts)
            return sparse.hstack([w, c])

        print("Transforming datasets...")
        X_train = transform(train_text)
        X_val = transform(val_text)
        X_test = transform(test_text)

        # Save to cache
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        print("Saving TF-IDF features to cache...")
        sparse.save_npz(train_path, X_train)
        sparse.save_npz(val_path, X_val)
        sparse.save_npz(test_path, X_test)

        return X_train, X_val, X_test


class TransformerDataset(Dataset):
    """
    PyTorch Dataset for DeBERTa model.
    """

    def __init__(
        self, texts, labels=None, tokenizer=None, max_length=Config.MAX_LENGTH
    ):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = Config.LABEL2ID

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        if self.labels is not None:
            label_str = self.labels[idx]
            label_id = self.label2id[label_str]
            item["labels"] = torch.tensor(label_id, dtype=torch.long)

        return item
