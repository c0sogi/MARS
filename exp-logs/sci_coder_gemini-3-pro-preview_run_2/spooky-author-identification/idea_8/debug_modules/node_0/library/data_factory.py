import os
import re
import pandas as pd
import numpy as np
import torch
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from library.config import Config


class AuthorDataset(Dataset):
    """
    PyTorch Dataset for the Stylometric-Fusion Transformer.
    Returns tokenized text, dense style features, and labels.
    """

    def __init__(
        self, df, tokenizer, style_features, max_len=Config.MAX_LEN, is_test=False
    ):
        self.texts = df[Config.TEXT_COL].values
        self.style_features = style_features
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        if not self.is_test:
            self.labels = df[Config.TARGET_COL].map(Config.LABEL_MAP).values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        style = self.style_features[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "style_features": torch.tensor(style, dtype=torch.float),
        }

        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


class DataManager:
    """
    Handles data loading, feature engineering (Style, TF-IDF, SVD), and caching.
    """

    @staticmethod
    def load_metadata():
        """
        Loads the train, validation, and test metadata CSVs.
        """
        train_df = pd.read_csv(Config.TRAIN_META)
        val_df = pd.read_csv(Config.VAL_META)
        test_df = pd.read_csv(Config.TEST_META)
        return train_df, val_df, test_df

    @staticmethod
    def _extract_raw_style_features(texts):
        """
        Computes dense stylometric features for a list of texts.
        Features:
        1. Character count
        2. Word count
        3. Unique word count
        4. Sentence count
        5. Avg word length
        6. Avg sentence length
        7. Type-Token Ratio
        8. Punctuation counts/densities (;, :, ,, ?, !, ", ')
        """
        features = []

        # Pre-compile regex for speed
        # Simple word tokenization by whitespace for statistics

        for text in texts:
            text = str(text)
            chars = len(text)
            words = text.split()
            word_count = len(words)
            unique_words = len(set(words))

            # Sentence count (approximate by splitting on .!?)
            sentences = re.split(r"[.!?]+", text)
            sentences = [s for s in sentences if len(s.strip()) > 0]
            sentence_count = len(sentences) if len(sentences) > 0 else 1

            avg_word_len = np.mean([len(w) for w in words]) if word_count > 0 else 0
            avg_sentence_len = word_count / sentence_count
            ttr = unique_words / word_count if word_count > 0 else 0

            # Punctuation counts
            semicolons = text.count(";")
            colons = text.count(":")
            commas = text.count(",")
            questions = text.count("?")
            exclamations = text.count("!")
            quotes = text.count('"') + text.count("'")

            # Densities (per 100 chars to keep scale reasonable)
            # Or just raw counts + length features.
            # We will use raw counts and let StandardScaler handle the range.

            row = [
                chars,
                word_count,
                unique_words,
                sentence_count,
                avg_word_len,
                avg_sentence_len,
                ttr,
                semicolons,
                colons,
                commas,
                questions,
                exclamations,
                quotes,
            ]
            features.append(row)

        return np.array(features, dtype=np.float32)

    @staticmethod
    def get_style_features(train_df, val_df, test_df, load_cached_data=True):
        """
        Computes or loads scaled stylometric features.
        """
        cache_dir = Config.WORKING_DIR
        os.makedirs(cache_dir, exist_ok=True)

        f_train = os.path.join(cache_dir, "style_train.npy")
        f_val = os.path.join(cache_dir, "style_val.npy")
        f_test = os.path.join(cache_dir, "style_test.npy")

        if (
            load_cached_data
            and os.path.exists(f_train)
            and os.path.exists(f_val)
            and os.path.exists(f_test)
        ):
            print("Loading cached style features...")
            train_feats = np.load(f_train)
            val_feats = np.load(f_val)
            test_feats = np.load(f_test)
        else:
            print("Computing style features...")
            raw_train = DataManager._extract_raw_style_features(
                train_df[Config.TEXT_COL].values
            )
            raw_val = DataManager._extract_raw_style_features(
                val_df[Config.TEXT_COL].values
            )
            raw_test = DataManager._extract_raw_style_features(
                test_df[Config.TEXT_COL].values
            )

            # Scale features based on training distribution
            scaler = StandardScaler()
            train_feats = scaler.fit_transform(raw_train)
            val_feats = scaler.transform(raw_val)
            test_feats = scaler.transform(raw_test)

            np.save(f_train, train_feats)
            np.save(f_val, val_feats)
            np.save(f_test, test_feats)

        return train_feats, val_feats, test_feats

    @staticmethod
    def get_tfidf_features(train_df, val_df, test_df, load_cached_data=True):
        """
        Computes or loads sparse TF-IDF features (Word + Char n-grams).
        """
        cache_dir = Config.WORKING_DIR
        os.makedirs(cache_dir, exist_ok=True)

        f_train = os.path.join(cache_dir, "tfidf_train.npz")
        f_val = os.path.join(cache_dir, "tfidf_val.npz")
        f_test = os.path.join(cache_dir, "tfidf_test.npz")

        if (
            load_cached_data
            and os.path.exists(f_train)
            and os.path.exists(f_val)
            and os.path.exists(f_test)
        ):
            print("Loading cached TF-IDF features...")
            train_tfidf = scipy.sparse.load_npz(f_train)
            val_tfidf = scipy.sparse.load_npz(f_val)
            test_tfidf = scipy.sparse.load_npz(f_test)
        else:
            print("Computing TF-IDF features...")
            # Word TF-IDF
            word_vec = TfidfVectorizer(**Config.TFIDF_PARAMS)
            train_word = word_vec.fit_transform(train_df[Config.TEXT_COL].astype(str))
            val_word = word_vec.transform(val_df[Config.TEXT_COL].astype(str))
            test_word = word_vec.transform(test_df[Config.TEXT_COL].astype(str))

            # Char TF-IDF
            char_vec = TfidfVectorizer(**Config.TFIDF_CHAR_PARAMS)
            train_char = char_vec.fit_transform(train_df[Config.TEXT_COL].astype(str))
            val_char = char_vec.transform(val_df[Config.TEXT_COL].astype(str))
            test_char = char_vec.transform(test_df[Config.TEXT_COL].astype(str))

            # Concatenate
            train_tfidf = scipy.sparse.hstack([train_word, train_char])
            val_tfidf = scipy.sparse.hstack([val_word, val_char])
            test_tfidf = scipy.sparse.hstack([test_word, test_char])

            scipy.sparse.save_npz(f_train, train_tfidf)
            scipy.sparse.save_npz(f_val, val_tfidf)
            scipy.sparse.save_npz(f_test, test_tfidf)

        return train_tfidf, val_tfidf, test_tfidf

    @staticmethod
    def get_svd_features(train_tfidf, val_tfidf, test_tfidf, load_cached_data=True):
        """
        Computes or loads SVD projections of the TF-IDF matrices.
        """
        cache_dir = Config.WORKING_DIR
        os.makedirs(cache_dir, exist_ok=True)

        f_train = os.path.join(cache_dir, "svd_train.npy")
        f_val = os.path.join(cache_dir, "svd_val.npy")
        f_test = os.path.join(cache_dir, "svd_test.npy")

        if (
            load_cached_data
            and os.path.exists(f_train)
            and os.path.exists(f_val)
            and os.path.exists(f_test)
        ):
            print("Loading cached SVD features...")
            train_svd = np.load(f_train)
            val_svd = np.load(f_val)
            test_svd = np.load(f_test)
        else:
            print("Computing SVD features...")
            svd = TruncatedSVD(
                n_components=Config.SVD_COMPONENTS, random_state=Config.SEED
            )
            train_svd = svd.fit_transform(train_tfidf)
            val_svd = svd.transform(val_tfidf)
            test_svd = svd.transform(test_tfidf)

            np.save(f_train, train_svd)
            np.save(f_val, val_svd)
            np.save(f_test, test_svd)

        return train_svd, val_svd, test_svd
