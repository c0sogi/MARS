import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
import torch

from library.config import (
    VOCAB_SIZE,
    NGRAM_RANGE,
    USE_IDF,
    SUBLINEAR_TF,
    STRIP_ACCENTS,
    VECTORIZER_PATH,
    MODEL_NAME,
    MAX_LEN,
    SEED,
)
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(SEED)


class SparseVectorizer:
    """
    Manages the TF-IDF Vectorization for the Sparse Regression Stream.
    Wraps sklearn's TfidfVectorizer with specific configurations optimized for code/markdown.
    """

    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        ngram_range=NGRAM_RANGE,
        use_idf=USE_IDF,
        sublinear_tf=SUBLINEAR_TF,
        strip_accents=STRIP_ACCENTS,
    ):
        self.vocab_size = vocab_size
        self.ngram_range = ngram_range
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.strip_accents = strip_accents

        self.vectorizer = TfidfVectorizer(
            max_features=self.vocab_size,
            ngram_range=self.ngram_range,
            use_idf=self.use_idf,
            sublinear_tf=self.sublinear_tf,
            strip_accents=self.strip_accents,
            token_pattern=r"(?u)\b\w\w+\b",  # Standard alphanumeric token pattern
            dtype=np.float32,
        )
        self.is_fitted = False

    def fit(self, texts):
        """
        Fits the vectorizer on the provided iterable of text documents.
        """
        print("Fitting SparseVectorizer on training corpus...")
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts):
        """
        Transforms documents to a sparse document-term matrix.
        """
        if not self.is_fitted:
            raise ValueError(
                "SparseVectorizer must be fitted before calling transform."
            )
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        """
        Fits and transforms documents in a single step.
        """
        print("Fitting and transforming with SparseVectorizer...")
        result = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return result

    def save(self, path=VECTORIZER_PATH):
        """
        Saves the fitted vectorizer to disk using joblib.
        """
        if not self.is_fitted:
            print("Warning: Attempting to save an unfitted vectorizer.")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.vectorizer, path)
        print(f"SparseVectorizer saved to {path}")

    def load(self, path=VECTORIZER_PATH):
        """
        Loads a fitted vectorizer from disk.
        """
        if os.path.exists(path):
            self.vectorizer = joblib.load(path)
            self.is_fitted = True
            print(f"SparseVectorizer loaded from {path}")
        else:
            print(f"Warning: No vectorizer found at {path}. Initialize clean.")
        return self


class DenseInputProcessor:
    """
    Manages tokenization and input formatting for the Dense Transformer Stream.
    Constructs inputs in the format: [CLS] Markdown Text [SEP] Structural Context [SEP]
    """

    def __init__(self, model_name=MODEL_NAME, max_len=MAX_LEN):
        self.model_name = model_name
        self.max_len = max_len
        # Suppress tokenizer warnings if possible
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def process_batch(self, texts, contexts):
        """
        Tokenizes a batch of text and context pairs.

        Args:
            texts (list of str): The markdown content.
            contexts (list of str): The structural context strings (anchors + keywords).

        Returns:
            dict: Dictionary containing 'input_ids' and 'attention_mask' tensors.
        """
        # Ensure inputs are strings
        texts = [str(t) for t in texts]
        contexts = [str(c) for c in contexts]

        # Use the tokenizer's pair encoding capability.
        # This automatically handles [SEP] placement:
        # Sequence 1: texts
        # Sequence 2: contexts
        encodings = self.tokenizer(
            texts,
            text_pair=contexts,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }
