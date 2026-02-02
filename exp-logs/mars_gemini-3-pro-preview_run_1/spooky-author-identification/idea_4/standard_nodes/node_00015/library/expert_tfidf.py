import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion

from library.config import Config
from library.utils import seed_everything, compute_log_loss
from library.feature_engineering import SpacyPreprocessor


class TfidfExpert:
    """
    Expert model based on TF-IDF features and Logistic Regression.
    Supports two modes:
    1. 'lexical': Uses Word N-grams and Character N-grams via FeatureUnion.
    2. 'syntactic': Uses Part-of-Speech (POS) Tag N-grams.
    """

    def __init__(self, expert_type="lexical", model_params=None):
        """
        Args:
            expert_type (str): 'lexical' or 'syntactic'.
            model_params (dict): Hyperparameters for LogisticRegression.
        """
        self.expert_type = expert_type

        # Default parameters for Logistic Regression if not provided
        # Using lbfgs + multinomial to directly optimize multi-class log loss
        if model_params is None:
            self.model_params = {
                "C": 1.0,
                "solver": "lbfgs",
                "multi_class": "multinomial",
                "random_state": Config.SEED,
                "max_iter": 1000,
                "n_jobs": -1,
            }
        else:
            self.model_params = model_params
            if "random_state" not in self.model_params:
                self.model_params["random_state"] = Config.SEED

        self.pipeline = None
        # Initialize preprocessor only if needed for syntactic mode
        self.spacy_preprocessor = (
            SpacyPreprocessor() if expert_type == "syntactic" else None
        )

    def _get_vectorizer(self):
        """
        Constructs the appropriate vectorizer or feature union based on expert type.
        """
        if self.expert_type == "lexical":
            # Expert B: Word + Char N-grams
            word_vect = TfidfVectorizer(
                ngram_range=Config.TFIDF_NGRAMS,
                max_features=Config.TFIDF_MAX_FEATURES,
                analyzer="word",
                token_pattern=r"\w{1,}",
                strip_accents="unicode",
                sublinear_tf=True,
            )
            char_vect = TfidfVectorizer(
                ngram_range=Config.TFIDF_CHAR_NGRAMS,
                max_features=Config.TFIDF_MAX_FEATURES,
                analyzer="char",
                strip_accents="unicode",
                sublinear_tf=True,
            )
            return FeatureUnion([("word", word_vect), ("char", char_vect)])

        elif self.expert_type == "syntactic":
            # Expert C: POS N-grams
            # Input is a string of space-separated POS tags (e.g., "DET NOUN VERB")
            return TfidfVectorizer(
                ngram_range=Config.POS_NGRAMS,
                max_features=Config.POS_MAX_FEATURES,
                analyzer="word",
                token_pattern=r"\S+",  # Split by whitespace
                lowercase=False,  # Tags are usually uppercase (e.g., DET, NOUN)
                sublinear_tf=True,
            )
        else:
            raise ValueError(f"Unknown expert type: {self.expert_type}")

    def fit(
        self,
        texts,
        labels,
        dataset_name="train",
        val_texts=None,
        val_labels=None,
        val_dataset_name="val",
    ):
        """
        Fits the pipeline to the training data.

        Args:
            texts (list): Training text samples.
            labels (list): Training labels (integers).
            dataset_name (str): Name of the training dataset (for caching POS tags).
            val_texts (list, optional): Validation text samples.
            val_labels (list, optional): Validation labels.
            val_dataset_name (str): Name of the validation dataset.
        """
        seed_everything()

        print(f"Preparing data for {self.expert_type} expert...")

        # Preprocessing
        X_train = texts
        if self.expert_type == "syntactic":
            # Transform raw text to POS sequences
            # SpacyPreprocessor handles caching internally (saving to parquet)
            X_train = self.spacy_preprocessor.transform(
                texts, dataset_name=dataset_name
            )

        # Build Pipeline
        vectorizer = self._get_vectorizer()
        clf = LogisticRegression(**self.model_params)

        self.pipeline = Pipeline([("vec", vectorizer), ("clf", clf)])

        print(f"Fitting {self.expert_type} expert pipeline...")
        self.pipeline.fit(X_train, labels)

        # Validation
        if val_texts is not None and val_labels is not None:
            print(f"Evaluating {self.expert_type} expert on validation set...")
            val_probs = self.predict_proba(val_texts, dataset_name=val_dataset_name)
            loss = compute_log_loss(val_labels, val_probs)
            # Print full precision as requested
            print(f"Validation Log Loss ({self.expert_type}): {loss}")

        return self

    def predict_proba(self, texts, dataset_name="test"):
        """
        Generates probability predictions.

        Args:
            texts (list): Input text samples.
            dataset_name (str): Name of the dataset (for caching POS tags if syntactic).

        Returns:
            np.ndarray: Predicted probabilities.
        """
        if self.pipeline is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = texts
        if self.expert_type == "syntactic":
            X = self.spacy_preprocessor.transform(texts, dataset_name=dataset_name)

        return self.pipeline.predict_proba(X)

    def save(self, filepath):
        """
        Saves the fitted pipeline to disk using joblib.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.pipeline, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """
        Loads a fitted pipeline from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        self.pipeline = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self
