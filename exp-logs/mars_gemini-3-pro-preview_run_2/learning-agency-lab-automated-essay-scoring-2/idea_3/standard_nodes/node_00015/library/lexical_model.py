import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from library.config import Config
from library.utils import compute_qwk, seed_everything
from library.dataset import load_data


class LexicalRidgePipeline:
    """
    Pipeline for the Lexical Branch (TF-IDF + Ridge Regression).
    Encapsulates vectorization and regression logic.
    """

    def __init__(self):
        # Initialize TfidfVectorizer with parameters from Config
        self.vectorizer = TfidfVectorizer(
            ngram_range=Config.tfidf_ngram_range,
            min_df=Config.tfidf_min_df,
            max_features=Config.tfidf_max_features,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{1,}",
            stop_words="english",
        )

        # Initialize Ridge Regressor
        self.model = Ridge(alpha=1.0, random_state=Config.seed, solver="auto")

    def run_fold(self, train_df, val_df, test_df):
        """
        Fits the vectorizer and model on training data, predicts on val and test.

        Args:
            train_df (pd.DataFrame): Training data.
            val_df (pd.DataFrame): Validation data.
            test_df (pd.DataFrame): Test data.

        Returns:
            dict: Contains 'val_preds', 'test_preds', 'val_score', 'model', 'vectorizer'.
        """
        # Ensure reproducibility
        seed_everything(Config.seed)

        print("Initializing TF-IDF Vectorizer...")
        print(
            f"Params: ngram_range={Config.tfidf_ngram_range}, min_df={Config.tfidf_min_df}, max_features={Config.tfidf_max_features}"
        )

        # --- Vectorization ---
        print("Fitting vectorizer on training text...")
        # Fit only on train, then transform all
        X_train = self.vectorizer.fit_transform(train_df["full_text"])
        y_train = train_df["score"].values.astype(float)

        print("Transforming validation and test text...")
        X_val = self.vectorizer.transform(val_df["full_text"])
        y_val = val_df["score"].values.astype(float)

        X_test = self.vectorizer.transform(test_df["full_text"])

        print(f"Train features shape: {X_train.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

        # --- Modeling ---
        print("Fitting Ridge Regression...")
        self.model.fit(X_train, y_train)

        # --- Prediction ---
        print("Generating predictions...")
        val_preds = self.model.predict(X_val)
        test_preds = self.model.predict(X_test)

        # Clip predictions to valid range (1-6) since Ridge can overshoot
        val_preds = np.clip(val_preds, 1, 6)
        test_preds = np.clip(test_preds, 1, 6)

        # --- Evaluation ---
        val_score = compute_qwk(y_val, val_preds)
        print(f"Validation QWK: {val_score}")

        return {
            "val_preds": val_preds,
            "test_preds": test_preds,
            "val_score": val_score,
            "model": self.model,
            "vectorizer": self.vectorizer,
        }


def train_lexical_model(load_cached_data=True, debug=False):
    """
    Orchestrates the loading of data and training of the lexical model.
    Uses library.dataset.load_data for caching dataframes.

    Args:
        load_cached_data (bool): Whether to use cached dataframes.
        debug (bool): Whether to run in debug mode (subset of data).

    Returns:
        dict: Results from run_fold.
    """
    # Load Data using the shared dataset utility
    print("Loading data for Lexical Model...")
    train_df = load_data("train", load_cached_data=load_cached_data)
    val_df = load_data("val", load_cached_data=load_cached_data)
    test_df = load_data("test", load_cached_data=load_cached_data)

    # Apply debug subsetting if requested
    if debug or Config.debug:
        print(f"Debug mode: using {Config.debug_subset_size} samples.")
        train_df = train_df.head(Config.debug_subset_size).reset_index(drop=True)
        val_df = val_df.head(Config.debug_subset_size).reset_index(drop=True)
        test_df = test_df.head(Config.debug_subset_size).reset_index(drop=True)

    # Initialize Pipeline
    pipeline = LexicalRidgePipeline()

    # Run Pipeline
    results = pipeline.run_fold(train_df, val_df, test_df)

    return results
