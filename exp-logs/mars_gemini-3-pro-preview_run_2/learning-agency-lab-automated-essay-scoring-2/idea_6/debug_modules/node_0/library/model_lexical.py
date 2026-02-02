import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from library.config import Config
from library.utils import quadratic_weighted_kappa


class LexicalRegressor:
    """
    Lexical Branch Model: Sparse TF-IDF N-grams + Ridge Regression.
    Wraps scikit-learn components into a unified interface.
    """

    def __init__(self):
        """
        Initializes the pipeline with hyperparameters from Config.
        """
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=Config.TFIDF_NGRAM_RANGE,
                        min_df=Config.TFIDF_MIN_DF,
                        sublinear_tf=True,
                        strip_accents="unicode",
                        analyzer="word",
                        token_pattern=r"\w{1,}",
                        stop_words="english",
                    ),
                ),
                (
                    "ridge",
                    Ridge(
                        alpha=Config.RIDGE_ALPHA,
                        random_state=Config.SEED,
                        solver="auto",
                    ),
                ),
            ]
        )

    def fit(self, texts, scores):
        """
        Fits the TF-IDF Vectorizer and Ridge Regressor.

        Args:
            texts (list or pd.Series): List of essay texts.
            scores (list or np.array): Target scores.
        """
        self.pipeline.fit(texts, scores)
        return self

    def predict(self, texts):
        """
        Generates predictions using the trained pipeline.

        Args:
            texts (list or pd.Series): List of essay texts.

        Returns:
            np.array: Predicted scores.
        """
        return self.pipeline.predict(texts)

    def save(self, path):
        """
        Saves the trained pipeline to disk.

        Args:
            path (str): File path to save the model.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)

    def load(self, path):
        """
        Loads a trained pipeline from disk.

        Args:
            path (str): File path to load the model from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Lexical model not found at {path}")
        self.pipeline = joblib.load(path)
        return self


def train_lexical_fold(fold_idx, train_df, val_df):
    """
    Trains the lexical model for a single fold, evaluates it, and saves the artifact.

    Args:
        fold_idx (int): The current fold index.
        train_df (pd.DataFrame): Training data for this fold.
        val_df (pd.DataFrame): Validation data for this fold.

    Returns:
        tuple: (val_qwk, val_preds)
            val_qwk (float): Quadratic Weighted Kappa score on validation set.
            val_preds (np.array): Predictions for the validation set.
    """
    print(f"\n=== Training Lexical Model | Fold {fold_idx} ===")

    # Prepare Data
    # Ensure text is string and handle potential NaNs (though EDA showed none)
    X_train = train_df["full_text"].fillna("").astype(str).tolist()
    y_train = train_df["score"].values

    X_val = val_df["full_text"].fillna("").astype(str).tolist()
    y_val = val_df["score"].values

    # Initialize Model
    model = LexicalRegressor()

    # Train
    print(f"Fitting TF-IDF and Ridge on {len(X_train)} samples...")
    model.fit(X_train, y_train)

    # Predict on Validation
    val_preds = model.predict(X_val)

    # Evaluate
    # Printing full precision as requested
    qwk = quadratic_weighted_kappa(y_val, val_preds)
    print(f"Fold {fold_idx} Lexical QWK: {qwk}")

    # Save Model
    save_path = os.path.join(Config.MODEL_DIR, f"lexical_fold_{fold_idx}.joblib")
    model.save(save_path)
    print(f"Saved lexical model to {save_path}")

    return qwk, val_preds


def predict_lexical(model_path, df):
    """
    Loads a trained lexical model and generates predictions for a dataframe.

    Args:
        model_path (str): Path to the saved .joblib model file.
        df (pd.DataFrame): Dataframe containing 'full_text'.

    Returns:
        np.array: Predicted scores.
    """
    model = LexicalRegressor()
    model.load(model_path)

    texts = df["full_text"].fillna("").astype(str).tolist()
    preds = model.predict(texts)

    return preds
