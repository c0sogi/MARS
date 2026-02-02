import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import joblib
from library.config import Config


class SparseRanker:
    """
    Implements the Sparse Lexical Stream using TF-IDF and Ridge Regression.
    """

    def __init__(self):
        """
        Initializes the SparseRanker with TfidfVectorizer and Ridge model.
        Hyperparameters are drawn from the Config class.
        """
        # TF-IDF Vectorizer with Bigrams and Logarithmic Term Frequency Scaling
        # as per the solution design.
        self.vectorizer = TfidfVectorizer(
            min_df=2,  # Ignore extremely rare terms
            max_features=Config.VOCAB_SIZE,
            analyzer="word",
            ngram_range=(1, 2),  # Unigrams and Bigrams
            sublinear_tf=True,  # Logarithmic term frequency scaling
            dtype=np.float32,
            strip_accents=None,  # Do not strip accents
            lowercase=True,
        )

        # Ridge Regression
        self.model = Ridge(
            solver="auto", fit_intercept=True, random_state=Config.SEED, alpha=1.0
        )

    def fit(self, df_train, df_val=None):
        """
        Trains the vectorizer and the regression model.

        Args:
            df_train (pd.DataFrame): Training data with 'source' and 'rank'.
            df_val (pd.DataFrame, optional): Validation data for evaluation.
        """
        print("Fitting SparseRanker (TF-IDF + Ridge)...")

        # Extract text and targets
        train_text = df_train["source"].fillna("").tolist()
        train_y = df_train["rank"].values

        # 1. Fit Vectorizer
        print(f"Vectorizing {len(train_text)} documents...")
        self.vectorizer.fit(train_text)
        X_train = self.vectorizer.transform(train_text)

        # 2. Fit Ridge Model
        print("Training Ridge Regressor...")
        self.model.fit(X_train, train_y)

        # 3. Evaluate if validation set is provided
        if df_val is not None:
            print("Evaluating on validation set...")
            val_text = df_val["source"].fillna("").tolist()
            val_y = df_val["rank"].values

            X_val = self.vectorizer.transform(val_text)
            preds_val = self.model.predict(X_val)

            mse = mean_squared_error(val_y, preds_val)
            print(f"Sparse Stream Validation MSE: {mse}")

    def predict(self, df):
        """
        Generates rank predictions for the provided data.

        Args:
            df (pd.DataFrame): Data containing 'source' column.

        Returns:
            np.ndarray: Predicted ranks.
        """
        text = df["source"].fillna("").tolist()
        X = self.vectorizer.transform(text)
        preds = self.model.predict(X)
        return preds

    def save(self, output_dir=Config.WORKING_DIR):
        """
        Saves the vectorizer and model to the specified directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        vec_path = os.path.join(output_dir, "tfidf_vectorizer.joblib")
        model_path = os.path.join(output_dir, "ridge_model.joblib")

        print(f"Saving SparseRanker artifacts to {output_dir}...")
        joblib.dump(self.vectorizer, vec_path)
        joblib.dump(self.model, model_path)

    def load(self, input_dir=Config.WORKING_DIR):
        """
        Loads the vectorizer and model from the specified directory.
        """
        vec_path = os.path.join(input_dir, "tfidf_vectorizer.joblib")
        model_path = os.path.join(input_dir, "ridge_model.joblib")

        if not os.path.exists(vec_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"SparseRanker artifacts not found in {input_dir}")

        print(f"Loading SparseRanker artifacts from {input_dir}...")
        self.vectorizer = joblib.load(vec_path)
        self.model = joblib.load(model_path)
