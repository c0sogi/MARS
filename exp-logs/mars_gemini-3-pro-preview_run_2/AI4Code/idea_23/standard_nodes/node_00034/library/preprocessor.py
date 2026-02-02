import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from library.config import Config


class TextPipeline:
    """
    Manages text vectorization and dimensionality reduction.
    Wraps TfidfVectorizer and TruncatedSVD to project notebook cells into a shared semantic space.
    """

    def __init__(self):
        self.config = Config
        self.working_dir = self.config.WORKING_DIR

        # Define paths for cached models
        self.tfidf_path = os.path.join(self.working_dir, "text_vectorizer_tfidf.joblib")
        self.svd_path = os.path.join(self.working_dir, "text_vectorizer_svd.joblib")

        self.vectorizer = None
        self.svd = None

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

    def fit_transform_corpus(
        self, df: pd.DataFrame, load_cached_models: bool = True
    ) -> np.ndarray:
        """
        Fits the pipeline on the markdown cells of the provided dataframe (or loads cached models)
        and transforms all cells (code + markdown) into the latent space.

        Args:
            df (pd.DataFrame): DataFrame containing 'source' and 'cell_type' columns.
            load_cached_models (bool): If True, attempts to load fitted models from disk.

        Returns:
            np.ndarray: SVD-transformed features for the entire input dataframe.
        """
        models_loaded = False

        # 1. Try to load cached models
        if (
            load_cached_models
            and os.path.exists(self.tfidf_path)
            and os.path.exists(self.svd_path)
        ):
            try:
                print(f"Loading text models from {self.working_dir}...")
                self.vectorizer = joblib.load(self.tfidf_path)
                self.svd = joblib.load(self.svd_path)
                models_loaded = True
            except Exception as e:
                print(f"Failed to load cached models: {e}. Proceeding to retrain.")

        # 2. Fit models if not loaded
        if not models_loaded:
            print("Fitting text models from scratch...")

            # Filter for markdown cells for training the vocabulary/semantic space
            # as per the strategy: "Fit a TF-IDF Vectorizer on the full markdown corpus."
            markdown_mask = df["cell_type"] == "markdown"
            train_corpus = df.loc[markdown_mask, "source"].astype(str).fillna("")

            # Initialize Vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=self.config.TFIDF_VOCAB_SIZE,
                ngram_range=self.config.TFIDF_NGRAM_RANGE,
                min_df=self.config.TFIDF_MIN_DF,
                sublinear_tf=self.config.TFIDF_SUBLINEAR_TF,
                strip_accents=None,  # "No Accent Stripping"
            )

            # Fit Vectorizer
            tfidf_matrix = self.vectorizer.fit_transform(train_corpus)

            # Initialize SVD
            self.svd = TruncatedSVD(
                n_components=self.config.SVD_N_COMPONENTS,
                random_state=self.config.SVD_RANDOM_STATE,
            )

            # Fit SVD
            self.svd.fit(tfidf_matrix)

            # Save models
            joblib.dump(self.vectorizer, self.tfidf_path)
            joblib.dump(self.svd, self.svd_path)
            print(f"Models saved to {self.working_dir}")

        # 3. Transform the entire corpus (Code + Markdown)
        # "Project Code Cells: Transform the notebook's code cells into this same SVD space"
        print("Transforming entire corpus to SVD space...")
        full_corpus = df["source"].astype(str).fillna("")
        return self.transform(full_corpus)

    def transform(self, text_sequence) -> np.ndarray:
        """
        Projects a sequence of text into the fitted SVD space.

        Args:
            text_sequence: Iterable of strings (e.g., pandas Series, list).

        Returns:
            np.ndarray: The transformed features (n_samples, n_components).
        """
        # Ensure models are loaded
        if self.vectorizer is None or self.svd is None:
            if os.path.exists(self.tfidf_path) and os.path.exists(self.svd_path):
                self.vectorizer = joblib.load(self.tfidf_path)
                self.svd = joblib.load(self.svd_path)
            else:
                raise RuntimeError(
                    "Models not fitted. Call fit_transform_corpus first."
                )

        # Transform
        tfidf_features = self.vectorizer.transform(text_sequence)
        svd_features = self.svd.transform(tfidf_features)

        return svd_features.astype(np.float32)
