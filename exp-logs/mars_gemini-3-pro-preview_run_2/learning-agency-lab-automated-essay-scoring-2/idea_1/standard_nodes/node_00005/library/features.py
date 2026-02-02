import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from library.config import Config
from library.utils import save_artifact, load_artifact


class TextVectorizer:
    """
    Wrapper around sklearn's TfidfVectorizer to handle configuration,
    fitting, transforming, and persistence.
    """

    def __init__(self, **kwargs):
        """
        Initializes the vectorizer with parameters from Config.
        Allows overriding parameters via kwargs.
        """
        self.params = Config.TFIDF_PARAMS.copy()
        self.params.update(kwargs)
        self.vectorizer = TfidfVectorizer(**self.params)

    def fit(self, raw_documents):
        """
        Fits the vectorizer to the raw documents.
        """
        self.vectorizer.fit(raw_documents)
        return self

    def transform(self, raw_documents):
        """
        Transforms raw documents to a sparse TF-IDF matrix.
        """
        return self.vectorizer.transform(raw_documents)

    def fit_transform(self, raw_documents):
        """
        Fits and transforms the raw documents.
        """
        return self.vectorizer.fit_transform(raw_documents)

    def save(self, path):
        """
        Saves the underlying sklearn vectorizer to the specified path.
        """
        save_artifact(self.vectorizer, path)

    @classmethod
    def load(cls, path):
        """
        Loads the sklearn vectorizer from the specified path and returns
        a TextVectorizer instance wrapping it.
        """
        # Load the sklearn object
        sklearn_vectorizer = load_artifact(path)

        # Create a new instance of the wrapper
        instance = cls()

        # Inject the loaded vectorizer
        instance.vectorizer = sklearn_vectorizer

        # Update params to match the loaded vectorizer's params (for consistency)
        instance.params = sklearn_vectorizer.get_params()

        return instance


def extract_features(df: pd.DataFrame, split: str):
    """
    Extracts TF-IDF features from the dataframe.

    Logic:
    - If split is 'train': Fits a new vectorizer, saves it, and transforms the text.
    - If split is 'val' or 'test': Loads the saved vectorizer and transforms the text.

    Args:
        df (pd.DataFrame): The dataframe containing the text column.
        split (str): The data split ('train', 'val', 'test').

    Returns:
        scipy.sparse.csr_matrix: The sparse feature matrix.
    """
    # Ensure text column exists
    if Config.TEXT_COL not in df.columns:
        raise KeyError(f"Column '{Config.TEXT_COL}' not found in dataframe.")

    # Fill NaNs with empty string to prevent vectorizer errors and ensure string type
    texts = df[Config.TEXT_COL].fillna("").astype(str)

    if split == "train":
        # Initialize and fit new vectorizer
        vectorizer = TextVectorizer()
        features = vectorizer.fit_transform(texts)

        # Save the vectorizer for use in val/test
        print(f"Saving fitted vectorizer to {Config.VECTORIZER_PATH}")
        vectorizer.save(Config.VECTORIZER_PATH)

    else:
        # Load existing vectorizer
        if not os.path.exists(Config.VECTORIZER_PATH):
            raise FileNotFoundError(
                f"Vectorizer not found at {Config.VECTORIZER_PATH}. "
                "Please run with split='train' first to generate the vectorizer."
            )

        vectorizer = TextVectorizer.load(Config.VECTORIZER_PATH)
        features = vectorizer.transform(texts)

    return features
