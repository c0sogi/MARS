import os
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from library.config import Config


class StreamClassifier:
    """
    A wrapper class for modality-specific classification pipelines using LDA.

    Implements the strategy:
    - Visual Streams (DINOv2, ConvNeXt): PCA (99% variance) -> LDA
    - Tabular Stream: QuantileTransformer (Normal) -> LDA
    """

    def __init__(self, stream_type: str):
        """
        Initialize the classifier.

        Args:
            stream_type (str): Type of data stream. Must be 'visual' or 'tabular'.
        """
        self.stream_type = stream_type
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        """
        Constructs the sklearn pipeline based on stream type.
        """
        # Common LDA settings
        # solver='lsqr' supports shrinkage, which is crucial for HDLSS (High Dimension Low Sample Size)
        # shrinkage='auto' uses the Ledoit-Wolf lemma for covariance estimation
        lda = LinearDiscriminantAnalysis(
            solver=Config.LDA_SOLVER, shrinkage=Config.LDA_SHRINKAGE
        )

        if self.stream_type == "visual":
            # Visual Pipeline:
            # 1. PCA to reduce dimensionality while retaining 99% variance.
            #    This removes noise and makes covariance estimation in LDA more stable.
            pca = PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED)
            return Pipeline([("pca", pca), ("lda", lda)])

        elif self.stream_type == "tabular":
            # Tabular Pipeline:
            # 1. QuantileTransformer to force features into a Gaussian distribution.
            #    LDA assumes class-conditional densities are Gaussian.
            qt = QuantileTransformer(
                output_distribution="normal", random_state=Config.SEED
            )
            return Pipeline([("qt", qt), ("lda", lda)])

        else:
            raise ValueError(
                f"Invalid stream_type '{self.stream_type}'. Expected 'visual' or 'tabular'."
            )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the pipeline to the training data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
        """
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Probability matrix (N_samples, N_classes).
        """
        return self.pipeline.predict_proba(X)

    def save(self, filepath: str):
        """
        Save the fitted pipeline to disk using pickle.

        Args:
            filepath (str): Destination path.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.pipeline, f)

    def load(self, filepath: str):
        """
        Load a fitted pipeline from disk.

        Args:
            filepath (str): Source path.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")

        with open(filepath, "rb") as f:
            self.pipeline = pickle.load(f)
        return self
