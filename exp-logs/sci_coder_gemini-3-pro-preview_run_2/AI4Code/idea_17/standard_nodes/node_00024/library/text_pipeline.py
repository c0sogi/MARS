import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from library.config import Config


class TextVectorizers:
    """
    Manages TF-IDF vectorizers for Markdown (Target) and Code (Anchor) text.
    Persists fitted models to disk to avoid re-computation.
    """

    def __init__(self):
        self.config = Config
        self.md_params = self.config.MD_TFIDF_PARAMS
        self.code_params = self.config.CODE_TFIDF_PARAMS

        # Initialize vectorizers
        self.md_vectorizer = TfidfVectorizer(**self.md_params)
        self.code_vectorizer = TfidfVectorizer(**self.code_params)

        # Paths for caching models
        self.md_model_path = os.path.join(
            self.config.WORKING_DIR, "md_tfidf_vectorizer.joblib"
        )
        self.code_model_path = os.path.join(
            self.config.WORKING_DIR, "code_tfidf_vectorizer.joblib"
        )

    def fit_markdown(self, texts, load_cached_data: bool = True):
        """
        Fits the Markdown TF-IDF vectorizer.

        Args:
            texts: Iterable of markdown source strings.
            load_cached_data: If True, attempts to load a pre-trained vectorizer from disk.
        """
        if load_cached_data and os.path.exists(self.md_model_path):
            print(f"Loading cached Markdown TF-IDF model from {self.md_model_path}")
            self.md_vectorizer = joblib.load(self.md_model_path)
        else:
            print("Fitting Markdown TF-IDF vectorizer...")
            self.md_vectorizer.fit(texts)
            print(f"Saving Markdown TF-IDF model to {self.md_model_path}")
            joblib.dump(self.md_vectorizer, self.md_model_path)

    def transform_markdown(self, texts):
        """
        Transforms texts using the fitted Markdown vectorizer.
        """
        return self.md_vectorizer.transform(texts)

    def fit_code(self, texts, load_cached_data: bool = True):
        """
        Fits the Code TF-IDF vectorizer.

        Args:
            texts: Iterable of code source strings.
            load_cached_data: If True, attempts to load a pre-trained vectorizer from disk.
        """
        if load_cached_data and os.path.exists(self.code_model_path):
            print(f"Loading cached Code TF-IDF model from {self.code_model_path}")
            self.code_vectorizer = joblib.load(self.code_model_path)
        else:
            print("Fitting Code TF-IDF vectorizer...")
            self.code_vectorizer.fit(texts)
            print(f"Saving Code TF-IDF model to {self.code_model_path}")
            joblib.dump(self.code_vectorizer, self.code_model_path)

    def transform_code(self, texts):
        """
        Transforms texts using the fitted Code vectorizer.
        """
        return self.code_vectorizer.transform(texts)


class FunctionalCodeClusterer:
    """
    Implements the Unsupervised Learning pipeline for Functional Landmark Triangulation.
    Consists of:
    1. TruncatedSVD: Reduces high-dimensional Code TF-IDF to dense vectors.
    2. MiniBatchKMeans: Clusters dense vectors into 'Functional Types' (e.g., Imports, Plots).
    """

    def __init__(self):
        self.config = Config

        # Initialize models
        self.svd = TruncatedSVD(
            n_components=self.config.CODE_SVD_COMPONENTS,
            random_state=self.config.RANDOM_STATE,
        )
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.config.NUM_CODE_CLUSTERS,
            batch_size=1024,
            random_state=self.config.RANDOM_STATE,
            n_init=10,
        )

        # Paths for caching models
        self.svd_path = os.path.join(self.config.WORKING_DIR, "code_svd_model.joblib")
        self.kmeans_path = os.path.join(
            self.config.WORKING_DIR, "code_kmeans_model.joblib"
        )

    def fit(self, sparse_matrix, load_cached_data: bool = True):
        """
        Fits the SVD and KMeans models on the Code TF-IDF sparse matrix.

        Args:
            sparse_matrix: Scipy sparse matrix (output of TextVectorizers.transform_code).
            load_cached_data: If True, attempts to load pre-trained models from disk.
        """
        # Check if both models exist
        models_exist = os.path.exists(self.svd_path) and os.path.exists(
            self.kmeans_path
        )

        if load_cached_data and models_exist:
            print(
                f"Loading cached Code SVD and KMeans models from {self.config.WORKING_DIR}"
            )
            self.svd = joblib.load(self.svd_path)
            self.kmeans = joblib.load(self.kmeans_path)
        else:
            print("Fitting TruncatedSVD on code vectors...")
            # Fit SVD
            dense_vecs = self.svd.fit_transform(sparse_matrix)
            print(
                f"Explained Variance Ratio: {self.svd.explained_variance_ratio_.sum():.4f}"
            )

            print("Fitting MiniBatchKMeans on SVD vectors...")
            # Fit KMeans on the reduced dense vectors
            self.kmeans.fit(dense_vecs)

            # Save models
            print(f"Saving SVD model to {self.svd_path}")
            joblib.dump(self.svd, self.svd_path)
            print(f"Saving KMeans model to {self.kmeans_path}")
            joblib.dump(self.kmeans, self.kmeans_path)

    def get_svd_vectors(self, sparse_matrix):
        """
        Projects the sparse matrix into the SVD latent space.
        Used for calculating cosine similarities between Markdown and Code cells.

        Args:
            sparse_matrix: Scipy sparse matrix.

        Returns:
            np.ndarray: Dense matrix of shape (n_samples, n_components).
        """
        return self.svd.transform(sparse_matrix)

    def predict_clusters(self, sparse_matrix):
        """
        Predicts the Functional Cluster ID for code cells.

        Args:
            sparse_matrix: Scipy sparse matrix.

        Returns:
            np.ndarray: Array of cluster labels.
        """
        dense_vecs = self.svd.transform(sparse_matrix)
        return self.kmeans.predict(dense_vecs)
