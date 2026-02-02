import os
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, normalize
from library.config import Config


class FoldPipeline:
    """
    Manages the in-fold feature transformation logic to prevent data leakage.
    Encapsulates PCA for auxiliary embeddings, QuantileTransformer for metadata,
    and L2 normalization for anchor embeddings.
    """

    def __init__(self):
        """
        Initializes the pipeline components based on Config settings.
        """
        # Initialize PCA for Auxiliary Title (Deep Hook)
        self.pca_title = PCA(
            n_components=Config.AUX_TITLE_PCA_COMPONENTS, random_state=Config.SEED
        )

        # Initialize PCA for Auxiliary Body (Deep Narrative)
        self.pca_body = PCA(
            n_components=Config.AUX_BODY_PCA_COMPONENTS, random_state=Config.SEED
        )

        # Initialize RankGauss Scaler for Metadata
        self.scaler = QuantileTransformer(
            output_distribution="normal", random_state=Config.SEED
        )

        self.is_fitted = False

    def fit(self, features_dict: dict):
        """
        Fits the stateful transformers (PCA, Scaler) on the provided training data.

        Args:
            features_dict (dict): Dictionary containing the training features.
                Expected keys:
                    - 'aux_title': np.ndarray (MPNet embeddings for title)
                    - 'aux_body': np.ndarray (MPNet embeddings for body)
                    - 'meta': pd.DataFrame (Numerical metadata)
        """
        # 1. Fit PCA on Auxiliary Title
        if "aux_title" in features_dict:
            self.pca_title.fit(features_dict["aux_title"])

        # 2. Fit PCA on Auxiliary Body
        if "aux_body" in features_dict:
            self.pca_body.fit(features_dict["aux_body"])

        # 3. Fit Scaler on Metadata
        if "meta" in features_dict:
            meta_df = features_dict["meta"]
            # Ensure we only use the configured numerical features
            meta_data = meta_df[Config.NUMERICAL_FEATURES].values
            self.scaler.fit(meta_data)

        self.is_fitted = True
        return self

    def transform(self, features_dict: dict) -> np.ndarray:
        """
        Applies transformations to the data and concatenates views.

        Args:
            features_dict (dict): Dictionary containing the features to transform.
                Expected keys:
                    - 'anchor_title': np.ndarray (MiniLM embeddings for title)
                    - 'anchor_body': np.ndarray (MiniLM embeddings for body)
                    - 'aux_title': np.ndarray (MPNet embeddings for title)
                    - 'aux_body': np.ndarray (MPNet embeddings for body)
                    - 'meta': pd.DataFrame (Numerical metadata)

        Returns:
            np.ndarray: The concatenated feature matrix.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform.")

        feature_list = []

        # ---------------------------------------------------------
        # View 1: Title Anchor (384 dims) -> L2 Norm
        # ---------------------------------------------------------
        if "anchor_title" in features_dict:
            # Normalize directly
            f_anchor_title = normalize(features_dict["anchor_title"], norm="l2")
            feature_list.append(f_anchor_title)

        # ---------------------------------------------------------
        # View 2: Body Anchor (384 dims) -> L2 Norm
        # ---------------------------------------------------------
        if "anchor_body" in features_dict:
            # Normalize directly
            f_anchor_body = normalize(features_dict["anchor_body"], norm="l2")
            feature_list.append(f_anchor_body)

        # ---------------------------------------------------------
        # View 3: Deep Hook (20 dims) -> PCA -> L2 Norm
        # ---------------------------------------------------------
        if "aux_title" in features_dict:
            # Project then normalize
            f_aux_title = self.pca_title.transform(features_dict["aux_title"])
            f_aux_title = normalize(f_aux_title, norm="l2")
            feature_list.append(f_aux_title)

        # ---------------------------------------------------------
        # View 4: Deep Narrative (30 dims) -> PCA -> L2 Norm
        # ---------------------------------------------------------
        if "aux_body" in features_dict:
            # Project then normalize
            f_aux_body = self.pca_body.transform(features_dict["aux_body"])
            f_aux_body = normalize(f_aux_body, norm="l2")
            feature_list.append(f_aux_body)

        # ---------------------------------------------------------
        # View 5: Robust Metadata (~10 dims) -> RankGauss
        # ---------------------------------------------------------
        if "meta" in features_dict:
            meta_df = features_dict["meta"]
            meta_data = meta_df[Config.NUMERICAL_FEATURES].values
            f_meta = self.scaler.transform(meta_data)
            feature_list.append(f_meta)

        # ---------------------------------------------------------
        # Fusion
        # ---------------------------------------------------------
        if not feature_list:
            raise ValueError("No features provided to transform.")

        # Concatenate all views horizontally
        X_combined = np.hstack(feature_list)
        return X_combined

    def save(self, path: str):
        """
        Saves the fitted pipeline to disk.

        Args:
            path (str): File path to save the joblib object.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        """
        Loads a fitted pipeline from disk.

        Args:
            path (str): File path to load the joblib object from.

        Returns:
            FoldPipeline: The loaded pipeline instance.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pipeline file not found at {path}")
        return joblib.load(path)
