import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, normalize
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from library.config import Config


class FoldProcessor:
    """
    Handles stateful feature transformations within a cross-validation fold.
    Encapsulates QuantileTransformer for metadata and PCA for auxiliary embeddings
    to ensure no statistical leakage occurs between train and validation sets.
    """

    def __init__(self):
        """
        Initialize the processors with configurations from the library.
        """
        # Metadata processing: Impute missing values then apply RankGauss (QuantileTransform)
        self.meta_imputer = SimpleImputer(strategy="median")
        self.meta_scaler = QuantileTransformer(
            output_distribution="normal", random_state=Config.RANDOM_SEED
        )

        # Auxiliary embedding processing: PCA for dimensionality reduction
        self.aux_pca = PCA(
            n_components=Config.PCA_COMPONENTS, random_state=Config.RANDOM_SEED
        )

        self.numerical_cols = Config.NUMERICAL_COLS

    def _extract_metadata(self, df):
        """
        Extracts numerical columns from the DataFrame based on Config.

        Args:
            df (pd.DataFrame): Input dataframe containing metadata.

        Returns:
            np.ndarray: Array of numerical features.
        """
        # Select only the configured numerical columns
        # If a column is missing (unlikely given data_loader), fill with NaN
        meta_data = df[self.numerical_cols].values
        return meta_data

    def fit(self, df_meta, aux_emb):
        """
        Fits the internal transformers on the training data.

        Args:
            df_meta (pd.DataFrame): Metadata for the training fold.
            aux_emb (np.ndarray): Auxiliary embeddings (MPNet) for the training fold.

        Returns:
            self: Returns the instance itself.
        """
        # 1. Metadata Fitting
        X_meta = self._extract_metadata(df_meta)
        # Fit imputer on training data
        X_meta_imputed = self.meta_imputer.fit_transform(X_meta)
        # Fit scaler on imputed training data
        self.meta_scaler.fit(X_meta_imputed)

        # 2. Auxiliary Embeddings Fitting (PCA)
        # aux_emb is expected to be raw embeddings (N, 768)
        self.aux_pca.fit(aux_emb)

        return self

    def transform(self, df_meta, anchor_emb, aux_emb):
        """
        Applies learned transformations to data to create feature views.

        Args:
            df_meta (pd.DataFrame): Metadata dataframe.
            anchor_emb (np.ndarray): Anchor embeddings (MiniLM).
            aux_emb (np.ndarray): Auxiliary embeddings (MPNet).

        Returns:
            dict: Dictionary containing 'view_A' and 'view_B' feature matrices.
                  - 'view_A': Anchor + Metadata
                  - 'view_B': Anchor + PCA(Aux) + Metadata
        """
        # 1. Metadata Transformation
        X_meta = self._extract_metadata(df_meta)
        # Apply imputation using statistics from training set
        X_meta = self.meta_imputer.transform(X_meta)
        # Apply scaling using statistics from training set
        X_meta_trans = self.meta_scaler.transform(X_meta)

        # 2. Anchor Embedding Transformation
        # Apply L2 Normalization (stateless operation)
        # axis=1 normalizes each sample vector to unit length
        X_anchor_norm = normalize(anchor_emb, norm="l2", axis=1)

        # 3. Auxiliary Embedding Transformation
        # Apply PCA projection using components learned from training set
        X_aux_pca = self.aux_pca.transform(aux_emb)
        # Apply L2 Normalization to the projected embeddings
        X_aux_norm = normalize(X_aux_pca, norm="l2", axis=1)

        # 4. Construct Views
        # View A: The Parsimonious Expert (Safety Anchor)
        # Features: [Normalized MiniLM (384d), Transformed Metadata]
        X_view_A = np.hstack([X_anchor_norm, X_meta_trans])

        # View B: The Augmented Expert (Deep Context)
        # Features: [Normalized MiniLM (384d), Normalized PCA MPNet (50d), Transformed Metadata]
        X_view_B = np.hstack([X_anchor_norm, X_aux_norm, X_meta_trans])

        return {
            "view_A": X_view_A.astype(np.float32),
            "view_B": X_view_B.astype(np.float32),
        }

    def fit_transform(self, df_meta, anchor_emb, aux_emb):
        """
        Convenience method to fit and transform in one step.

        Args:
            df_meta (pd.DataFrame): Metadata for the training fold.
            anchor_emb (np.ndarray): Anchor embeddings for the training fold.
            aux_emb (np.ndarray): Auxiliary embeddings for the training fold.

        Returns:
            dict: Transformed feature views.
        """
        self.fit(df_meta, aux_emb)
        return self.transform(df_meta, anchor_emb, aux_emb)
