import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.decomposition import PCA
from library.config import Config


class TransductivePipeline:
    """
    Implements a Transductive Unsupervised Manifold Learning pipeline.

    This pipeline fits feature transformations on the union of training, validation,
    and test data to leverage the global correlation structure and stabilize
    covariance estimation (Transductive Learning).

    Pipeline steps:
    1. Yeo-Johnson Power Transformation (Gaussianization)
    2. Standard Scaling (Z-score normalization)
    3. PCA (Orthogonalization and Denoising)
    """

    def __init__(self):
        self.pca_variance = Config.PCA_VARIANCE
        self.seed = Config.SEED
        self.working_dir = Config.WORKING_DIR

    def fit_transform_combined(self, X_train, X_val, X_test, load_cached_data=True):
        """
        Fits the pipeline on the combined dataset and transforms inputs.

        Args:
            X_train (pd.DataFrame): Training features.
            X_val (pd.DataFrame): Validation features.
            X_test (pd.DataFrame): Test features.
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            tuple: (X_train_trans, X_val_trans, X_test_trans) as numpy arrays.
        """
        # Define cache paths
        cache_paths = {
            "train": os.path.join(self.working_dir, "X_train_transformed.npy"),
            "val": os.path.join(self.working_dir, "X_val_transformed.npy"),
            "test": os.path.join(self.working_dir, "X_test_transformed.npy"),
        }

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

        # 1. Attempt to Load from Cache
        if load_cached_data:
            if all(os.path.exists(p) for p in cache_paths.values()):
                print(
                    f"Loading transformed datasets from cache at {self.working_dir}..."
                )
                try:
                    X_train_trans = np.load(cache_paths["train"])
                    X_val_trans = np.load(cache_paths["val"])
                    X_test_trans = np.load(cache_paths["test"])
                    return X_train_trans, X_val_trans, X_test_trans
                except Exception as e:
                    print(f"Cache load failed: {e}. Recomputing...")
            else:
                print("Cache incomplete. Recomputing...")
        else:
            print("Ignoring cache. Recomputing...")

        # 2. Prepare Data for Transductive Fitting
        # We combine Train, Val, and Test to capture the full manifold structure.
        # X_train and X_val come from the original 'train.csv', X_test from 'test.csv'.
        # Combining them reconstructs the full available feature space.
        print("Concatenating datasets for transductive learning...")
        X_combined = pd.concat([X_train, X_val, X_test], axis=0, ignore_index=True)

        # 3. Define and Fit Pipeline
        print(
            "Fitting Transductive Pipeline (PowerTransformer -> StandardScaler -> PCA)..."
        )

        # Step A: Power Transformer (Yeo-Johnson)
        # Forces features to be more Gaussian-like, satisfying LDA assumptions.
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        pt.fit(X_combined)

        # Step B: Standard Scaler
        # Centers and scales the Gaussian-ized data.
        # We transform the combined data first to pass to StandardScaler
        X_combined_pt = pt.transform(X_combined)
        ss = StandardScaler()
        ss.fit(X_combined_pt)

        # Step C: PCA
        # Orthogonalizes features and removes noise.
        X_combined_ss = ss.transform(X_combined_pt)
        pca = PCA(n_components=self.pca_variance, random_state=self.seed)
        pca.fit(X_combined_ss)

        print(
            f"PCA retained {pca.n_components_} components to explain {self.pca_variance*100}% variance."
        )

        # 4. Transform Individual Sets
        # We must apply the exact same sequence of transformations
        print("Transforming individual datasets...")

        def transform_subset(X_subset):
            x_pt = pt.transform(X_subset)
            x_ss = ss.transform(x_pt)
            x_pca = pca.transform(x_ss)
            return x_pca.astype(np.float32)  # Optimize memory

        X_train_trans = transform_subset(X_train)
        X_val_trans = transform_subset(X_val)
        X_test_trans = transform_subset(X_test)

        # 5. Save to Cache
        print(f"Saving transformed datasets to {self.working_dir}...")
        np.save(cache_paths["train"], X_train_trans)
        np.save(cache_paths["val"], X_val_trans)
        np.save(cache_paths["test"], X_test_trans)

        return X_train_trans, X_val_trans, X_test_trans
