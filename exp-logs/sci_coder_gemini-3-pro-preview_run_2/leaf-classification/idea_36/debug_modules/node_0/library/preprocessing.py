import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from library.config import (
    BASIS_CONFIGS,
    VIEW_CONFIGS,
    CACHE_DIR,
    FLOAT_PRECISION,
)


class GaussianBasisFactory:
    """
    Manages the creation, fitting, and application of Gaussian Basis transformations.
    Projects data into multiple latent spaces (Parametric, Coarse-Quantile, Fine-Quantile)
    to be used by the ensemble experts.
    """

    def __init__(self):
        self.basis_configs = BASIS_CONFIGS
        self.view_configs = VIEW_CONFIGS
        # Structure: self.transformers[basis_name][view_name] = fitted_transformer
        self.transformers = {}

    def _build_transformer(self, config):
        """
        Factory method to instantiate sklearn transformers based on config.
        """
        t_type = config.get("type")
        if t_type == "power":
            return PowerTransformer(
                method=config["method"], standardize=config["standardize"]
            )
        elif t_type == "quantile":
            return QuantileTransformer(
                n_quantiles=config["n_quantiles"],
                output_distribution=config["output_distribution"],
                random_state=config["random_state"],
            )
        else:
            raise ValueError(f"Unknown transformer type: {t_type}")

    def fit(self, X_train_views):
        """
        Fits all basis transformers on the training views.

        Args:
            X_train_views (dict): Dictionary of {view_name: DataFrame/Array} for training data.
        """
        for basis_name, basis_cfg in self.basis_configs.items():
            self.transformers[basis_name] = {}
            for view_name in self.view_configs.keys():
                # Get data for this view
                X = X_train_views[view_name]

                # Instantiate and fit
                transformer = self._build_transformer(basis_cfg)

                # Ensure float64 input
                X_f64 = np.array(X, dtype=FLOAT_PRECISION)

                transformer.fit(X_f64)
                self.transformers[basis_name][view_name] = transformer

    def transform(self, X_views):
        """
        Transforms a set of views using the fitted transformers.

        Args:
            X_views (dict): Dictionary of {view_name: DataFrame/Array}.

        Returns:
            dict: Nested dictionary {basis_name: {view_name: transformed_array}}.
        """
        transformed_data = {}
        for basis_name in self.basis_configs.keys():
            transformed_data[basis_name] = {}
            for view_name in self.view_configs.keys():
                if (
                    basis_name not in self.transformers
                    or view_name not in self.transformers[basis_name]
                ):
                    raise RuntimeError(
                        f"Transformer for {basis_name}/{view_name} not fitted."
                    )

                transformer = self.transformers[basis_name][view_name]
                X = X_views[view_name]
                X_f64 = np.array(X, dtype=FLOAT_PRECISION)

                # Transform and enforce precision
                X_trans = transformer.transform(X_f64).astype(FLOAT_PRECISION)
                transformed_data[basis_name][view_name] = X_trans

        return transformed_data

    def _save_cache(self, data_dict, filename):
        """
        Saves a nested dictionary of transformed data to an npz file.
        Flattens keys to 'basis__view'.
        """
        flat_dict = {}
        for basis_name, views in data_dict.items():
            for view_name, arr in views.items():
                key = f"{basis_name}__{view_name}"
                flat_dict[key] = arr

        filepath = os.path.join(CACHE_DIR, filename)
        np.savez(filepath, **flat_dict)

    def _load_cache(self, filename):
        """
        Loads transformed data from an npz file and reconstructs nested structure.
        """
        filepath = os.path.join(CACHE_DIR, filename)
        if not os.path.exists(filepath):
            return None

        loaded = np.load(filepath)
        nested_dict = {}

        for key in loaded.files:
            basis_name, view_name = key.split("__")
            if basis_name not in nested_dict:
                nested_dict[basis_name] = {}
            nested_dict[basis_name][view_name] = loaded[key].astype(FLOAT_PRECISION)

        return nested_dict

    def process(self, views_train, views_val, views_test, load_cached_data=True):
        """
        Orchestrates the fitting, transformation, and caching process.

        Args:
            views_train (dict): Training views.
            views_val (dict): Validation views.
            views_test (dict): Test views.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (processed_train, processed_val, processed_test)
                   Each is a dict: {basis_name: {view_name: X_transformed}}
        """
        # Define cache filenames
        files = {
            "train": "transformed_train.npz",
            "val": "transformed_val.npz",
            "test": "transformed_test.npz",
        }

        # 1. Try Loading Cache
        if load_cached_data:
            data_train = self._load_cache(files["train"])
            data_val = self._load_cache(files["val"])
            data_test = self._load_cache(files["test"])

            if (
                data_train is not None
                and data_val is not None
                and data_test is not None
            ):
                # print("GaussianBasisFactory: Loaded transformed features from cache.")
                return data_train, data_val, data_test

        # 2. Compute from Scratch
        # print("GaussianBasisFactory: Computing features from scratch...")

        # Fit on Train ONLY
        self.fit(views_train)

        # Transform all splits
        data_train = self.transform(views_train)
        data_val = self.transform(views_val)
        data_test = self.transform(views_test)

        # 3. Save Cache
        # Ensure cache directory exists (redundant with config but safe)
        os.makedirs(CACHE_DIR, exist_ok=True)

        self._save_cache(data_train, files["train"])
        self._save_cache(data_val, files["val"])
        self._save_cache(data_test, files["test"])

        # print("GaussianBasisFactory: Features computed and cached.")

        return data_train, data_val, data_test
