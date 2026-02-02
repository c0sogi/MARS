import os
import pandas as pd
import numpy as np
from library.utils import set_seed
from library.preprocessing import get_preprocessed_data


class LeafDataManager:
    """
    Manages data loading and preparation for the leaf classification task.

    This class serves as an interface to the underlying preprocessing logic provided
    in library.preprocessing. It facilitates access to training, validation, and
    test datasets with appropriate preprocessing (raw for trees, transformed for
    linear/kernel models) as required by the Heterogeneous Kernel-Linear-Tree Ensemble.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the LeafDataManager.

        Args:
            seed (int): Random seed for reproducibility.
        """
        set_seed(seed)
        self.seed = seed

    def get_train_data(self, model_type: str = "tree", load_cached_data: bool = True):
        """
        Retrieves the training dataset (features and target labels).

        Args:
            model_type (str): The type of model to retrieve data for.
                              Options: 'tree' (raw features) or 'linear_kernel' (transformed).
            load_cached_data (bool): If True, attempts to load pre-processed data from cache.

        Returns:
            tuple: (X_train, y_train) where X_train is the feature matrix and y_train are the labels.
        """
        (X_train, y_train), _, _, _ = get_preprocessed_data(
            model_type=model_type, load_cached_data=load_cached_data
        )
        return X_train, y_train

    def get_val_data(self, model_type: str = "tree", load_cached_data: bool = True):
        """
        Retrieves the validation dataset (features and target labels).

        Args:
            model_type (str): The type of model to retrieve data for.
                              Options: 'tree' or 'linear_kernel'.
            load_cached_data (bool): If True, attempts to load pre-processed data from cache.

        Returns:
            tuple: (X_val, y_val) where X_val is the feature matrix and y_val are the labels.
        """
        _, (X_val, y_val), _, _ = get_preprocessed_data(
            model_type=model_type, load_cached_data=load_cached_data
        )
        return X_val, y_val

    def get_test_data(self, model_type: str = "tree", load_cached_data: bool = True):
        """
        Retrieves the test dataset (features and image IDs).

        Args:
            model_type (str): The type of model to retrieve data for.
                              Options: 'tree' or 'linear_kernel'.
            load_cached_data (bool): If True, attempts to load pre-processed data from cache.

        Returns:
            tuple: (X_test, test_ids) where X_test is the feature matrix and test_ids are the image IDs.
        """
        _, _, (X_test, test_ids), _ = get_preprocessed_data(
            model_type=model_type, load_cached_data=load_cached_data
        )
        return X_test, test_ids

    def get_classes(self, load_cached_data: bool = True):
        """
        Retrieves the list of unique class names (species).

        Args:
            load_cached_data (bool): If True, attempts to load data from cache.

        Returns:
            np.ndarray: Array of class names corresponding to the encoded labels.
        """
        # Class names are independent of feature preprocessing, so we can use default 'tree'
        _, _, _, classes = get_preprocessed_data(
            model_type="tree", load_cached_data=load_cached_data
        )
        return classes

    def get_submission_format(self):
        """
        Loads the sample submission file to determine the required output format.

        Returns:
            pd.DataFrame: The sample submission dataframe containing 'id' and species columns.
        """
        return pd.read_csv("./input/sample_submission.csv")
