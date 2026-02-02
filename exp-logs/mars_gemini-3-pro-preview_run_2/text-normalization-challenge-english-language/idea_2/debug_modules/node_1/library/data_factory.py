import os
import pandas as pd
import numpy as np
from library.config import Config
from library.features import FeaturePipeline
from library.dictionary import NormalizationDictionary
from library.utils import set_seed


class DataFactory:
    """
    Orchestrates data loading, preparation, and feature engineering for the
    Text Normalization task. Acts as a facade over FeaturePipeline and
    NormalizationDictionary.
    """

    def __init__(self):
        """
        Initialize the DataFactory.
        Sets global random seed and initializes helper classes.
        """
        set_seed(Config.SEED)
        self.feature_pipeline = FeaturePipeline()
        self.dictionary = NormalizationDictionary()

    def prepare_training_data(self, load_cached_data=True):
        """
        Prepares all necessary data for the training phase.

        Steps:
        1. Builds (or loads) the Normalization Dictionary from training data.
        2. Generates (or loads) Training Features (X_train, y_train).
           - Note: The FeaturePipeline handles downsampling of the 'PLAIN' class.
        3. Generates (or loads) Validation Features (X_val, y_val).

        Args:
            load_cached_data (bool): If True, attempts to load processed data from disk.
                                     If False or load fails, re-computes everything.

        Returns:
            tuple: ((X_train, y_train), (X_val, y_val))
        """
        print("DataFactory: Preparing training data...")

        # 1. Build Normalization Dictionary
        # The dictionary is essential for the inference logic (Idea 2),
        # mapping (class, token) -> normalized_text.
        self.dictionary.build(load_cached_data=load_cached_data)

        # 2. Get Training Data
        # FeaturePipeline.get_train_data handles:
        # - Loading raw CSV
        # - Downsampling PLAIN class (balancing)
        # - Feature extraction (N-grams, context, orthography)
        # - Label Encoding
        # - Caching to Parquet
        X_train, y_train = self.feature_pipeline.get_train_data(
            load_cached_data=load_cached_data
        )

        # 3. Get Validation Data
        X_val, y_val = self.feature_pipeline.get_val_data(
            load_cached_data=load_cached_data
        )

        return (X_train, y_train), (X_val, y_val)

    def prepare_test_data(self, load_cached_data=True):
        """
        Prepares data for the inference/submission phase.

        Args:
            load_cached_data (bool): If True, attempts to load processed data from disk.

        Returns:
            tuple: (X_test, test_metadata)
                - X_test: Feature matrix for the test set.
                - test_metadata: DataFrame containing 'id' and 'before' columns
                  needed for submission and dictionary lookup.
        """
        print("DataFactory: Preparing test data...")
        return self.feature_pipeline.get_test_data(load_cached_data=load_cached_data)

    def get_normalization_dictionary(self):
        """
        Returns the NormalizationDictionary instance.
        Ensures the dictionary mapping is loaded into memory.

        Returns:
            NormalizationDictionary: The ready-to-use dictionary object.
        """
        # If mapping is empty, try to load it
        if not self.dictionary.mapping:
            try:
                self.dictionary.load()
            except FileNotFoundError:
                print("DataFactory: Dictionary not found on disk. Building now...")
                self.dictionary.build(load_cached_data=False)

        return self.dictionary
