import pandas as pd
import numpy as np
import os
import gc
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    RANDOM_SEED,
    DEBUG_SAMPLE_SIZE,
)
from library.features import GraphFeatureEngine, DataProcessor
from library.utils import timer, reduce_mem_usage


class DataManager:
    """
    Manages data loading, feature generation, and dataset creation.
    Orchestrates the GraphFeatureEngine and DataProcessor to produce
    ready-to-use feature matrices for training and inference.
    """

    def __init__(self):
        self.feature_engine = GraphFeatureEngine()
        self.node_features = None
        self.edges = None

    def load_structures(self, load_cached_data=True):
        """
        Loads and processes molecular structures to generate atom-level features.
        Uses GraphFeatureEngine to compute Level 1 (Local Field) and Level 2 (Message Passing) features.

        Args:
            load_cached_data (bool): If True, attempts to load pre-computed features from parquet cache.

        Returns:
            tuple: (node_features, edges)
        """
        # Avoid reloading if already in memory
        if self.node_features is not None and self.edges is not None:
            return self.node_features, self.edges

        self.node_features, self.edges = self.feature_engine.process_structures(
            load_cached_data=load_cached_data
        )
        return self.node_features, self.edges

    def get_train_data(self, load_cached_data=True, debug_mode=False):
        """
        Generates the training and validation datasets.
        Merges atom-level features onto coupling pairs and calculates pairwise geometric features.

        Args:
            load_cached_data (bool): Whether to use cached structure features.
            debug_mode (bool): If True, subsamples the metadata for rapid prototyping.

        Returns:
            tuple: (X_train, y_train, X_val, y_val)
                   X_train/X_val contain features (including 'type' for stratification).
                   y_train/y_val contain the target scalar_coupling_constant.
        """
        # Ensure node features are loaded
        node_features, edges = self.load_structures(load_cached_data=load_cached_data)
        processor = DataProcessor(node_features, edges)

        # Load Metadata
        print(f"Loading training metadata from {TRAIN_METADATA_PATH}...")
        train_meta = pd.read_csv(TRAIN_METADATA_PATH)
        print(f"Loading validation metadata from {VAL_METADATA_PATH}...")
        val_meta = pd.read_csv(VAL_METADATA_PATH)

        # Debug Sampling
        if debug_mode:
            print(f"Debug Mode: Sampling {DEBUG_SAMPLE_SIZE} rows from metadata...")
            # Sample randomly; note that this might break molecule grouping if not careful,
            # but for debugging pipeline flow it is acceptable.
            # train/val splits are already disjoint by molecule in the metadata files.
            train_meta = train_meta.sample(
                n=min(DEBUG_SAMPLE_SIZE, len(train_meta)), random_state=RANDOM_SEED
            ).reset_index(drop=True)
            val_meta = val_meta.sample(
                n=min(DEBUG_SAMPLE_SIZE, len(val_meta)), random_state=RANDOM_SEED
            ).reset_index(drop=True)

        # Create Datasets
        # DataProcessor.create_dataset handles the merge and pairwise distance calcs (1/d, 1/d^2, etc.)
        with timer("Training Dataset Creation"):
            X_train, y_train = processor.create_dataset(train_meta, is_train=True)

        with timer("Validation Dataset Creation"):
            X_val, y_val = processor.create_dataset(val_meta, is_train=True)

        # Memory Optimization
        print("Optimizing memory usage for training data...")
        X_train = reduce_mem_usage(X_train)
        X_val = reduce_mem_usage(X_val)

        return X_train, y_train, X_val, y_val

    def get_test_data(self, load_cached_data=True):
        """
        Generates the test dataset.

        Args:
            load_cached_data (bool): Whether to use cached structure features.

        Returns:
            tuple: (X_test, test_ids)
                   X_test contains features.
                   test_ids is a Series containing the 'id' column for submission mapping.
        """
        # Ensure node features are loaded
        node_features, edges = self.load_structures(load_cached_data=load_cached_data)
        processor = DataProcessor(node_features, edges)

        # Load Metadata
        print(f"Loading test metadata from {TEST_METADATA_PATH}...")
        test_meta = pd.read_csv(TEST_METADATA_PATH)

        # Create Dataset
        with timer("Test Dataset Creation"):
            X_test, _ = processor.create_dataset(test_meta, is_train=False)

        # Memory Optimization
        print("Optimizing memory usage for test data...")
        X_test = reduce_mem_usage(X_test)

        # Return features and the corresponding IDs (needed for submission)
        return X_test, test_meta["id"]
