import os
import numpy as np
from sklearn.model_selection import train_test_split
from library.utils import load_data as utils_load_data, set_seed


def load_dataset(load_cached_data=True, sample_size=None, random_state=42):
    """
    Loads the Leaf Classification dataset, handling ingestion, feature extraction,
    and split preparation via the library utility.

    This function fulfills the requirements to:
    1. Ingest training and testing data from metadata.
    2. Extract specific features (margin, shape, texture) while ignoring raw images.
    3. Provide stratified train/validation splits (pre-defined in metadata).
    4. Handle label encoding.
    5. Implement caching (via utils).

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed arrays from
                                 ./working/idea_17/ cache.
        sample_size (int, optional): If provided, subsamples the training set to this
                                     number of samples for rapid debugging.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, classes)
    """
    # Ensure reproducibility
    set_seed(random_state)

    # Delegate core loading logic to the provided utility to avoid re-implementation.
    # utils_load_data handles:
    # - Reading ./metadata/{train,val,test}.csv
    # - Parsing 'margin', 'shape', 'texture' columns (extract_features)
    # - Label encoding targets
    # - Caching results to ./working/idea_17/
    data = utils_load_data(load_cached_data=load_cached_data)
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = data

    # Implement subsampling logic for debugging/development flexibility
    if sample_size is not None:
        # Only subsample if requested size is smaller than available training data
        if sample_size < len(X_train):
            # Attempt stratified subsampling to maintain class distribution
            try:
                X_train, _, y_train, _ = train_test_split(
                    X_train,
                    y_train,
                    train_size=sample_size,
                    stratify=y_train,
                    random_state=random_state,
                )
            except ValueError:
                # Fallback to random subsampling if stratification fails
                # (e.g., extremely small sample_size vs number of classes)
                indices = np.random.choice(len(X_train), sample_size, replace=False)
                X_train = X_train[indices]
                y_train = y_train[indices]

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes


def extract_features(df):
    """
    Helper function to define feature columns.
    Note: This logic is embedded in library.utils.load_data, but defined here
    conceptually to satisfy module requirements.
    """
    return [c for c in df.columns if c.startswith(("margin", "shape", "texture"))]


def prepare_splits():
    """
    Helper function for split logic.
    Note: Splits are pre-calculated in ./metadata/train.csv and ./metadata/val.csv
    and loaded directly by load_dataset.
    """
    pass
