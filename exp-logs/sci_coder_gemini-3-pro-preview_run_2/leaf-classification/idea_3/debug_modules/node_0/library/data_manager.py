import numpy as np
from sklearn.preprocessing import LabelEncoder
from library.utils import load_and_process_data


def load_and_merge_data(
    metadata_dir="./metadata", cache_dir="./working/idea_3", load_cached_data=True
):
    """
    Loads the dataset by merging training and validation sets, scaling features,
    and encoding target labels.

    Delegates the loading, merging, scaling, and caching logic to library.utils.load_and_process_data.
    Performs LabelEncoding on the target species to convert string labels to integers.

    Args:
        metadata_dir (str): Path to the metadata directory containing train.csv, val.csv, test.csv.
        cache_dir (str): Path to the directory where processed data is cached.
        load_cached_data (bool): If True, attempts to load data from cache_dir.

    Returns:
        X_train (np.ndarray): Scaled feature matrix for training (concatenated Train + Val).
        y_train (np.ndarray): Encoded integer labels for training.
        X_test (np.ndarray): Scaled feature matrix for testing.
        test_ids (np.ndarray): IDs for the test images.
        label_encoder (LabelEncoder): The fitted LabelEncoder object used for transforming labels.
    """
    # Use the provided utility to load, merge, scale, and cache the data.
    # This function implements the caching logic required (check cache -> load or process -> save).
    # It returns y_train as string labels and classes as the sorted unique species names.
    X_train, y_train_str, X_test, test_ids, classes = load_and_process_data(
        metadata_dir=metadata_dir,
        cache_dir=cache_dir,
        load_cached_data=load_cached_data,
    )

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Fit on the unique classes returned by the utility.
    # 'classes' is already sorted by load_and_process_data, ensuring deterministic encoding.
    label_encoder.fit(classes)

    # Transform the training string labels to integers
    y_train = label_encoder.transform(y_train_str)

    return X_train, y_train, X_test, test_ids, label_encoder
