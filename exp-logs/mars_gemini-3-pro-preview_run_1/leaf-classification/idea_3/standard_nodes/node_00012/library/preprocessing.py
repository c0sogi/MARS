import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library import config, data_loader


def create_preprocessor(method=config.PT_METHOD, standardize=config.PT_STANDARDIZE):
    """
    Creates a scikit-learn pipeline for feature preprocessing.

    The pipeline consists of:
    1. PowerTransformer (Yeo-Johnson): To Gaussianize features.
    2. StandardScaler: To normalize features to zero mean and unit variance.

    Args:
        method (str): The power transform method (e.g., 'yeo-johnson').
        standardize (bool): Whether to add a StandardScaler step.

    Returns:
        sklearn.pipeline.Pipeline: The constructed preprocessing pipeline.
    """
    steps = []

    # Step 1: Power Transform
    # We set standardize=False to strictly separate the power transform from the scaling step
    # allowing for explicit control via the pipeline structure.
    pt = PowerTransformer(method=method, standardize=False)
    steps.append(("power_transformer", pt))

    # Step 2: Standard Scaler
    # Applied if configured, to ensure numerical stability for LDA
    if standardize:
        scaler = StandardScaler()
        steps.append(("scaler", scaler))

    pipeline = Pipeline(steps)
    return pipeline


def get_preprocessed_data(load_cached_data=True):
    """
    Loads raw data, applies the preprocessing pipeline, and returns transformed datasets.
    Implements caching to avoid re-computing the expensive PowerTransformer step.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed numpy arrays
                                 from the working directory.

    Returns:
        tuple: (X_train_trans, y_train, X_val_trans, y_val, X_test_trans, test_ids, classes)
    """
    # Define cache paths for the *preprocessed* data
    # We use distinct names from the raw data cache to avoid conflicts
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    cache_files = {
        "X_train": os.path.join(cache_dir, "X_train_preprocessed.npy"),
        "X_val": os.path.join(cache_dir, "X_val_preprocessed.npy"),
        "X_test": os.path.join(cache_dir, "X_test_preprocessed.npy"),
    }

    # 1. Check Cache
    if load_cached_data:
        all_exist = all(os.path.exists(path) for path in cache_files.values())
        if all_exist:
            print("Loading preprocessed data from cache...")
            X_train_trans = np.load(cache_files["X_train"])
            X_val_trans = np.load(cache_files["X_val"])
            X_test_trans = np.load(cache_files["X_test"])

            # We still need the targets and IDs, which we can get from the data_loader
            # data_loader has its own caching, so this is fast
            _, y_train, _, y_val, _, test_ids, classes = data_loader.load_data(
                load_cached_data=True
            )

            return (
                X_train_trans,
                y_train,
                X_val_trans,
                y_val,
                X_test_trans,
                test_ids,
                classes,
            )
        else:
            print("Preprocessed cache miss. Processing data from scratch...")

    # 2. Load Raw Data
    # We rely on data_loader to handle the raw CSV parsing and initial feature extraction
    print("Loading raw data for preprocessing...")
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = data_loader.load_data(
        load_cached_data=load_cached_data
    )

    # 3. Create and Fit Preprocessor
    print("Fitting preprocessing pipeline (PowerTransformer + StandardScaler)...")
    preprocessor = create_preprocessor()

    # Fit only on training data to prevent data leakage
    preprocessor.fit(X_train)

    # 4. Transform Data
    print("Transforming datasets...")
    X_train_trans = preprocessor.transform(X_train).astype(np.float64)
    X_val_trans = preprocessor.transform(X_val).astype(np.float64)
    X_test_trans = preprocessor.transform(X_test).astype(np.float64)

    # 5. Save to Cache
    print(f"Saving preprocessed data to {cache_dir}...")
    np.save(cache_files["X_train"], X_train_trans)
    np.save(cache_files["X_val"], X_val_trans)
    np.save(cache_files["X_test"], X_test_trans)

    return X_train_trans, y_train, X_val_trans, y_val, X_test_trans, test_ids, classes
