from library.preprocessing import load_and_preprocess_data


def load_data(debug: bool = False, load_cached_data: bool = True):
    """
    Loads and preprocesses the leaf dataset for the Taxonomy-Regularized Hierarchical LDA.

    This function acts as a facade for the library's preprocessing pipeline. It ensures
    that data is loaded from the metadata CSVs, features are extracted, the taxonomic
    hierarchy (Genus) is derived from the species labels, and the data is transformed
    (Yeo-Johnson + Standard Scaler) and cached.

    Args:
        debug (bool): If True, loads a small subset (e.g., 50 rows) of the data
                      for rapid debugging and testing.
        load_cached_data (bool): If True, attempts to load pre-processed numpy arrays
                                 from the cache directory to save time. If False or
                                 if the cache is missing, data is processed from scratch.

    Returns:
        tuple: A tuple containing the following elements:
            - X_train (np.ndarray): Transformed training feature matrix (N_train, 192).
            - y_train (np.ndarray): Encoded training species labels.
            - X_val (np.ndarray): Transformed validation feature matrix (N_val, 192).
            - y_val (np.ndarray): Encoded validation species labels.
            - X_test (np.ndarray): Transformed test feature matrix (N_test, 192).
            - test_ids (np.ndarray): IDs for the test set images.
            - species_encoder (LabelEncoder): Fitted encoder mapping integers to species names.
    """
    # The library function 'load_and_preprocess_data' already implements:
    # 1. Metadata loading from ./metadata/
    # 2. Feature extraction
    # 3. Preprocessing (PowerTransform + Scaling)
    # 4. Strict caching logic (checking file existence, saving to ./working/idea_4/)
    return load_and_preprocess_data(debug=debug, load_cached_data=load_cached_data)
