import pandas as pd
from library import config


def get_ordered_features():
    """
    Cite solution_lesson_node_00029: Always explicitly define and enforce feature column order.
    Implicit ordering (e.g., via df.columns) can introduce numerical noise that prevents
    the model from reaching the optimal loss floor.
    """
    feature_groups = ["margin", "shape", "texture"]
    features = []
    for group in feature_groups:
        for i in range(1, 65):
            features.append(f"{group}_{i}")
    return features


def _split_features_target_ids(df, is_test=False):
    """
    Helper function to separate features, target, and IDs from the dataframe.
    Enforces explicit feature ordering.
    """
    # Cite solution_lesson_node_00029
    feature_cols = get_ordered_features()

    # Extract Features
    X = df[feature_cols].copy()

    # Extract IDs
    ids = df["id"].copy()

    if is_test:
        return X, ids
    else:
        # Verify target exists
        if "species" not in df.columns:
            raise ValueError("Column 'species' missing from non-test dataset.")

        # Extract Target
        y = df["species"].copy()
        return X, y, ids


def load_dataset(
    load_cached_data=True,
    debug_mode=config.DEBUG_MODE,
    debug_size=config.DEBUG_SUBSET_SIZE,
):
    """
    Loads the training, validation, and test datasets directly from metadata.
    Cite solution_lesson_node_00028: Avoid adding heteroscedastic features (augmentation).
    """
    print("Loading metadata CSVs...")
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    df_val = pd.read_csv(config.VAL_DATA_PATH)
    df_test = pd.read_csv(config.TEST_DATA_PATH)

    # Apply Debugging Subsampling if requested
    if debug_mode:
        print(
            f"Debug Mode: Filtering to common classes and sampling {debug_size} rows."
        )

        # Cite debug_lesson_1: Filter Classes, Don't Just Slice Rows.
        train_species = set(df_train["species"].unique())
        val_species = set(df_val["species"].unique())
        common_species = sorted(list(train_species.intersection(val_species)))

        if not common_species:
            common_species = sorted(list(train_species))

        selected_species = common_species[:5]

        df_train = df_train[df_train["species"].isin(selected_species)]
        df_train = df_train.iloc[:debug_size]

        active_train_species = df_train["species"].unique()

        df_val = df_val[df_val["species"].isin(active_train_species)]
        df_val = df_val.iloc[:debug_size]

        df_test = df_test.iloc[:debug_size]

    # Structure the data into (X, y, id) tuples
    train_data = _split_features_target_ids(df_train, is_test=False)
    val_data = _split_features_target_ids(df_val, is_test=False)
    test_data = _split_features_target_ids(df_test, is_test=True)

    return train_data, val_data, test_data
