import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from library.config import Config


def get_preprocessor():
    """
    Returns a scikit-learn Pipeline that applies Yeo-Johnson power transformation
    followed by Standard Scaling. This is optimal for LDA which assumes Gaussian
    distributed features.
    """
    return Pipeline(
        [
            ("power", PowerTransformer(method=Config.POWER_TRANSFORM_METHOD)),
            ("scaler", StandardScaler()),
        ]
    )


def load_and_preprocess_data(debug=False, load_cached_data=True):
    """
    Loads data, extracts features, generates genus labels, applies preprocessing,
    and handles caching of the processed numpy arrays.

    Args:
        debug (bool): If True, uses a small subset of the data.
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        tuple: (X_train, y_train, genus_train,
                X_val, y_val, genus_val,
                X_test, test_ids,
                species_encoder, genus_encoder)
    """
    # Define cache file paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "genus_train": os.path.join(cache_dir, "genus_train.npy"),
        "X_val": os.path.join(cache_dir, "X_val.npy"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "genus_val": os.path.join(cache_dir, "genus_val.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
        "species_classes": os.path.join(cache_dir, "species_classes.npy"),
        "genus_classes": os.path.join(cache_dir, "genus_classes.npy"),
    }

    # 1. Try to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(f) for f in files.values())
        if all_exist:
            print("Loading preprocessed data from cache...")
            X_train = np.load(files["X_train"])
            y_train = np.load(files["y_train"])
            genus_train = np.load(files["genus_train"])
            X_val = np.load(files["X_val"])
            y_val = np.load(files["y_val"])
            genus_val = np.load(files["genus_val"])
            X_test = np.load(files["X_test"])
            test_ids = np.load(files["test_ids"])

            # Reconstruct encoders
            species_encoder = LabelEncoder()
            species_encoder.classes_ = np.load(
                files["species_classes"], allow_pickle=True
            )

            genus_encoder = LabelEncoder()
            genus_encoder.classes_ = np.load(files["genus_classes"], allow_pickle=True)

            return (
                X_train,
                y_train,
                genus_train,
                X_val,
                y_val,
                genus_val,
                X_test,
                test_ids,
                species_encoder,
                genus_encoder,
            )
        else:
            print("Cache miss. Processing data from scratch...")

    # 2. Load Raw Data
    print("Loading raw metadata...")
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    if debug:
        print("Debug mode: Subsampling data...")
        # Cite debug_lesson_1: Filter Classes, Don't Just Slice Rows
        # Filter to top 5 species to ensure density, then slice.
        top_species = df_train[Config.TARGET_COL].value_counts().nlargest(5).index
        df_train = df_train[df_train[Config.TARGET_COL].isin(top_species)].head(50)

        # Ensure validation set only contains species present in the training subset
        present_species = df_train[Config.TARGET_COL].unique()
        df_val = df_val[df_val[Config.TARGET_COL].isin(present_species)].head(50)

        df_test = df_test.head(50)

    # 3. Feature Extraction
    feature_cols = Config.get_feature_columns()

    # Extract features
    X_train_raw = df_train[feature_cols].values
    X_val_raw = df_val[feature_cols].values
    X_test_raw = df_test[feature_cols].values

    test_ids = df_test[Config.ID_COL].values

    # 4. Target Engineering (Species & Genus)
    y_train_raw = df_train[Config.TARGET_COL].values
    y_val_raw = df_val[Config.TARGET_COL].values

    # Derive Genus
    genus_train_raw = (
        df_train[Config.TARGET_COL].apply(Config.get_genus_from_species).values
    )
    genus_val_raw = (
        df_val[Config.TARGET_COL].apply(Config.get_genus_from_species).values
    )

    # 5. Preprocessing (Transformation + Scaling)
    print("Fitting preprocessor (Yeo-Johnson + StandardScaler)...")
    preprocessor = get_preprocessor()

    # Fit on training data only
    X_train = preprocessor.fit_transform(X_train_raw)

    # Transform validation and test
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    # 6. Label Encoding
    print("Encoding targets...")
    species_encoder = LabelEncoder()
    y_train = species_encoder.fit_transform(y_train_raw)
    y_val = species_encoder.transform(y_val_raw)

    genus_encoder = LabelEncoder()
    genus_train = genus_encoder.fit_transform(genus_train_raw)
    genus_val = genus_encoder.transform(genus_val_raw)

    # 7. Cache Results
    print(f"Caching processed data to {cache_dir}...")
    np.save(files["X_train"], X_train)
    np.save(files["y_train"], y_train)
    np.save(files["genus_train"], genus_train)

    np.save(files["X_val"], X_val)
    np.save(files["y_val"], y_val)
    np.save(files["genus_val"], genus_val)

    np.save(files["X_test"], X_test)
    np.save(files["test_ids"], test_ids)

    np.save(files["species_classes"], species_encoder.classes_)
    np.save(files["genus_classes"], genus_encoder.classes_)

    return (
        X_train,
        y_train,
        genus_train,
        X_val,
        y_val,
        genus_val,
        X_test,
        test_ids,
        species_encoder,
        genus_encoder,
    )
