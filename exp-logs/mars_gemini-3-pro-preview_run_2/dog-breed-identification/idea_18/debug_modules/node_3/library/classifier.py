import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
import library.config as config
import library.feature_extractor as feature_extractor


def get_fused_data(stream_config, split, load_cached_data=True):
    """
    Retrieves and fuses embeddings from Global, Standard, and Local views.
    Implements caching for the fused representation.

    Args:
        stream_config (dict): Stream configuration dictionary.
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached fused data.

    Returns:
        X (np.ndarray): Fused feature matrix (N, D*3).
        ids (np.ndarray): Image IDs (N,).
        y (np.ndarray): Labels (N,).
    """
    stream_name = stream_config["name"]

    # Define cache paths for fused data
    cache_prefix = f"{stream_name}_{split}_fused"
    x_path = os.path.join(config.WORKING_DIR, f"{cache_prefix}_X.npy")
    ids_path = os.path.join(config.WORKING_DIR, f"{cache_prefix}_ids.npy")
    y_path = os.path.join(config.WORKING_DIR, f"{cache_prefix}_y.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(x_path)
            and os.path.exists(ids_path)
            and os.path.exists(y_path)
        ):
            print(f"Loading cached fused data for {stream_name} | {split} ...")
            X = np.load(x_path)
            ids = np.load(ids_path, allow_pickle=True)
            y = np.load(y_path)
            return X, ids, y

    # 2. Compute from scratch
    print(f"Generating fused data for {stream_name} | {split} ...")

    views = ["global", "standard", "local"]
    embeddings_list = []
    ids_ref = None
    y_ref = None

    for view in views:
        # Extract features for this view (feature_extractor handles its own caching)
        emb, ids, labels = feature_extractor.extract_features(
            stream_config, split, view, load_cached_data=load_cached_data
        )

        embeddings_list.append(emb)

        # Consistency check for IDs
        if ids_ref is None:
            ids_ref = ids
            y_ref = labels
        else:
            if not np.array_equal(ids_ref, ids):
                raise ValueError(
                    f"ID mismatch detected between views for {stream_name} {split} {view}"
                )

    # Concatenate features (Early Fusion)
    # Shape: (N, D) -> (N, D*3)
    X = np.concatenate(embeddings_list, axis=1)

    # 3. Save to cache
    print(f"Saving fused data to {config.WORKING_DIR}...")
    np.save(x_path, X)
    np.save(ids_path, ids_ref)
    np.save(y_path, y_ref)

    return X, ids_ref, y_ref


def train_stream_head(stream_config, load_cached_data=True):
    """
    Trains a LogisticRegressionCV classifier on the fused training data.
    Saves the trained model to disk.

    Args:
        stream_config (dict): Stream configuration dictionary.
        load_cached_data (bool): Whether to load fused data from cache.

    Returns:
        model (sklearn.linear_model.LogisticRegressionCV): Trained model.
    """
    stream_name = stream_config["name"]
    model_path = os.path.join(config.WORKING_DIR, f"{stream_name}_logreg.joblib")

    # Load fused training data
    X_train, _, y_train = get_fused_data(
        stream_config, "train", load_cached_data=load_cached_data
    )

    print(f"Training Head for {stream_name}...")
    print(f"Input Shape: {X_train.shape}")

    # Initialize Logistic Regression with CV
    # Using multinomial loss for multi-class classification
    clf = LogisticRegressionCV(
        cv=config.ENSEMBLE["cv_folds"],
        max_iter=config.ENSEMBLE["max_iter"],
        n_jobs=config.ENSEMBLE["n_jobs"],
        random_state=config.ENSEMBLE["random_state"],
        multi_class="multinomial",
        scoring="neg_log_loss",
        verbose=0,
    )

    # Fit model
    clf.fit(X_train, y_train)

    # Evaluate on training set (sanity check)
    train_probs = clf.predict_proba(X_train)
    train_loss = log_loss(y_train, train_probs)
    print(f"Stream {stream_name} Training Log Loss: {train_loss}")

    # Save model
    print(f"Saving model to {model_path}...")
    joblib.dump(clf, model_path)

    return clf


def predict_stream(stream_config, split, model=None, load_cached_data=True):
    """
    Generates predictions for a specific split using the trained stream model.

    Args:
        stream_config (dict): Stream configuration dictionary.
        split (str): 'val' or 'test'.
        model (sklearn estimator, optional): Pre-loaded model. If None, loads from disk.
        load_cached_data (bool): Whether to load fused data from cache.

    Returns:
        probs (np.ndarray): Probability matrix (N, C).
        ids (np.ndarray): Image IDs (N,).
        labels (np.ndarray): True labels (N,).
    """
    stream_name = stream_config["name"]

    # Load fused data
    X, ids, y = get_fused_data(stream_config, split, load_cached_data=load_cached_data)

    # Load model if not provided
    if model is None:
        model_path = os.path.join(config.WORKING_DIR, f"{stream_name}_logreg.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model for {stream_name} not found at {model_path}. Train it first."
            )
        print(f"Loading model for {stream_name} from {model_path}...")
        model = joblib.load(model_path)

    print(f"Predicting {stream_name} on {split} set...")
    probs = model.predict_proba(X)

    # If validation, print metric
    if split == "val":
        val_loss = log_loss(y, probs)
        print(f"Stream {stream_name} Validation Log Loss: {val_loss}")

    return probs, ids, y
