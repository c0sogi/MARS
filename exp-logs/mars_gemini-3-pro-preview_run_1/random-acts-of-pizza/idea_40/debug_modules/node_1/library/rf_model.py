import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library import config, utils
from library.features import FeatureEngineer


def train_rf(X_train, y_train, X_val, y_val, params=None):
    """
    Trains the Random Forest Classifier and evaluates on validation data.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.
        params (dict, optional): Hyperparameters for RandomForestClassifier.
                                 Defaults to config.RF_PARAMS if None.

    Returns:
        tuple: (trained_model, val_auc_score)
    """
    if params is None:
        params = config.RF_PARAMS

    print("Initializing Random Forest Classifier...")
    # Initialize model with config parameters
    rf_model = RandomForestClassifier(**params)

    print(
        f"Training Random Forest on {X_train.shape[0]} samples with {X_train.shape[1]} features..."
    )
    rf_model.fit(X_train, y_train)

    print("Evaluating on validation set...")
    # Predict probabilities for the positive class
    val_probs = rf_model.predict_proba(X_val)[:, 1]

    # Calculate AUC
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"Random Forest Validation AUC: {val_auc}")

    return rf_model, val_auc


def predict_rf(model, X_test):
    """
    Generates predictions for the test set using the trained model.

    Args:
        model (RandomForestClassifier): Trained model.
        X_test (np.ndarray): Test features.

    Returns:
        np.ndarray: Predicted probabilities for the positive class.
    """
    print(f"Generating predictions for {X_test.shape[0]} test samples...")
    test_probs = model.predict_proba(X_test)[:, 1]
    return test_probs


def run_rf_pipeline(load_cached_data=True, debug=config.DEBUG):
    """
    Orchestrates the Stream A (Random Forest) pipeline:
    1. Loads/Computes features via FeatureEngineer.
    2. Trains the model.
    3. Generates predictions.

    Args:
        load_cached_data (bool): Whether to try loading features from cache.
        debug (bool): Whether to run in debug mode (subset of data).

    Returns:
        dict: Contains 'model', 'val_auc', 'test_preds', 'request_ids'.
    """
    utils.set_seed()

    # Temporarily override config.DEBUG to control data sampling in utils.load_data
    # This ensures FeatureEngineer processes the correct subset of data
    original_debug = config.DEBUG
    config.DEBUG = debug

    # If debugging, force re-computation (load_cached_data=False) to avoid loading
    # the full dataset from cache, as cache filenames do not distinguish debug/full.
    if debug:
        print("Debug mode enabled: Disabling cache loading to process data subset.")
        load_cached_data = False

    try:
        print("Retrieving features for Random Forest stream...")
        fe = FeatureEngineer()
        # Retrieve RF-specific features (stream A)
        # We ignore the second return value (mlp_out) as this pipeline is for RF
        rf_out, _ = fe.process_data(load_cached_data=load_cached_data)
    finally:
        # Restore original configuration
        config.DEBUG = original_debug

    # Extract data arrays
    X_train = rf_out["X_train"]
    y_train = rf_out["y_train"]
    X_val = rf_out["X_val"]
    y_val = rf_out["y_val"]
    X_test = rf_out["X_test"]
    request_ids_test = rf_out["request_ids_test"]

    # Train Model
    model, val_auc = train_rf(X_train, y_train, X_val, y_val, config.RF_PARAMS)

    # Predict on Test
    test_preds = predict_rf(model, X_test)

    return {
        "model": model,
        "val_auc": val_auc,
        "test_preds": test_preds,
        "request_ids": request_ids_test,
    }
