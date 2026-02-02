import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import set_seed


def train_predict_rf(
    train_tfidf, train_meta, train_y, val_tfidf, val_meta, val_y, test_tfidf, test_meta
):
    """
    Executes the Stream A (Random Forest) pipeline.

    1. Combines TF-IDF features and Metadata features.
    2. Trains a Random Forest Classifier using Config.RF_PARAMS.
    3. Evaluates on Validation set.
    4. Generates predictions for Test set.

    Args:
        train_tfidf (np.ndarray): TF-IDF features for training set (Dense).
        train_meta (np.ndarray): Imputed metadata features for training set (Dense).
        train_y (np.ndarray or pd.Series): Target labels for training set.
        val_tfidf (np.ndarray): TF-IDF features for validation set (Dense).
        val_meta (np.ndarray): Imputed metadata features for validation set (Dense).
        val_y (np.ndarray or pd.Series): Target labels for validation set.
        test_tfidf (np.ndarray): TF-IDF features for test set (Dense).
        test_meta (np.ndarray): Imputed metadata features for test set (Dense).

    Returns:
        tuple: (val_probs, test_probs, model)
            - val_probs (np.ndarray): Probability predictions for validation set.
            - test_probs (np.ndarray): Probability predictions for test set.
            - model (RandomForestClassifier): The trained model.
    """
    # Ensure reproducibility
    set_seed()

    print("Stream A: Preparing feature sets...")

    # Concatenate TF-IDF (Dense) and Metadata (Dense)
    # The TabularProcessor in library/feature_engineering.py converts TF-IDF to dense arrays via .toarray()
    # so np.hstack is appropriate here.
    X_train = np.hstack([train_tfidf, train_meta])
    X_val = np.hstack([val_tfidf, val_meta])
    X_test = np.hstack([test_tfidf, test_meta])

    print(f"Stream A: Input shape: {X_train.shape}")

    # Initialize Random Forest
    rf_params = Config.RF_PARAMS
    print(f"Stream A: Initializing Random Forest with params: {rf_params}")
    model = RandomForestClassifier(**rf_params)

    # Train
    print("Stream A: Training model...")
    model.fit(X_train, train_y)

    # Validation
    print("Stream A: Validating...")
    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(val_y, val_probs)

    # Print full precision as requested
    print(f"Stream A Validation AUC: {val_auc}")

    # Test Predictions
    print("Stream A: Generating test predictions...")
    test_probs = model.predict_proba(X_test)[:, 1]

    return val_probs, test_probs, model
