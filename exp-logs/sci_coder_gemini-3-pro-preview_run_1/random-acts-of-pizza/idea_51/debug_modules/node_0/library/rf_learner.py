import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import RF_PARAMS
from library.utils import load_from_cache, seed_everything


def extract_sparse(arr):
    """
    Helper to extract sparse matrix from numpy 0-d object array if necessary.
    When scipy.sparse matrices are saved via np.savez, they are often wrapped
    in a 0-d object array.
    """
    if isinstance(arr, np.ndarray) and arr.ndim == 0:
        return arr.item()
    return arr


def train_rf(X_train, y_train):
    """
    Initializes and trains the Random Forest Classifier.

    Args:
        X_train: Sparse matrix or array of training features.
        y_train: Array of training labels.

    Returns:
        Trained RandomForestClassifier.
    """
    print(f"Initializing Random Forest with params: {RF_PARAMS}")
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)
    return clf


def predict_rf(model, X):
    """
    Generates probability predictions for the positive class (received pizza).

    Args:
        model: Trained classifier.
        X: Sparse matrix or array of features.

    Returns:
        Array of probabilities for class 1.
    """
    # predict_proba returns [n_samples, n_classes], we want column 1 (True)
    return model.predict_proba(X)[:, 1]


def run_rf_learner():
    """
    Main pipeline for Stream A (Random Forest).
    Loads cached features, trains the model, evaluates on validation data,
    and generates predictions for the test set.

    Returns:
        val_preds (np.array): Predictions for the validation set.
        test_preds (np.array): Predictions for the test set.
        model (RandomForestClassifier): The trained model object.
    """
    seed_everything(RF_PARAMS["random_state"])

    # 1. Load Data
    print("Loading RF features from cache...")
    data = load_from_cache("features_rf.npz")
    if data is None:
        raise FileNotFoundError(
            "features_rf.npz not found in cache. Run feature engineering first."
        )

    # Extract arrays (handling potential 0-d object wrapping for sparse matrices)
    X_train = extract_sparse(data["X_train"])
    y_train = extract_sparse(data["y_train"])
    X_val = extract_sparse(data["X_val"])
    y_val = extract_sparse(data["y_val"])
    X_test = extract_sparse(data["X_test"])

    print(
        f"RF Data shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}"
    )

    # 2. Train
    print("Training Random Forest...")
    model = train_rf(X_train, y_train)

    # 3. Evaluate
    print("Evaluating on Validation set...")
    val_preds = predict_rf(model, X_val)
    val_auc = roc_auc_score(y_val, val_preds)
    print(f"RF Validation AUC: {val_auc}")

    # 4. Predict Test
    print("Generating Test predictions...")
    test_preds = predict_rf(model, X_test)

    return val_preds, test_preds, model
