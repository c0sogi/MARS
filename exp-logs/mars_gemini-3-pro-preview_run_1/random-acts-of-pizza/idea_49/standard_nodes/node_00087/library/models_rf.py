import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import Config


def _prepare_features(feature_dict):
    """
    Combines dense interaction features and sparse TF-IDF features into a single sparse matrix.

    Args:
        feature_dict (dict): Dictionary containing 'dense' (numpy array) and 'tfidf' (sparse matrix).

    Returns:
        scipy.sparse.csr_matrix: Combined feature matrix.
    """
    dense_feats = feature_dict["dense"]
    tfidf_feats = feature_dict["tfidf"]

    # Ensure dense features are 2D
    if len(dense_feats.shape) == 1:
        dense_feats = dense_feats.reshape(-1, 1)

    # Convert dense to sparse for efficient stacking
    dense_sparse = sparse.csr_matrix(dense_feats)

    # Stack horizontally
    combined = sparse.hstack([dense_sparse, tfidf_feats], format="csr")

    return combined


def train_rf_model(train_features, train_labels, val_features, val_labels):
    """
    Trains a Random Forest classifier on the interaction-augmented feature set.

    Args:
        train_features (dict): Dictionary with 'dense' and 'tfidf' keys for training data.
        train_labels (array-like): Training labels.
        val_features (dict): Dictionary with 'dense' and 'tfidf' keys for validation data.
        val_labels (array-like): Validation labels.

    Returns:
        RandomForestClassifier: The trained model.
    """
    print("Preparing Random Forest features...")
    X_train = _prepare_features(train_features)
    X_val = _prepare_features(val_features)

    print(f"Feature matrix shape: {X_train.shape}")

    # Initialize model with Config hyperparameters
    clf = RandomForestClassifier(
        n_estimators=Config.RF_N_ESTIMATORS,
        min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
        class_weight=Config.RF_CLASS_WEIGHT,
        n_jobs=Config.RF_N_JOBS,
        random_state=Config.RF_RANDOM_STATE,
        verbose=0,  # Keep it silent as requested
    )

    print("Training Random Forest...")
    clf.fit(X_train, train_labels)

    # Validate
    print("Evaluating on validation set...")
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(val_labels, val_probs)

    # Print full precision metric
    print(f"Random Forest Validation AUC: {val_auc}")

    return clf


def predict_rf(model, test_features):
    """
    Generates predictions using the trained Random Forest model.

    Args:
        model (RandomForestClassifier): Trained model.
        test_features (dict): Dictionary with 'dense' and 'tfidf' keys for test data.

    Returns:
        np.array: Predicted probabilities for the positive class.
    """
    X_test = _prepare_features(test_features)

    # Predict probabilities for class 1 (Received Pizza)
    probs = model.predict_proba(X_test)[:, 1]

    return probs
