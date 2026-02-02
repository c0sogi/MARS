import numpy as np
from sklearn.ensemble import RandomForestClassifier
from library.config import Config
from library.utils import set_seed, compute_auc


def train_rf_model(X_train, y_train, X_val, y_val):
    """
    Trains a Random Forest Classifier using parameters from Config.
    Evaluates the model on the validation set and prints the AUC.

    Args:
        X_train (sparse matrix or array): Training features.
        y_train (array): Training labels.
        X_val (sparse matrix or array): Validation features.
        y_val (array): Validation labels.

    Returns:
        RandomForestClassifier: The trained model.
    """
    set_seed(Config.SEED)

    print("Initializing Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=Config.RF_N_ESTIMATORS,
        max_depth=Config.RF_MAX_DEPTH,
        min_samples_split=Config.RF_MIN_SAMPLES_SPLIT,
        class_weight=Config.RF_CLASS_WEIGHT,
        n_jobs=Config.RF_N_JOBS,
        random_state=Config.SEED,
        verbose=0,  # Keep silent as per requirements
    )

    print(f"Training Random Forest on {X_train.shape[0]} samples...")
    model.fit(X_train, y_train)

    print("Evaluating Random Forest on validation set...")
    # Predict probabilities for the positive class (index 1)
    y_val_pred = model.predict_proba(X_val)[:, 1]

    auc_score = compute_auc(y_val, y_val_pred)
    print(f"Random Forest Validation AUC: {auc_score}")

    return model


def predict_rf_model(model, X_test):
    """
    Generates predictions for the test set using the trained Random Forest model.

    Args:
        model (RandomForestClassifier): Trained model.
        X_test (sparse matrix or array): Test features.

    Returns:
        np.array: Predicted probabilities for the positive class.
    """
    print(f"Generating Random Forest predictions for {X_test.shape[0]} test samples...")
    # Predict probabilities for the positive class (index 1)
    y_pred = model.predict_proba(X_test)[:, 1]
    return y_pred
