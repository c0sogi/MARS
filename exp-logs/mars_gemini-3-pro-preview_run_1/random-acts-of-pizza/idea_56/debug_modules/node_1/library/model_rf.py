import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import print_log, seed_everything
from library.feature_manager import FeatureManager


def train_rf(load_cached_data=True):
    """
    Orchestrates the loading of data, training of the Random Forest model,
    and evaluation on the validation set.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.
                                 If False or cache missing, re-computes features.

    Returns:
        model (RandomForestClassifier): The trained model.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation targets.
        X_test (pd.DataFrame): Test features.
        test_ids (pd.Series): Test request IDs.
    """
    seed_everything()

    # 1. Load Dataset via FeatureManager
    # This handles TF-IDF, Metadata, Top-K, and Interaction features
    print_log("Initializing FeatureManager and loading RF dataset...")
    fm = FeatureManager()
    X_train, y_train, X_val, y_val, X_test, test_ids = fm.get_rf_dataset(
        load_cached_data=load_cached_data
    )

    # 2. Initialize Random Forest with Configuration
    print_log(
        f"Initializing Random Forest Classifier with {Config.RF_ESTIMATORS} estimators..."
    )
    model = RandomForestClassifier(
        n_estimators=Config.RF_ESTIMATORS,
        max_depth=Config.RF_MAX_DEPTH,
        min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
        class_weight=Config.RF_CLASS_WEIGHT,
        n_jobs=Config.RF_N_JOBS,
        random_state=Config.RANDOM_SEED,
        verbose=0,  # Keep stdout clean
    )

    # 3. Train Model
    print_log("Fitting Random Forest model on training data...")
    model.fit(X_train, y_train)

    # 4. Evaluate on Validation Set
    print_log("Evaluating model on validation set...")
    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)

    # Print validation metric with full precision
    print_log(f"RF Validation ROC AUC: {val_auc}")

    return model, X_val, y_val, X_test, test_ids


def predict_rf(model, X):
    """
    Generates probability predictions for a given feature set using the trained model.

    Args:
        model (RandomForestClassifier): The trained model.
        X (pd.DataFrame or np.ndarray): Features to predict on.

    Returns:
        np.ndarray: Probabilities of the positive class (received pizza).
    """
    # Predict probabilities for class 1 (True)
    return model.predict_proba(X)[:, 1]
