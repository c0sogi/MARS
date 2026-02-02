import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import library.config as config
from library.features import FeatureProcessor


class RFWrapper:
    """
    Wrapper class for the Random Forest stream of the ensemble.
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=config.RF_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            class_weight=config.RF_CLASS_WEIGHT,
            n_jobs=config.RF_N_JOBS,
            random_state=config.RANDOM_STATE,
            verbose=0,
        )

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the Random Forest model.

        Args:
            X_train: Training features (sparse matrix).
            y_train: Training labels.
            X_val: Validation features (optional).
            y_val: Validation labels (optional).
        """
        print("Training Random Forest...")
        self.model.fit(X_train, y_train)

        if X_val is not None and y_val is not None:
            print("Evaluating Random Forest on Validation set...")
            preds = self.predict(X_val)
            auc = roc_auc_score(y_val, preds)
            print(f"RF Validation AUC: {auc}")
            return auc
        return None

    def predict(self, X):
        """
        Generates probability predictions.

        Args:
            X: Feature matrix.

        Returns:
            1D numpy array of probabilities for the positive class.
        """
        # predict_proba returns [prob_0, prob_1], we want prob_1
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        """Saves the trained model to disk."""
        joblib.dump(self.model, path)

    def load(self, path):
        """Loads a trained model from disk."""
        self.model = joblib.load(path)


def run_rf_stream(load_cached_data=True):
    """
    Executes the full Random Forest pipeline:
    1. Loads processed features.
    2. Trains the model.
    3. Generates and saves predictions for Val and Test.

    Args:
        load_cached_data (bool): Whether to load features from cache.

    Returns:
        tuple: (val_predictions, test_predictions)
    """
    # 1. Load Data
    print("Initializing FeatureProcessor for RF Stream...")
    processor = FeatureProcessor()
    train_data, val_data, test_data = processor.process(
        load_cached_data=load_cached_data
    )

    # Extract specific features for RF (stored under 'rf_features' key in the dicts)
    X_train = train_data["rf_features"]
    y_train = train_data["labels"]
    X_val = val_data["rf_features"]
    y_val = val_data["labels"]
    X_test = test_data["rf_features"]

    # 2. Train Model
    rf_wrapper = RFWrapper()
    rf_wrapper.train(X_train, y_train, X_val, y_val)

    # 3. Generate Predictions
    print("Generating predictions...")
    val_preds = rf_wrapper.predict(X_val)
    test_preds = rf_wrapper.predict(X_test)

    # 4. Save Artifacts
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Save predictions as npy files for the ensemble
    val_preds_path = os.path.join(config.WORKING_DIR, "rf_val_preds.npy")
    test_preds_path = os.path.join(config.WORKING_DIR, "rf_test_preds.npy")
    model_path = os.path.join(config.WORKING_DIR, "rf_model.joblib")

    np.save(val_preds_path, val_preds)
    np.save(test_preds_path, test_preds)
    rf_wrapper.save(model_path)

    print(f"RF predictions saved to {val_preds_path} and {test_preds_path}")
    print(f"RF model saved to {model_path}")

    return val_preds, test_preds
