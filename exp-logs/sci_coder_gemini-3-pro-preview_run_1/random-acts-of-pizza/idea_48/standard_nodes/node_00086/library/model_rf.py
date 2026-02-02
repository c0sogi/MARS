import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from typing import Dict, Tuple, Optional

from library.config import Config
from library.utils import seed_everything, print_metric, ensure_directory


class InteractionRandomForest:
    """
    Stream A: Interaction-Expanded Top-K Random Forest.
    Wraps sklearn's RandomForestClassifier with specific configurations for
    handling high-dimensional interaction and TF-IDF features.
    """

    def __init__(self):
        self.model_path = os.path.join(Config.CACHE_DIR, "best_rf.joblib")
        self.model = RandomForestClassifier(
            n_estimators=Config.RF_ESTIMATORS,
            min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
            class_weight=Config.RF_CLASS_WEIGHT,
            n_jobs=Config.RF_N_JOBS,
            random_state=Config.RANDOM_SEED,
            verbose=0,  # Silent execution
        )
        self.is_fitted = False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        """
        Trains the Random Forest model.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features (optional, for metric logging).
            y_val: Validation targets (optional, for metric logging).
        """
        seed_everything()
        print("Training Random Forest...")

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        if X_val is not None and y_val is not None:
            print("Evaluating Random Forest on Validation Set...")
            val_probs = self.model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, val_probs)
            print_metric("RF_Validation_AUC", auc)

        # Save the trained model
        ensure_directory(self.model_path)
        joblib.dump(self.model, self.model_path)
        print(f"Random Forest model saved to {self.model_path}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates probability predictions for the positive class.

        Args:
            X: Features to predict on.

        Returns:
            np.ndarray: Probabilities of class 1.
        """
        if not self.is_fitted:
            # Try to load from cache if not in memory
            if os.path.exists(self.model_path):
                print("Loading Random Forest model from cache...")
                self.model = joblib.load(self.model_path)
                self.is_fitted = True
            else:
                raise RuntimeError("Model is not fitted and no cached model found.")

        return self.model.predict_proba(X)[:, 1]


def run_rf_pipeline(
    rf_data: Dict[str, np.ndarray], force_retrain: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orchestrates the Random Forest pipeline: training (or loading) and prediction.

    Args:
        rf_data (Dict[str, np.ndarray]): Dictionary containing 'X_train', 'y_train',
                                         'X_val', 'y_val', 'X_test'.
        force_retrain (bool): If True, retrains the model even if a cache exists.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (Validation Probabilities, Test Probabilities)
    """
    rf_model = InteractionRandomForest()

    # Check if we need to train
    model_exists = os.path.exists(rf_model.model_path)

    if force_retrain or not model_exists:
        rf_model.train(
            rf_data["X_train"], rf_data["y_train"], rf_data["X_val"], rf_data["y_val"]
        )
    else:
        print("Skipping training, model will be loaded from cache during prediction.")

    # Generate predictions
    print("Generating predictions with Random Forest...")
    val_probs = rf_model.predict(rf_data["X_val"])
    test_probs = rf_model.predict(rf_data["X_test"])

    return val_probs, test_probs
