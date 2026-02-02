import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import save_pickle, load_pickle, set_seed


class RandomForestStream:
    """
    Manages the Random Forest stream (Stream A) of the solution.
    Handles training, evaluation, and inference using the features prepared by FeatureEngine.
    """

    def __init__(self):
        self.model = None
        self.model_filename = "rf_model.pkl"

    def train(self, rf_data, force_retrain=False):
        """
        Trains the Random Forest model using the provided data.

        Args:
            rf_data (dict): Dictionary containing 'train' and 'val' keys.
                            Each inner dict must have 'X' and 'y' keys.
            force_retrain (bool): If True, ignores cached model and retrains.

        Returns:
            float: Validation ROC AUC score.
            np.ndarray: Validation predictions (probabilities).
        """
        set_seed()

        # Try to load cached model first unless forced to retrain
        if not force_retrain:
            loaded_model = load_pickle(self.model_filename)
            if loaded_model is not None:
                print(
                    f"Loading cached Random Forest model from {self.model_filename}..."
                )
                self.model = loaded_model

                # Generate validation metrics for the loaded model
                X_val = rf_data["val"]["X"]
                y_val = rf_data["val"]["y"]
                val_preds = self.predict(X_val)
                val_auc = roc_auc_score(y_val, val_preds)
                print(f"Loaded RF Validation ROC AUC: {val_auc}")
                return val_auc, val_preds

        print("Training Random Forest Model...")

        # Extract training data
        X_train = rf_data["train"]["X"]
        y_train = rf_data["train"]["y"]

        # Extract validation data
        X_val = rf_data["val"]["X"]
        y_val = rf_data["val"]["y"]

        # Initialize model with Config hyperparameters
        self.model = RandomForestClassifier(
            n_estimators=Config.RF_N_ESTIMATORS,
            min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
            class_weight=Config.RF_CLASS_WEIGHT,
            n_jobs=Config.RF_N_JOBS,
            random_state=Config.RANDOM_SEED,
            verbose=0,  # Silent execution
        )

        # Fit model
        self.model.fit(X_train, y_train)

        # Save model to cache
        save_pickle(self.model, self.model_filename)

        # Evaluate on validation set
        val_preds = self.predict(X_val)
        val_auc = roc_auc_score(y_val, val_preds)

        print(f"Random Forest Validation ROC AUC: {val_auc}")

        return val_auc, val_preds

    def predict(self, X):
        """
        Generates probability predictions for the positive class.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Array of probabilities for class 1.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")

        # Predict probabilities (index 1 is for the positive class 'True')
        probs = self.model.predict_proba(X)[:, 1]
        return probs
