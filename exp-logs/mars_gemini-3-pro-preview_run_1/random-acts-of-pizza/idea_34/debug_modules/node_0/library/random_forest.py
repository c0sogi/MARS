import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import set_seed


class RandomForestTrainer:
    """
    Trainer for the Random Forest stream of the Hybrid Ensemble.
    Encapsulates training, validation, persistence, and inference logic.
    """

    def __init__(self):
        set_seed(Config.RANDOM_STATE)
        self.model = None
        self.imputer = None
        self.model_path = os.path.join(Config.WORKING_DIR, "rf_model.pkl")

    def train(self, train_data, val_data):
        """
        Trains the Random Forest model using features prepared by the FeaturePipeline.

        Args:
            train_data (dict): Dictionary containing 'rf_features' and 'labels'.
            val_data (dict): Dictionary containing 'rf_features' and 'labels'.

        Returns:
            float: Validation AUC score.
        """
        print("Initializing Random Forest training...")

        # Extract features and labels
        X_train = train_data["rf_features"]
        y_train = train_data["labels"]
        X_val = val_data["rf_features"]
        y_val = val_data["labels"]

        # Imputation
        # Although FeaturePipeline handles metadata imputation, we apply a safety imputer
        # here to ensure the concatenated matrix (including TF-IDF/PeakRelevance) is clean.
        print("  Fitting imputer (strategy='median')...")
        self.imputer = SimpleImputer(strategy="median")
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_val_imputed = self.imputer.transform(X_val)

        # Initialize Random Forest with Config hyperparameters
        self.model = RandomForestClassifier(
            n_estimators=Config.RF_N_ESTIMATORS,
            class_weight=Config.RF_CLASS_WEIGHT,
            min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
            max_depth=Config.RF_MAX_DEPTH,
            n_jobs=Config.RF_N_JOBS,
            random_state=Config.RANDOM_STATE,
            verbose=0,
        )

        # Train
        print(f"  Training Random Forest with {Config.RF_N_ESTIMATORS} estimators...")
        self.model.fit(X_train_imputed, y_train)

        # Validate
        print("  Evaluating on validation set...")
        val_probs = self.model.predict_proba(X_val_imputed)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)

        # Print full precision as requested
        print(f"Random Forest Validation AUC: {val_auc}")

        # Save artifacts
        self._save_model()

        return val_auc

    def predict(self, test_data):
        """
        Generates predictions for the test set.

        Args:
            test_data (dict): Dictionary containing 'rf_features'.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        # Ensure model is loaded
        if self.model is None or self.imputer is None:
            self._load_model()

        X_test = test_data["rf_features"]

        # Impute using the imputer fitted on training data
        X_test_imputed = self.imputer.transform(X_test)

        # Predict probabilities (class 1)
        probs = self.model.predict_proba(X_test_imputed)[:, 1]

        return probs

    def _save_model(self):
        """Saves the model and imputer to disk."""
        payload = {"model": self.model, "imputer": self.imputer}
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump(payload, f)
            print(f"Saved RF model to {self.model_path}")
        except Exception as e:
            print(f"Warning: Failed to save RF model: {e}")

    def _load_model(self):
        """Loads the model and imputer from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Please train the model first."
            )

        try:
            with open(self.model_path, "rb") as f:
                payload = pickle.load(f)

            self.model = payload["model"]
            self.imputer = payload["imputer"]
            print(f"Loaded RF model from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load RF model: {e}")
