import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import seed_everything


class RFPipeline:
    """
    Manages the Random Forest Stream of the ensemble.
    Handles training, evaluation, and inference for the tree-based component.
    """

    def __init__(self):
        self.config = Config
        seed_everything(self.config.RANDOM_SEED)

    def train(self, X_train, y_train):
        """
        Trains the Random Forest Classifier using hyperparameters from Config.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training target vector.

        Returns:
            RandomForestClassifier: The fitted model.
        """
        print(
            f"Initializing Random Forest with {self.config.RF_N_ESTIMATORS} estimators..."
        )

        # Initialize classifier with config parameters
        clf = RandomForestClassifier(
            n_estimators=self.config.RF_N_ESTIMATORS,
            max_depth=self.config.RF_MAX_DEPTH,
            min_samples_leaf=self.config.RF_MIN_SAMPLES_LEAF,
            class_weight=self.config.RF_CLASS_WEIGHT,
            n_jobs=self.config.RF_N_JOBS,
            random_state=self.config.RANDOM_SEED,
            verbose=0,  # Silent execution
        )

        # Fit the model
        clf.fit(X_train, y_train)
        return clf

    def predict(self, model, X):
        """
        Generates probability predictions for the positive class.

        Args:
            model (RandomForestClassifier): Trained model.
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Probability scores for class 1.
        """
        # Predict probability of class 1 (Received Pizza)
        return model.predict_proba(X)[:, 1]

    def run(self, data_dict):
        """
        Executes the full RF pipeline: Train, Validate, Predict Test.

        Args:
            data_dict (dict): Output from FeatureEngineer.run() containing 'train', 'val', 'test' keys.
                              Each split dict must contain 'X_rf' and 'y' (except test).

        Returns:
            dict: {
                'model': trained_model,
                'val_probs': np.array,
                'test_probs': np.array,
                'val_auc': float
            }
        """
        print("--- Starting Random Forest Pipeline ---")

        # 1. Extract Data
        # The FeatureEngineer has already assembled X_rf as:
        # [TFIDF | Raw Metadata | TopK Flags | Interaction Features]
        X_train = data_dict["train"]["X_rf"]
        y_train = data_dict["train"]["y"]

        X_val = data_dict["val"]["X_rf"]
        y_val = data_dict["val"]["y"]

        X_test = data_dict["test"]["X_rf"]

        print(f"Train Input Shape: {X_train.shape}")
        print(f"Val Input Shape:   {X_val.shape}")
        print(f"Test Input Shape:  {X_test.shape}")

        # 2. Train
        print("Training Random Forest...")
        model = self.train(X_train, y_train)

        # 3. Validate
        print("Evaluating on Validation set...")
        val_probs = self.predict(model, X_val)
        val_auc = roc_auc_score(y_val, val_probs)

        # Print full precision as requested
        print(f"Random Forest Validation ROC AUC: {val_auc}")

        # 4. Predict Test
        print("Predicting on Test set...")
        test_probs = self.predict(model, X_test)

        print("--- Random Forest Pipeline Complete ---")

        return {
            "model": model,
            "val_probs": val_probs,
            "test_probs": test_probs,
            "val_auc": val_auc,
        }
