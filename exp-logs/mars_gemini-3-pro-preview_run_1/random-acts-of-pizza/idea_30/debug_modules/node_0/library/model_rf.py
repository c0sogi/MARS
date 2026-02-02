import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import set_seed, save_artifact, load_artifact


class RFPredictor:
    """
    Stream A: Alignment-Augmented Top-K Random Forest.
    Wraps sklearn RandomForestClassifier with specific configuration for the ensemble.
    """

    def __init__(self):
        """
        Initializes the Random Forest predictor with hyperparameters from Config.
        """
        # Ensure reproducibility
        set_seed(Config.RANDOM_STATE)

        # Initialize Random Forest with Config hyperparameters
        # min_samples_leaf=1 is crucial for preserving sparse Top-K signals
        self.model = RandomForestClassifier(
            n_estimators=Config.RF_N_ESTIMATORS,
            class_weight=Config.RF_CLASS_WEIGHT,
            min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
            max_depth=Config.RF_MAX_DEPTH,
            n_jobs=Config.RF_N_JOBS,
            random_state=Config.RANDOM_STATE,
            verbose=0,
        )
        self.target_col = "requester_received_pizza"

    def train(self, train_df: pd.DataFrame):
        """
        Trains the Random Forest model.

        Args:
            train_df (pd.DataFrame): Training data containing features and target.
        """
        if self.target_col not in train_df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in training data."
            )

        # Separate features and target
        y = train_df[self.target_col]
        X = train_df.drop(columns=[self.target_col])

        print(
            f"Training Random Forest on {X.shape[0]} samples and {X.shape[1]} features..."
        )
        self.model.fit(X, y)
        print("RF Training complete.")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generates probability predictions for the positive class.

        Args:
            df (pd.DataFrame): Data to predict on. Can contain target column (will be dropped).

        Returns:
            np.ndarray: Probabilities for class 1 (pizza received).
        """
        # Drop target if present (e.g., in validation set)
        if self.target_col in df.columns:
            X = df.drop(columns=[self.target_col])
        else:
            X = df

        # Predict class 1 probabilities
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, val_df: pd.DataFrame):
        """
        Evaluates the model on validation data and prints AUC.

        Args:
            val_df (pd.DataFrame): Validation data with target.

        Returns:
            float: ROC AUC score.
        """
        if self.target_col not in val_df.columns:
            raise ValueError(
                "Validation data must contain target column for evaluation."
            )

        y_true = val_df[self.target_col]
        y_pred = self.predict_proba(val_df)

        auc = roc_auc_score(y_true, y_pred)
        print(f"Random Forest Validation AUC: {auc}")
        return auc

    def save(self, path: str):
        """
        Saves the predictor object to a file.

        Args:
            path (str): Destination path.
        """
        print(f"Saving RF model to {path}...")
        save_artifact(self, path)

    @staticmethod
    def load(path: str):
        """
        Loads a predictor object from a file.

        Args:
            path (str): Source path.

        Returns:
            RFPredictor: Loaded object.
        """
        print(f"Loading RF model from {path}...")
        return load_artifact(path)
