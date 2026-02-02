import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import Config, set_seed


class InteractionProjectedRF:
    """
    Wrapper for the Interaction-Projected Random Forest (Stream A).

    This model utilizes a feature matrix constructed by concatenating:
    - High-Fidelity TF-IDF vectors (Dense)
    - Full-Spectrum Metadata (Raw & Ratios)
    - Top-K Community Indicators
    - Global Consistency Scalars
    - Explicit Interaction Terms

    The construction of this matrix is handled by the feature engineering pipeline
    in library/feature_engine.py, ensuring the tree splits on explicit interaction
    signals projected from the neural network's logic.
    """

    def __init__(self, params=None):
        """
        Initialize the Random Forest model.

        Args:
            params (dict, optional): Hyperparameters for RandomForestClassifier.
                                     Defaults to Config.RF_PARAMS.
        """
        self.params = params if params is not None else Config.RF_PARAMS
        self.model = RandomForestClassifier(**self.params)

    def fit(self, X, y):
        """
        Train the Random Forest model.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.

        Returns:
            self
        """
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Generate probability predictions for the positive class.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Probabilities of class 1.
        """
        # predict_proba returns [prob_0, prob_1], we want prob_1
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X, y):
        """
        Evaluate the model on a validation set using ROC AUC.

        Args:
            X (np.ndarray): Validation features.
            y (np.ndarray): Validation labels.

        Returns:
            float: ROC AUC score.
        """
        preds = self.predict_proba(X)
        return roc_auc_score(y, preds)


def train_rf_model(data_dict, verbose=True):
    """
    Trains and evaluates the InteractionProjectedRF model using the provided data dictionary.

    Args:
        data_dict (dict): Dictionary containing processed data keys:
                          'rf_train', 'rf_val', 'y_train', 'y_val'.
        verbose (bool): Whether to print evaluation metrics.

    Returns:
        tuple: (trained_model, validation_auc)
    """
    set_seed(Config.SEED)

    if verbose:
        print("Initializing InteractionProjectedRF (Stream A)...")

    # Initialize model
    rf_model = InteractionProjectedRF(params=Config.RF_PARAMS)

    # Extract data
    X_train = data_dict["rf_train"]["X"]
    y_train = data_dict["y_train"]
    X_val = data_dict["rf_val"]["X"]
    y_val = data_dict["y_val"]

    if verbose:
        print(f"Training on {len(X_train)} samples with {X_train.shape[1]} features...")

    # Train
    rf_model.fit(X_train, y_train)

    # Evaluate
    val_auc = rf_model.evaluate(X_val, y_val)

    if verbose:
        # Print full precision as requested
        print(f"Random Forest Validation AUC: {val_auc}")

    return rf_model, val_auc


def predict_rf(model, data_dict):
    """
    Generates predictions for the test set.

    Args:
        model (InteractionProjectedRF): Trained model.
        data_dict (dict): Dictionary containing 'rf_test'.

    Returns:
        np.ndarray: Test set probabilities.
    """
    X_test = data_dict["rf_test"]["X"]
    return model.predict_proba(X_test)
