import numpy as np
from sklearn.ensemble import RandomForestClassifier
from library.config import Config
from library.utils import set_seed, calculate_roc_auc


class InteractionRandomForest:
    """
    Stream A: Interaction-Enhanced Consistency Random Forest.

    This model utilizes explicit features capturing the interaction between
    'Decoupled Consistency' (Topic vs. Narrative) and 'User Credibility'.
    It is configured with low regularization to preserve fine-grained sparse signals
    from the high-fidelity TF-IDF and Top-K binary indicators.
    """

    def __init__(self):
        """
        Initializes the Random Forest with hyperparameters from Config.
        """
        set_seed(Config.RF_RANDOM_STATE)

        self.model = RandomForestClassifier(
            n_estimators=Config.RF_N_ESTIMATORS,
            min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
            class_weight=Config.RF_CLASS_WEIGHT,
            random_state=Config.RF_RANDOM_STATE,
            n_jobs=-1,  # Use all available cores
            verbose=0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the Random Forest model.

        Args:
            X (np.ndarray): Training features (TF-IDF + Metadata + Interactions + Top-K).
            y (np.ndarray): Training labels.
        """
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts probabilities for the positive class (pizza received).

        Args:
            X (np.ndarray): Features to predict on.

        Returns:
            np.ndarray: Probabilities for class 1.
        """
        # predict_proba returns [n_samples, n_classes], we want column 1
        return self.model.predict_proba(X)[:, 1]


def train_rf_model(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> InteractionRandomForest:
    """
    Orchestrates the training and evaluation of the Random Forest model.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.

    Returns:
        InteractionRandomForest: The trained model instance.
    """
    print("Initializing Interaction-Enhanced Consistency Random Forest...")
    model = InteractionRandomForest()

    print(f"Training Random Forest with {Config.RF_N_ESTIMATORS} estimators...")
    model.fit(X_train, y_train)

    print("Evaluating on Validation Set...")
    y_pred_val = model.predict_proba(X_val)

    # Calculate and print metric with full precision
    auc = calculate_roc_auc(y_val, y_pred_val)
    print(f"Random Forest Validation ROC AUC: {auc}")

    return model
