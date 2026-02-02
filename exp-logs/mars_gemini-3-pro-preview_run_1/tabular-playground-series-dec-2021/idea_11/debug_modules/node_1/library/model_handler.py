import xgboost as xgb
from sklearn.metrics import accuracy_score
from library.config import Config


class ModelTrainer:
    """
    Handles the lifecycle of the XGBoost model, including initialization,
    training with early stopping, and prediction.
    """

    def __init__(self, params=None):
        """
        Initialize the ModelTrainer with XGBoost parameters.

        Args:
            params (dict, optional): Dictionary of parameters to override the defaults
                                     in Config.XGB_PARAMS. Useful for adjusting
                                     hyperparameters like n_estimators during debugging.
        """
        # Start with default parameters from Config
        self.params = Config.XGB_PARAMS.copy()

        # Update with any provided overrides
        if params:
            self.params.update(params)

        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the XGBoost model using the provided training and validation data.
        Implements early stopping based on the validation set.

        Args:
            X_train (array-like): Training features.
            y_train (array-like): Training labels.
            X_val (array-like): Validation features.
            y_val (array-like): Validation labels.

        Returns:
            xgb.XGBClassifier: The trained model instance.
        """
        # Initialize the classifier with the configured parameters
        # random_state is set for reproducibility
        self.model = xgb.XGBClassifier(**self.params, random_state=Config.SEED)

        # Fit the model
        # eval_set is used for early stopping (configured via early_stopping_rounds in params)
        # verbose=False suppresses the per-iteration logs
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Calculate and print validation accuracy
        val_preds = self.model.predict(X_val)
        acc = accuracy_score(y_val, val_preds)

        # Print full precision as requested
        print(f"Validation Accuracy: {acc}")

        return self.model

    def predict_proba(self, X):
        """
        Generates class probabilities for the input data.

        Args:
            X (array-like): Input features.

        Returns:
            np.ndarray: Class probabilities of shape (n_samples, n_classes).
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        return self.model.predict_proba(X)

    def predict(self, X):
        """
        Generates class predictions for the input data.

        Args:
            X (array-like): Input features.

        Returns:
            np.ndarray: Predicted class labels.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        return self.model.predict(X)
