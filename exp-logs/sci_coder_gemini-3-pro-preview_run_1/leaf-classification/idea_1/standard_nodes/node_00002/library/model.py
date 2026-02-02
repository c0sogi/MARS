import lightgbm as lgb
import numpy as np
from library.config import Config


class LeafModel:
    """
    A wrapper class for the LightGBM Classifier to handle training and prediction
    for the Leaf Classification task.
    """

    def __init__(self, params=None):
        """
        Initialize the LeafModel.

        Args:
            params (dict, optional): Hyperparameters for LightGBM.
                                     Defaults to Config.LGBM_PARAMS.
        """
        self.params = params.copy() if params else Config.LGBM_PARAMS.copy()
        self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the LightGBM model using the provided training data and optional validation data.
        Implements early stopping and metric logging.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_val (pd.DataFrame or np.ndarray, optional): Validation features.
            y_val (np.ndarray, optional): Validation labels.
        """
        # Dynamically set the number of classes based on the training target
        n_classes = len(np.unique(y_train))
        self.params["num_class"] = n_classes

        # Initialize the LightGBM Classifier
        # n_estimators corresponds to NUM_BOOST_ROUND
        self.model = lgb.LGBMClassifier(
            n_estimators=Config.NUM_BOOST_ROUND, **self.params
        )

        # Configure Callbacks
        # 1. Early Stopping: Stops training if validation score doesn't improve
        # 2. Log Evaluation: Prints metrics at specified intervals
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=True
            ),
            lgb.log_evaluation(period=Config.VERBOSE_EVAL),
        ]

        # Prepare evaluation set if validation data is provided
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # Train the model
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_metric="multi_logloss",
            callbacks=callbacks,
        )

        # Print the best score with full precision as required
        if self.model.best_score_:
            for data_name, metrics in self.model.best_score_.items():
                for metric_name, score in metrics.items():
                    print(
                        f"Best Validation Score [{data_name} - {metric_name}]: {score}"
                    )

    def predict(self, X):
        """
        Generates probability predictions for the input data.

        Args:
            X (pd.DataFrame or np.ndarray): Features to predict on.

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes).
        """
        if self.model is None:
            raise RuntimeError("The model must be trained before making predictions.")

        return self.model.predict_proba(X)
