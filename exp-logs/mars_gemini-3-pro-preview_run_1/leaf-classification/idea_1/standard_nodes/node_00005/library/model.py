import lightgbm as lgb
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import Config


class LeafModel:
    """
    A wrapper class for the LightGBM Classifier and LDA to handle training and prediction
    for the Leaf Classification task.
    Ensembles GBDT with LDA for better performance on small datasets.
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
        self.lda = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the LightGBM model and LDA using the provided training data.
        Implements early stopping and metric logging for LightGBM.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_val (pd.DataFrame or np.ndarray, optional): Validation features.
            y_val (np.ndarray, optional): Validation labels.
        """
        # Train LDA (Linear Discriminant Analysis)
        # Using 'lsqr' solver with auto shrinkage is robust for high-dim/small-sample data
        print("Training LDA...")
        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self.lda.fit(X_train, y_train)

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
        print("Training LightGBM...")
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
        Averages predictions from LightGBM and LDA.

        Args:
            X (pd.DataFrame or np.ndarray): Features to predict on.

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes).
        """
        if self.model is None or self.lda is None:
            raise RuntimeError("The model must be trained before making predictions.")

        # Get probabilities from both models
        p_lgbm = self.model.predict_proba(X)
        p_lda = self.lda.predict_proba(X)

        # Simple average ensemble
        return 0.5 * p_lgbm + 0.5 * p_lda
