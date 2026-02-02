import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from library.utils import set_seed, compute_auc


class InsultClassifier:
    """
    A Logistic Regression classifier designed for insult detection using sparse TF-IDF features.
    """

    def __init__(
        self,
        C=1.0,
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ):
        """
        Args:
            C (float): Inverse of regularization strength; smaller values specify stronger regularization.
            solver (str): Algorithm to use in the optimization problem ('liblinear' is good for sparse data).
            max_iter (int): Maximum number of iterations taken for the solvers to converge.
            class_weight (str or dict): Weights associated with classes. 'balanced' adjusts weights inversely proportional to class frequencies.
            random_state (int): Seed used by the random number generator.
        """
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the logistic regression model and prints performance metrics.

        Args:
            X_train (sparse matrix): Training features.
            y_train (array-like): Training labels.
            X_val (sparse matrix, optional): Validation features.
            y_val (array-like, optional): Validation labels.

        Returns:
            self: The fitted instance.
        """
        # Ensure reproducibility
        set_seed(self.random_state)

        # Initialize model
        self.model = LogisticRegression(
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            verbose=0,  # We handle printing manually
        )

        # Train the model
        # Note: LogisticRegression with liblinear/lbfgs solves the convex problem to tolerance.
        # Explicit early stopping based on validation loss is less common/necessary than for NN/GBM,
        # as L2 regularization (C) controls overfitting effectively.
        self.model.fit(X_train, y_train)

        # Evaluate on Training Data
        train_probs = self.model.predict_proba(X_train)[:, 1]
        train_auc = compute_auc(y_train, train_probs)
        train_loss = log_loss(y_train, train_probs)

        print(f"Training AUC: {train_auc}")
        print(f"Training LogLoss: {train_loss}")

        # Evaluate on Validation Data (if provided)
        if X_val is not None and y_val is not None:
            val_probs = self.model.predict_proba(X_val)[:, 1]
            val_auc = compute_auc(y_val, val_probs)
            val_loss = log_loss(y_val, val_probs)

            print(f"Validation AUC: {val_auc}")
            print(f"Validation LogLoss: {val_loss}")

        return self

    def predict(self, X):
        """
        Predicts the probability of the comment being an insult.

        Args:
            X (sparse matrix): Input features.

        Returns:
            np.array: Probabilities for the positive class (1).
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        # Return probabilities for class 1 (Insult)
        return self.model.predict_proba(X)[:, 1]
