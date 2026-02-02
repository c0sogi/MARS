import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library import config


class LeafLDA:
    """
    A wrapper for Linear Discriminant Analysis (LDA) specifically configured for
    high-dimensional, low-sample datasets using Ledoit-Wolf shrinkage.

    Implements a Transductive Self-Training loop to leverage unlabeled test data
    for robust covariance estimation.
    """

    def __init__(self, solver=config.LDA_SOLVER, shrinkage=config.LDA_SHRINKAGE):
        """
        Initialize the LeafLDA classifier.

        Args:
            solver (str): Solver to use ('lsqr' or 'eigen'). Defaults to config.LDA_SOLVER.
            shrinkage (str or float): Shrinkage parameter ('auto' for Ledoit-Wolf).
                                      Defaults to config.LDA_SHRINKAGE.
        """
        self.solver = solver
        self.shrinkage = shrinkage
        self.model = LinearDiscriminantAnalysis(
            solver=self.solver, shrinkage=self.shrinkage
        )
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit the Linear Discriminant Analysis model according to the given training data.

        Args:
            X (pd.DataFrame or np.ndarray): Training vector, where n_samples is the number of samples
                                            and n_features is the number of features.
            y (pd.Series or np.ndarray): Target values (class labels).

        Returns:
            self: Returns the instance itself.
        """
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Args:
            X (pd.DataFrame or np.ndarray): The input samples.

        Returns:
            np.ndarray: Vector of predicted class labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Estimate probability.

        Args:
            X (pd.DataFrame or np.ndarray): The input samples.

        Returns:
            np.ndarray: Returns the probability of the sample for each class in the model.
        """
        return self.model.predict_proba(X)

    def fit_transductive(
        self,
        X_train,
        y_train,
        X_test,
        pseudo_label_threshold=config.PSEUDO_LABEL_THRESHOLD,
    ):
        """
        Performs Transductive Self-Training (Pseudo-Labeling).

        1. Trains a Supervisor model on (X_train, y_train).
        2. Generates predictions for the unlabeled X_test.
        3. Selects test samples where prediction confidence > pseudo_label_threshold.
        4. Retrains the model on the combined set (Train + High-Confidence Test).

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_test (pd.DataFrame): Test features (unlabeled).
            pseudo_label_threshold (float): Probability threshold for accepting a pseudo-label.

        Returns:
            self: Returns the retrained instance.
        """
        # 1. Supervisor Training
        print("Starting Supervisor Training on labeled data...")
        self.fit(X_train, y_train)

        # 2. Pseudo-Label Generation
        print("Generating predictions for Transductive Pseudo-Labeling...")
        probs_test = self.predict_proba(X_test)

        # Identify predicted class and max probability for each test sample
        pred_indices = np.argmax(probs_test, axis=1)
        max_probs = np.max(probs_test, axis=1)
        pred_labels = self.classes_[pred_indices]

        # 3. Confidence Filtering
        high_conf_mask = max_probs > pseudo_label_threshold
        n_pseudo = np.sum(high_conf_mask)

        if n_pseudo > 0:
            print(
                f"Transductive Learning: Found {n_pseudo} high-confidence test samples (Threshold > {pseudo_label_threshold})."
            )

            # Extract pseudo-labeled features
            if isinstance(X_test, pd.DataFrame):
                X_pseudo = X_test[high_conf_mask].copy()
            else:
                X_pseudo = X_test[high_conf_mask].copy()

            # Extract pseudo-labels
            y_pseudo_values = pred_labels[high_conf_mask]

            # Combine Training Data with Pseudo-Labeled Data
            if isinstance(X_train, pd.DataFrame) and isinstance(y_train, pd.Series):
                # Ensure y_pseudo is a Series with matching index
                y_pseudo = pd.Series(y_pseudo_values, index=X_pseudo.index)

                X_combined = pd.concat([X_train, X_pseudo], axis=0)
                y_combined = pd.concat([y_train, y_pseudo], axis=0)
            else:
                # Fallback for numpy arrays
                X_combined = np.vstack([X_train, X_pseudo])
                y_combined = np.hstack([y_train, y_pseudo_values])

            # 4. Retraining
            print(f"Retraining model on combined dataset (Size: {len(X_combined)})...")
            self.model.fit(X_combined, y_combined)
            self.classes_ = self.model.classes_

        else:
            print(
                "Transductive Learning: No test samples met the confidence threshold. Skipping retraining."
            )

        return self
