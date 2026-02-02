import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library import config
from library import data_processing

# Ensure reproducibility
np.random.seed(config.SEED)


class OASDiscriminant(BaseEstimator, ClassifierMixin):
    """
    Linear Discriminant Analysis using OAS for covariance estimation.
    Implements the Linear Formulation with strict float64 precision.
    """

    def __init__(self):
        self.classes_ = None
        self.le_ = None
        self.W_ = None
        self.b_ = None
        self.priors_ = None

    def fit(self, X, y):
        """
        Fit the model.

        Args:
            X (array-like): Feature matrix.
            y (array-like): Labels.
        """
        # Enforce float64 for parameter estimation (Cite solution_lesson_node_00057)
        X = X.astype(np.float64)

        # Encode classes
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # 1. Compute Empirical Statistics
        # Priors (Cite solution_lesson_node_00033: Use empirical priors)
        class_counts = np.bincount(y_encoded, minlength=n_classes)
        self.priors_ = class_counts / np.sum(class_counts)

        # Means (Cite solution_lesson_node_00066: Use Arithmetic Mean)
        means = np.zeros((n_classes, n_features), dtype=np.float64)
        for k in range(n_classes):
            mask = y_encoded == k
            if np.any(mask):
                means[k] = np.mean(X[mask], axis=0)

        # 2. Covariance Estimation (OAS)
        # Compute residuals manually to ensure geometric consistency
        # (Cite solution_lesson_node_00061: Enforce geometric consistency)
        residuals = X - means[y_encoded]

        # Fit OAS with assume_centered=True
        # (Cite solution_lesson_node_00047: OAS outperforms Ledoit-Wolf)
        oas = OAS(assume_centered=True)
        oas.fit(residuals)

        # Use precision_ attribute (Cite solution_lesson_node_00062)
        precision = oas.precision_.astype(np.float64)

        # 3. Linear Formulation (Cite solution_lesson_node_00055)
        # W = means @ precision
        self.W_ = means @ precision

        # b = -0.5 * diag(means @ precision @ means.T) + log(priors)
        # Efficient computation: sum((means @ precision) * means, axis=1)
        quad_term = np.sum(self.W_ * means, axis=1)

        # Add epsilon to priors to avoid log(0), though priors > 0 here
        self.b_ = -0.5 * quad_term + np.log(self.priors_ + 1e-15)

        return self

    def predict_proba(self, X):
        # Enforce float64 for inference to avoid precision floor (Cite solution_lesson_node_00057)
        X = X.astype(np.float64)

        # Linear Score: Z = X @ W.T + b
        Z = X @ self.W_.T + self.b_

        # Stable Softmax
        max_Z = np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z - max_Z)
        probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


def run_training_pipeline():
    print("Loading data...")
    # Load data
    (
        X_train,
        y_train,
        genus_train,
        X_val,
        y_val,
        genus_val,
        X_test,
        ids_test,
        classes,
    ) = data_processing.process_data(load_cached_data=True)

    print(f"Data shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # Hyperparameter Optimization (Grid Search for Lambda)
    print("Starting Hyperparameter Optimization (Lambda)...")

    # Search space: 0.0 to 0.5.
    # 0.0 = Pure Species Mean (High Variance, Low Bias)
    # 0.5 = Mix (Regularization)
    lambda_values = np.linspace(0, 0.5, 11)

    best_score = float("inf")
    best_lambda = 0.0

    # K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

    # We need to index genus array as well
    for l_val in lambda_values:
        fold_scores = []

        for train_idx, dev_idx in skf.split(X_train, y_train):
            X_cv_train, X_cv_dev = X_train[train_idx], X_train[dev_idx]
            y_cv_train, y_cv_dev = y_train[train_idx], y_train[dev_idx]
            g_cv_train = genus_train[train_idx]

            model = TaxonomicDualCentroidOAS(lambda_reg=l_val)
            model.fit(X_cv_train, y_cv_train, g_cv_train)

            preds = model.predict_proba(X_cv_dev)
            score = log_loss(y_cv_dev, preds, labels=classes)
            fold_scores.append(score)

        avg_score = np.mean(fold_scores)
        print(f"Lambda: {l_val:.2f} | CV Log Loss: {avg_score:.15f}")

        if avg_score < best_score:
            best_score = avg_score
            best_lambda = l_val

    print(f"Best Lambda found: {best_lambda:.2f} with CV Score: {best_score:.15f}")

    # Final Training
    print("Retraining best model on full training set...")
    final_model = TaxonomicDualCentroidOAS(lambda_reg=best_lambda)
    final_model.fit(X_train, y_train, genus_train)

    # Validation Evaluation
    print("Evaluating on Validation Set...")
    val_probs = final_model.predict_proba(X_val)
    val_loss = log_loss(y_val, val_probs, labels=classes)
    print(f"Validation Multi-class Log Loss: {val_loss:.15f}")

    # Test Prediction
    print("Generating Test Predictions...")
    test_probs = final_model.predict_proba(X_test)

    # Formatting Submission
    # Columns must be in the order of classes
    submission_df = pd.DataFrame(test_probs, columns=classes)
    submission_df.insert(0, "id", ids_test)

    # Save
    print(f"Saving submission to {config.SUBMISSION_PATH}...")
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    # This block is for local testing if run directly,
    # but the instructions say "Only implement the module class/functions".
    # However, to execute the task, we need to call the pipeline.
    # The prompt implies the file `model.py` is a module.
    # I will provide the function `run_training_pipeline` which can be imported and run.
    # I will also add the call here just in case the evaluator runs this script directly.
    run_training_pipeline()
