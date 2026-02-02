import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import Config
from library.utils import seed_everything, calculate_log_loss


class EnsembleClassifier:
    """
    Ensemble classifier combining Logistic Regression and Linear Discriminant Analysis.
    Optimizes the mixing weight based on validation log loss.
    """

    def __init__(self):
        """
        Initializes the classifiers with hyperparameters from Config.
        """
        seed_everything(Config.SEED)

        # Initialize Logistic Regression
        # Uses L-BFGS solver suitable for multiclass problems
        self.lr = LogisticRegression(
            solver=Config.LR_SOLVER,
            max_iter=Config.LR_MAX_ITER,
            C=Config.LR_C,
            multi_class="multinomial",
            random_state=Config.SEED,
            n_jobs=-1,  # Utilize available vCPUs
        )

        # Initialize Linear Discriminant Analysis
        # Uses Ledoit-Wolf shrinkage ('auto') with lsqr solver
        self.lda = LinearDiscriminantAnalysis(
            solver=Config.LDA_SOLVER, shrinkage=Config.LDA_SHRINKAGE
        )

        self.best_weight = 0.5  # Default weight for LDA

    def fit(self, X, y):
        """
        Trains both the Logistic Regression and LDA models.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
        """
        print("Training Logistic Regression...")
        self.lr.fit(X, y)

        print("Training Linear Discriminant Analysis...")
        self.lda.fit(X, y)

    def _get_individual_probabilities(self, X):
        """
        Helper to get probabilities from both models.

        Args:
            X (np.ndarray): Features.

        Returns:
            tuple: (probs_lr, probs_lda)
        """
        probs_lr = self.lr.predict_proba(X)
        probs_lda = self.lda.predict_proba(X)
        return probs_lr, probs_lda

    def optimize_ensemble_weight(self, X_val, y_val):
        """
        Performs a grid search to find the optimal mixing weight for the ensemble
        on the validation set.

        Formula: P_final = w * P_LDA + (1 - w) * P_LR

        Args:
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels.

        Returns:
            float: The optimal weight for the LDA component.
        """
        print("Optimizing ensemble weight on validation set...")
        probs_lr, probs_lda = self._get_individual_probabilities(X_val)

        # Generate weights from 0.0 to 1.0 inclusive
        # Using linspace to avoid floating point issues with arange at boundaries
        num_steps = int(1.0 / Config.ENSEMBLE_WEIGHT_STEP) + 1
        weights = np.linspace(0.0, 1.0, num_steps)

        best_loss = float("inf")
        best_w = 0.5

        for w in weights:
            # Blend probabilities
            probs_blend = w * probs_lda + (1 - w) * probs_lr

            # Calculate metric
            loss = calculate_log_loss(y_val, probs_blend)

            if loss < best_loss:
                best_loss = loss
                best_w = w

        self.best_weight = best_w

        print(f"Best Ensemble Weight (LDA): {self.best_weight}")
        print(f"Best Validation Log Loss: {best_loss}")

        return best_w

    def predict(self, X, weight=None):
        """
        Generates blended probability predictions.

        Args:
            X (np.ndarray): Features.
            weight (float, optional): Weight for LDA. If None, uses best_weight.

        Returns:
            np.ndarray: Blended probabilities.
        """
        if weight is None:
            weight = self.best_weight

        probs_lr, probs_lda = self._get_individual_probabilities(X)

        # Blend: w * LDA + (1-w) * LR
        probs_blend = weight * probs_lda + (1 - weight) * probs_lr

        return probs_blend
