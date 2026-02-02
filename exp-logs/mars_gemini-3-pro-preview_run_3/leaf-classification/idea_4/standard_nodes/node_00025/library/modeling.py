import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from library import config, utils


class HybridEnsemble:
    """
    A hybrid ensemble classifier combining Linear Discriminant Analysis (LDA)
    with Ledoit-Wolf shrinkage and Regularized Multinomial Logistic Regression.
    """

    def __init__(self, random_state=config.SEED):
        """
        Initialize the ensemble models.

        Args:
            random_state (int): Seed for reproducibility.
        """
        # LDA with Ledoit-Wolf shrinkage (requires lsqr or eigen solver)
        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

        # Regularized Multinomial Logistic Regression
        self.lr = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            penalty="l2",
            C=1.0,  # Standard regularization strength
            max_iter=2000,  # Increased iterations to ensure convergence
            random_state=random_state,
            n_jobs=-1,
        )

        self.weight = 0.5  # Default mixing weight (0.5 LDA, 0.5 LR)
        self.classes_ = None
        self.random_state = random_state

    def fit_models(self, X, y):
        """
        Trains both the LDA and Logistic Regression models.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
        """
        # Fit LDA
        self.lda.fit(X, y)

        # Fit Logistic Regression
        self.lr.fit(X, y)

        # Store classes and verify consistency
        self.classes_ = self.lda.classes_
        if not np.array_equal(self.lda.classes_, self.lr.classes_):
            raise ValueError(
                "Class mismatch between LDA and Logistic Regression models."
            )

        return self

    def find_optimal_weight(self, X_val, y_val, step_size=config.GRID_SEARCH_STEP):
        """
        Performs a discrete grid search to find the optimal mixing weight 'w'
        that minimizes log loss on the validation set.

        Ensemble Prob = w * LDA_Prob + (1 - w) * LR_Prob

        Args:
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels.
            step_size (float): Step size for the grid search.

        Returns:
            float: The optimal weight for LDA.
        """
        if self.classes_ is None:
            raise RuntimeError("Models must be fitted before optimizing weight.")

        # Pre-compute probabilities for efficiency
        p_lda = self.lda.predict_proba(X_val)
        p_lr = self.lr.predict_proba(X_val)

        best_loss = float("inf")
        best_w = 0.5

        # Create grid including 0.0 and 1.0
        # Adding step_size/2 to upper bound to ensure 1.0 is included despite float precision
        weights = np.arange(0.0, 1.0 + step_size / 1000.0, step_size)

        # Ensure 1.0 is explicitly checked if arange missed it due to precision
        if weights[-1] < 1.0:
            weights = np.append(weights, 1.0)

        for w in weights:
            # Calculate weighted average
            p_ens = w * p_lda + (1 - w) * p_lr

            # Clip probabilities to avoid log(0)
            p_ens_clipped = utils.clip_probabilities(p_ens)

            # Calculate Log Loss
            # Explicitly set eps=1e-16 to match the clipping in utils.py and avoid sklearn's default 1e-15
            current_loss = log_loss(y_val, p_ens_clipped, labels=self.classes_)

            if current_loss < best_loss:
                best_loss = current_loss
                best_w = w

        self.weight = best_w

        print(f"Optimal Ensemble Weight (LDA): {best_w}")
        print(f"Best Validation Log Loss: {best_loss}")

        return best_w

    def predict_proba(self, X):
        """
        Predict class probabilities using the trained ensemble and optimal weight.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Weighted probabilities.
        """
        if self.classes_ is None:
            raise RuntimeError("Models must be fitted before prediction.")

        p_lda = self.lda.predict_proba(X)
        p_lr = self.lr.predict_proba(X)

        # Weighted combination
        p_ens = self.weight * p_lda + (1 - self.weight) * p_lr

        # Clip probabilities as per metric requirements
        return utils.clip_probabilities(p_ens)
