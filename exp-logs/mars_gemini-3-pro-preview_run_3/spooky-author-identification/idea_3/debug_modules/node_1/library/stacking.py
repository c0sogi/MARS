import numpy as np
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.utils import set_seed


class MetaLearner:
    """
    The Meta-Learner (The Arbitrator) of the stacking ensemble.
    Uses Logistic Regression to weigh the contributions of base experts
    (Statistical, DeBERTa, RoBERTa) dynamically based on input characteristics
    (specifically text length).
    """

    def __init__(self, C=Config.META_C, solver=Config.META_SOLVER, max_iter=1000):
        """
        Initializes the MetaLearner.

        Args:
            C (float): Inverse of regularization strength for Logistic Regression.
            solver (str): Algorithm to use in the optimization problem.
            max_iter (int): Maximum number of iterations for the solver.
        """
        set_seed(Config.SEED)
        self.clf = LogisticRegression(
            C=C,
            solver=solver,
            multi_class="multinomial",
            max_iter=max_iter,
            random_state=Config.SEED,
        )

    def prepare_level1_features(self, base_preds_list, meta_features):
        """
        Constructs the feature matrix for the meta-learner by concatenating
        base model probabilities and the meta-feature (log character length).

        Args:
            base_preds_list (list of np.ndarray): A list containing the probability
                predictions from each base model. Each array should have shape
                (n_samples, n_classes).
            meta_features (np.ndarray or pd.Series): The meta-feature values
                (e.g., log_char_len) corresponding to the samples. Shape (n_samples,).

        Returns:
            np.ndarray: The combined feature matrix of shape
                (n_samples, n_base_models * n_classes + 1).
        """
        # Ensure inputs are numpy arrays
        processed_preds = []
        for pred in base_preds_list:
            # Handle potential tensor inputs by converting to numpy
            if hasattr(pred, "detach"):
                pred = pred.detach().cpu().numpy()
            processed_preds.append(np.array(pred))

        # Ensure meta_features is a 2D column vector (n_samples, 1)
        meta_feat_reshaped = np.array(meta_features).reshape(-1, 1)

        # Horizontally stack all probability vectors and the meta-feature
        # Structure: [Model1_Prob_EAP, Model1_Prob_HPL, Model1_Prob_MWS, Model2..., Meta_Feat]
        X_meta = np.hstack(processed_preds + [meta_feat_reshaped])

        return X_meta

    def fit(self, X, y):
        """
        Trains the Logistic Regression meta-learner.

        Args:
            X (np.ndarray): The Level 1 feature matrix from `prepare_level1_features`.
            y (np.ndarray): The target labels.

        Returns:
            self: The fitted instance.
        """
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using the trained meta-learner.
        Applies numerical stability clipping to avoid extremes in the log loss function.

        Args:
            X (np.ndarray): The Level 1 feature matrix for the test/validation set.

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes).
        """
        # Generate raw probabilities
        probs = self.clf.predict_proba(X)

        # Apply clipping as per the metric definition: max(min(p, 1-10^-15), 10^-15)
        eps = 1e-15
        probs_clipped = np.clip(probs, eps, 1 - eps)

        return probs_clipped
