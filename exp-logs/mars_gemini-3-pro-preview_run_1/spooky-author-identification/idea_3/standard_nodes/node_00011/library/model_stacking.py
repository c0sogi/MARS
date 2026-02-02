import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score
from library.config import Config
from library.utils import clip_probabilities


class StackingMetaLearner:
    """
    Level 2 Meta-Learner using XGBoost.
    Aggregates predictions from Level 1 experts and meta-features to make final predictions.
    """

    def __init__(self):
        self.params = Config.XGB_PARAMS.copy()
        self.num_rounds = Config.XGB_NUM_ROUNDS
        self.early_stopping_rounds = Config.XGB_EARLY_STOPPING_ROUNDS
        self.model = None

    def prepare_meta_features(self, base_probs_list, meta_features):
        """
        Concatenates Level 1 probabilities with explicit meta-features and uncertainty statistics.
        Cite solution_lesson_node_00009: Explicit Uncertainty Signals in Stacking Ensembles.

        Args:
            base_probs_list (list of np.ndarray): List of arrays containing class probabilities
                                                  from base models. Each shape (n_samples, n_classes).
            meta_features (np.ndarray): Array of meta-features. Shape (n_samples, n_features).

        Returns:
            np.ndarray: Concatenated feature matrix.
        """
        # Ensure inputs are numpy arrays
        base_probs_list = [np.array(p) for p in base_probs_list]
        meta_features = np.array(meta_features)

        uncertainty_features = []
        epsilon = 1e-15

        # 1. Per-model uncertainty statistics
        for probs in base_probs_list:
            # Clip for numerical stability
            p = np.clip(probs, epsilon, 1 - epsilon)

            # Shannon Entropy: -sum(p * log(p))
            entropy = -np.sum(p * np.log(p), axis=1, keepdims=True)

            # Maximum Probability (Confidence)
            max_prob = np.max(p, axis=1, keepdims=True)

            # Standard Deviation of the probability distribution
            std_prob = np.std(p, axis=1, keepdims=True)

            uncertainty_features.extend([entropy, max_prob, std_prob])

        # 2. Cross-model uncertainty (Disagreement)
        if len(base_probs_list) > 1:
            # Stack models to shape (n_models, n_samples, n_classes)
            all_preds = np.stack(base_probs_list, axis=0)
            # Standard deviation across models for each class
            disagreement = np.std(all_preds, axis=0)  # Shape (n_samples, n_classes)
            uncertainty_features.append(disagreement)

        # Concatenate all features
        features = np.hstack(base_probs_list + [meta_features] + uncertainty_features)
        return features

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Trains the XGBoost meta-learner with early stopping.

        Args:
            X_train (np.ndarray): Training features (concatenated probs + meta).
            y_train (np.ndarray): Training labels (integers).
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels (integers).
        """
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Watchlist for monitoring performance during training
        watchlist = [(dtrain, "train"), (dval, "eval")]

        print("Starting Meta-Learner (XGBoost) training...")

        # Train the model
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_rounds,
            evals=watchlist,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=50,  # Print metrics every 50 rounds
        )

        print(f"Training finished. Best iteration: {self.model.best_iteration}")

        # Perform final validation to print full precision metrics
        self._validate(X_val, y_val)

    def predict(self, X):
        """
        Generates predictions using the trained meta-learner.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes).
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        dtest = xgb.DMatrix(X)

        # Use best iteration for prediction if available (handling different XGBoost versions)
        try:
            best_iteration = self.model.best_iteration
            # iteration_range is (start, end) exclusive
            preds = self.model.predict(dtest, iteration_range=(0, best_iteration + 1))
        except AttributeError:
            # Fallback if best_iteration is not available
            preds = self.model.predict(dtest)

        return preds

    def _validate(self, X_val, y_val):
        """
        Internal validation method to print final metrics with full precision.
        """
        preds = self.predict(X_val)

        # Clip probabilities to avoid log(0) and match metric definition
        clipped_preds = clip_probabilities(preds)

        # Calculate metrics
        # Labels for log_loss must be provided to handle potential missing classes in batch
        labels = list(range(len(Config.LABELS)))
        loss = log_loss(y_val, clipped_preds, labels=labels)

        pred_classes = np.argmax(preds, axis=1)
        acc = accuracy_score(y_val, pred_classes)

        print(f"Final Meta-Learner Validation Log Loss: {loss}")
        print(f"Final Meta-Learner Validation Accuracy: {acc}")

        return loss, acc
