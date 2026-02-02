import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import log_loss
from library.model_factory import (
    get_linear_expert,
    get_generative_expert,
    get_kernel_expert,
    get_meta_learner,
)


class StackingEnsemble:
    """
    Implements a 2-Level Stacking Ensemble for Leaf Classification.

    Level 0:
        1. Discriminative Linear Expert (LogisticRegressionCV)
        2. Generative Linear Expert (LDA)
        3. Discriminative Non-Linear Expert (Nystroem Kernel LogReg)

    Level 1:
        - Meta-Learner (Logistic Regression)
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        # Initialize base learners from factory
        self.base_models = {
            "linear": get_linear_expert(random_state),
            "generative": get_generative_expert(random_state),
            "kernel": get_kernel_expert(random_state),
        }
        # Initialize meta learner
        self.meta_learner = get_meta_learner(random_state)

        # State variables
        self.n_classes = None
        self.trained_base_models = {}
        self.is_meta_trained = False

    def generate_oof_predictions(self, X, y):
        """
        Generates Out-of-Fold (OOF) probability predictions using 3-Fold Stratified CV.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.

        Returns:
            np.ndarray: Matrix of OOF predictions (n_samples, n_models * n_classes).
        """
        self.n_classes = len(np.unique(y))
        n_samples = X.shape[0]
        n_models = len(self.base_models)

        # Initialize OOF matrix: [Model1_Probs | Model2_Probs | Model3_Probs]
        oof_preds = np.zeros((n_samples, n_models * self.n_classes))

        # 3-Fold Stratified CV as per design
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

        print(f"Starting OOF prediction generation with 3 folds...")
        model_scores = {name: [] for name in self.base_models}

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Iterate through base models in fixed order
            for i, (name, model) in enumerate(self.base_models.items()):
                # Clone model to ensure fresh training for this fold
                clf = clone(model)
                clf.fit(X_train, y_train)

                # Predict probabilities
                val_probs = clf.predict_proba(X_val)

                # Store in OOF matrix
                # Calculate column indices for this model
                start_col = i * self.n_classes
                end_col = (i + 1) * self.n_classes
                oof_preds[val_idx, start_col:end_col] = val_probs

                # Calculate fold score
                score = log_loss(y_val, val_probs, labels=np.arange(self.n_classes))
                model_scores[name].append(score)

        # Report performance
        print("OOF Generation Complete. Average Base Model Log Loss:")
        for name, scores in model_scores.items():
            print(f"  {name}: {np.mean(scores)}")

        return oof_preds

    def train_meta_learner(self, oof_preds, y):
        """
        Trains the Level 1 Meta-Learner on OOF predictions.

        Args:
            oof_preds (np.ndarray): The OOF predictions from generate_oof_predictions.
            y (np.ndarray): The true target labels.
        """
        print("Training Level 1 Meta-Learner...")
        self.meta_learner.fit(oof_preds, y)
        self.is_meta_trained = True

        # Evaluate in-sample fit (sanity check)
        meta_probs = self.meta_learner.predict_proba(oof_preds)
        loss = log_loss(y, meta_probs, labels=np.arange(self.n_classes))
        print(f"Meta-Learner In-Sample Log Loss: {loss}")

    def train_full_base_models(self, X, y):
        """
        Retrains all base models on the full dataset (Train + Val).

        Args:
            X (np.ndarray): Full training features.
            y (np.ndarray): Full training labels.
        """
        print("Retraining all base models on full dataset...")
        self.n_classes = len(np.unique(y))

        for name, model in self.base_models.items():
            print(f"  Fitting {name}...")
            # We fit the instances stored in self.base_models directly
            # These will be used for final inference
            model.fit(X, y)
            self.trained_base_models[name] = model

    def predict(self, X):
        """
        Generates final predictions for test data.

        Args:
            X (np.ndarray): Test features.

        Returns:
            np.ndarray: Final predicted probabilities (n_samples, n_classes).
        """
        if not self.is_meta_trained:
            raise RuntimeError(
                "Meta-learner is not trained. Call train_meta_learner first."
            )

        if not self.trained_base_models:
            raise RuntimeError(
                "Base models are not retrained. Call train_full_base_models first."
            )

        n_samples = X.shape[0]
        n_models = len(self.trained_base_models)

        # Generate meta-features for test set
        # Must maintain same order: Linear -> Generative -> Kernel
        test_meta_features = np.zeros((n_samples, n_models * self.n_classes))

        for i, (name, model) in enumerate(self.trained_base_models.items()):
            probs = model.predict_proba(X)
            start_col = i * self.n_classes
            end_col = (i + 1) * self.n_classes
            test_meta_features[:, start_col:end_col] = probs

        # Final prediction using meta-learner
        print("Generating final ensemble predictions...")
        final_probs = self.meta_learner.predict_proba(test_meta_features)

        return final_probs
