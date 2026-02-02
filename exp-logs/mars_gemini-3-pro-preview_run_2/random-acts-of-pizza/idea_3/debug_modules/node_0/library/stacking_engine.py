import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import setup_logger, timer
from library.model_factory import (
    get_linear_base_model,
    get_tree_base_model,
    get_meta_model,
)


class StackingEngine:
    """
    Orchestrates the Dual-Branch Stacking Ensemble.
    Manages Cross-Validation, OOF generation, Meta-Learner training,
    and Final Retraining of base models.
    """

    def __init__(self):
        self.logger = setup_logger("stacking_engine")
        self.n_folds = Config.N_FOLDS
        self.seed = Config.SEED

        # Placeholders for trained models
        self.meta_model = None
        self.final_linear_model = None
        self.final_tree_model = None

        # Placeholder for optimal tree iterations found during CV
        self.avg_best_iteration = None

    def train_cv(self, X_linear, X_tree, y):
        """
        Performs Stratified K-Fold CV to generate OOF predictions and determine optimal hyperparameters (e.g., n_estimators).

        Args:
            X_linear (np.ndarray): Features for the Linear Branch (Branch A).
            X_tree (np.ndarray): Features for the Tree Branch (Branch B).
            y (np.ndarray): Target labels.

        Returns:
            tuple: (oof_preds_linear, oof_preds_tree, y) - The OOF predictions and aligned targets.
        """
        self.logger.info(f"Starting {self.n_folds}-Fold Cross-Validation...")

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )

        # Arrays to store OOF predictions
        oof_preds_linear = np.zeros(len(y))
        oof_preds_tree = np.zeros(len(y))

        best_iterations = []
        fold_aucs_linear = []
        fold_aucs_tree = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_linear, y)):
            with timer(f"Fold {fold + 1}", self.logger):
                # Split Data
                X_lin_train, X_lin_val = X_linear[train_idx], X_linear[val_idx]
                X_tree_train, X_tree_val = X_tree[train_idx], X_tree[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # ----------------------------------------
                # Branch A: Linear Model (Logistic Regression)
                # ----------------------------------------
                linear_model = get_linear_base_model()
                linear_model.fit(X_lin_train, y_train)

                # Predict (Probability of class 1)
                p_lin_val = linear_model.predict_proba(X_lin_val)[:, 1]
                oof_preds_linear[val_idx] = p_lin_val

                auc_lin = roc_auc_score(y_val, p_lin_val)
                fold_aucs_linear.append(auc_lin)

                # ----------------------------------------
                # Branch B: Tree Model (LightGBM)
                # ----------------------------------------
                tree_model = get_tree_base_model()

                # Setup Early Stopping
                callbacks = [
                    lgb.early_stopping(
                        stopping_rounds=Config.LGBM_EARLY_STOPPING_ROUNDS, verbose=False
                    ),
                    lgb.log_evaluation(period=0),  # Suppress extensive logging
                ]

                tree_model.fit(
                    X_tree_train,
                    y_train,
                    eval_set=[(X_tree_val, y_val)],
                    eval_metric="auc",
                    callbacks=callbacks,
                )

                # Predict
                p_tree_val = tree_model.predict_proba(X_tree_val)[:, 1]
                oof_preds_tree[val_idx] = p_tree_val

                auc_tree = roc_auc_score(y_val, p_tree_val)
                fold_aucs_tree.append(auc_tree)

                # Record best iteration
                # LightGBM sklearn API stores best_iteration_
                best_iterations.append(tree_model.best_iteration_)

                self.logger.info(f"  Fold {fold + 1} AUC - Linear: {auc_lin}")
                self.logger.info(f"  Fold {fold + 1} AUC - Tree:   {auc_tree}")

        # Compute Global Metrics
        total_auc_linear = roc_auc_score(y, oof_preds_linear)
        total_auc_tree = roc_auc_score(y, oof_preds_tree)

        self.logger.info("-" * 40)
        self.logger.info(f"CV Complete.")
        self.logger.info(f"OOF AUC - Linear Branch: {total_auc_linear}")
        self.logger.info(f"OOF AUC - Tree Branch:   {total_auc_tree}")

        self.avg_best_iteration = int(np.mean(best_iterations))
        self.logger.info(
            f"Average Best Iteration for Tree Model: {self.avg_best_iteration}"
        )
        self.logger.info("-" * 40)

        return oof_preds_linear, oof_preds_tree

    def train_meta_learner(self, oof_linear, oof_tree, y):
        """
        Trains the Level-2 Meta-Learner (Logistic Regression) on OOF predictions.
        """
        self.logger.info("Training Meta-Learner...")

        # Stack predictions to form meta-features
        X_meta = np.column_stack([oof_linear, oof_tree])

        self.meta_model = get_meta_model()
        self.meta_model.fit(X_meta, y)

        # Evaluate fit on the training set (OOF)
        meta_preds = self.meta_model.predict_proba(X_meta)[:, 1]
        meta_auc = roc_auc_score(y, meta_preds)

        self.logger.info(f"Meta-Learner OOF AUC: {meta_auc}")
        self.logger.info(
            f"Meta-Learner Coefficients: Linear={self.meta_model.coef_[0][0]}, Tree={self.meta_model.coef_[0][1]}"
        )

    def train_final_base_models(self, X_linear, X_tree, y):
        """
        Retrains the base models on the full training dataset.
        Uses the average best iteration from CV for the Tree model.
        """
        self.logger.info("Retraining Base Models on Full Dataset...")

        # 1. Final Linear Model
        self.final_linear_model = get_linear_base_model()
        self.final_linear_model.fit(X_linear, y)

        # 2. Final Tree Model
        # Update n_estimators to the average best iteration found in CV
        if self.avg_best_iteration is None:
            self.logger.warning(
                "Average best iteration not set. Using default n_estimators."
            )
            n_estimators = Config.LGBM_PARAMS["n_estimators"]
        else:
            n_estimators = self.avg_best_iteration

        self.logger.info(f"Retraining Tree Model with n_estimators={n_estimators}")

        self.final_tree_model = get_tree_base_model(n_estimators=n_estimators)
        # No early stopping here as we use the fixed optimal number of rounds
        self.final_tree_model.fit(X_tree, y)

        self.logger.info("Final base models trained.")

    def predict(self, X_test_linear, X_test_tree):
        """
        Generates predictions for the test set using the stacked ensemble.

        Args:
            X_test_linear: Test features for Linear Branch.
            X_test_tree: Test features for Tree Branch.

        Returns:
            np.ndarray: Final probability predictions.
        """
        if (
            self.final_linear_model is None
            or self.final_tree_model is None
            or self.meta_model is None
        ):
            raise RuntimeError(
                "Models not trained. Run fit() or individual train methods first."
            )

        self.logger.info("Generating predictions on Test Set...")

        # 1. Base Model Predictions
        preds_linear = self.final_linear_model.predict_proba(X_test_linear)[:, 1]
        preds_tree = self.final_tree_model.predict_proba(X_test_tree)[:, 1]

        # 2. Stack Predictions
        X_meta_test = np.column_stack([preds_linear, preds_tree])

        # 3. Meta-Learner Prediction
        final_preds = self.meta_model.predict_proba(X_meta_test)[:, 1]

        return final_preds

    def run(self, data):
        """
        Executes the full training and prediction pipeline.

        Args:
            data (dict): Dictionary containing all processed data arrays
                         (train_linear, train_tree, y_train, test_linear, test_tree).

        Returns:
            np.ndarray: Predictions for the test set.
        """
        # Extract Training Data
        X_lin = data["train_linear"]
        X_tree = data["train_tree"]
        y = data["y_train"]

        # 1. Cross-Validation (Level 1)
        oof_linear, oof_tree = self.train_cv(X_lin, X_tree, y)

        # 2. Train Meta-Learner (Level 2)
        self.train_meta_learner(oof_linear, oof_tree, y)

        # 3. Retrain Base Models on Full Data
        self.train_final_base_models(X_lin, X_tree, y)

        # 4. Predict on Test Data
        X_test_lin = data["test_linear"]
        X_test_tree = data["test_tree"]

        predictions = self.predict(X_test_lin, X_test_tree)

        return predictions
