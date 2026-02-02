import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from library.config import (
    RF_ESTIMATORS,
    RF_CLASS_WEIGHT,
    RF_N_JOBS,
    LR_C,
    LR_PENALTY,
    LR_SOLVER,
    LR_MAX_ITER,
    LR_CLASS_WEIGHT,
    FUSION_WEIGHT_RF,
    FUSION_WEIGHT_LR,
    RANDOM_SEED,
)


class HybridEnsemble:
    """
    A Hybrid Ensemble model that combines a Random Forest (Sparse Stream)
    and a Logistic Regression (Dense Stream) via weighted averaging.
    """

    def __init__(
        self,
        rf_estimators=RF_ESTIMATORS,
        rf_class_weight=RF_CLASS_WEIGHT,
        rf_n_jobs=RF_N_JOBS,
        lr_c=LR_C,
        lr_penalty=LR_PENALTY,
        lr_solver=LR_SOLVER,
        lr_max_iter=LR_MAX_ITER,
        lr_class_weight=LR_CLASS_WEIGHT,
        fusion_weight_rf=FUSION_WEIGHT_RF,
        fusion_weight_lr=FUSION_WEIGHT_LR,
        random_state=RANDOM_SEED,
    ):
        """
        Initializes the ensemble with specific hyperparameters for both branches.
        """
        self.random_state = random_state
        self.fusion_weight_rf = fusion_weight_rf
        self.fusion_weight_lr = fusion_weight_lr

        # Initialize Sparse-Tabular Learner (Random Forest)
        self.rf_model = RandomForestClassifier(
            n_estimators=rf_estimators,
            class_weight=rf_class_weight,
            n_jobs=rf_n_jobs,
            random_state=self.random_state,
            verbose=0,
        )

        # Initialize Dense-Semantic Learner (Logistic Regression)
        self.lr_model = LogisticRegression(
            C=lr_c,
            penalty=lr_penalty,
            solver=lr_solver,
            max_iter=lr_max_iter,
            class_weight=lr_class_weight,
            random_state=self.random_state,
            verbose=0,
        )

    def fit(
        self,
        X_sparse_train,
        X_dense_train,
        y_train,
        X_sparse_val=None,
        X_dense_val=None,
        y_val=None,
    ):
        """
        Trains both models on their respective data streams.
        Optionally evaluates on validation data if provided.
        """
        print("Training Random Forest on Sparse Stream...")
        self.rf_model.fit(X_sparse_train, y_train)

        print("Training Logistic Regression on Dense Stream...")
        self.lr_model.fit(X_dense_train, y_train)

        # Evaluation if validation data is provided
        if X_sparse_val is not None and X_dense_val is not None and y_val is not None:
            print("Evaluating model on validation set...")

            # Get individual predictions
            rf_probs = self.rf_model.predict_proba(X_sparse_val)[:, 1]
            lr_probs = self.lr_model.predict_proba(X_dense_val)[:, 1]

            # Calculate individual metrics
            rf_auc = roc_auc_score(y_val, rf_probs)
            lr_auc = roc_auc_score(y_val, lr_probs)

            # Get fused predictions
            fused_probs = (
                self.fusion_weight_rf * rf_probs + self.fusion_weight_lr * lr_probs
            )
            fused_auc = roc_auc_score(y_val, fused_probs)

            print(f"Random Forest Validation AUC: {rf_auc}")
            print(f"Logistic Regression Validation AUC: {lr_auc}")
            print(f"Hybrid Ensemble Validation AUC: {fused_auc}")

        return self

    def predict_proba(self, X_sparse, X_dense):
        """
        Generates fused probability predictions.

        Args:
            X_sparse: Sparse matrix for RF
            X_dense: Dense array for LR

        Returns:
            np.array: Probability of the positive class (1).
        """
        # Get probabilities for positive class (index 1)
        rf_probs = self.rf_model.predict_proba(X_sparse)[:, 1]
        lr_probs = self.lr_model.predict_proba(X_dense)[:, 1]

        # Weighted Average Fusion
        final_probs = (
            self.fusion_weight_rf * rf_probs + self.fusion_weight_lr * lr_probs
        )

        return final_probs
