import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import log_loss
from library.config import Config


class HybridEnsemble:
    """
    Implements the Hybrid Deep-Linear Ensemble model.
    Fuses standardized tabular features with PCA-reduced deep image features.
    Uses an ensemble of Regularized Logistic Regression and LDA.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.lr_model = None
        self.lda_model = None
        self.classes_ = None
        self.best_C = None
        self.lda_weight = 0.5

    def train_models(
        self, X_train_tab, X_train_img, y_train, X_val_tab, X_val_img, y_val
    ):
        """
        Trains the ensemble pipeline.

        1. Performs hyperparameter tuning on the Training set using the Validation set.
        2. Reports validation metrics.
        3. Refits models on the combined (Train + Val) dataset for final inference.

        Args:
            X_train_tab (np.ndarray): Training tabular features.
            X_train_img (np.ndarray): Training image PCA features.
            y_train (np.ndarray): Training targets.
            X_val_tab (np.ndarray): Validation tabular features.
            X_val_img (np.ndarray): Validation image PCA features.
            y_val (np.ndarray): Validation targets.
        """
        print("--- Starting Validation Phase ---")

        # 1. Preprocessing for Validation
        # Fit scaler on Train only
        scaler_val = StandardScaler()
        X_train_tab_sc = scaler_val.fit_transform(X_train_tab)
        X_val_tab_sc = scaler_val.transform(X_val_tab)

        # Feature Fusion
        X_train_fused = np.hstack([X_train_tab_sc, X_train_img])
        X_val_fused = np.hstack([X_val_tab_sc, X_val_img])

        print(f"Fused Training Data Shape: {X_train_fused.shape}")
        print(f"Fused Validation Data Shape: {X_val_fused.shape}")

        # 2. Logistic Regression Tuning
        print("Tuning Logistic Regression with GridSearchCV...")

        # Prepare PredefinedSplit: -1 for train samples, 0 for validation samples
        X_comb_val = np.vstack([X_train_fused, X_val_fused])
        y_comb_val = np.concatenate([y_train, y_val])
        test_fold = np.concatenate(
            [
                np.full(X_train_fused.shape[0], -1),
                np.zeros(X_val_fused.shape[0], dtype=int),
            ]
        )
        ps = PredefinedSplit(test_fold)

        lr = LogisticRegression(
            solver=Config.LOG_REG_SOLVER,
            multi_class="multinomial",
            max_iter=Config.LOG_REG_MAX_ITER,
            random_state=Config.SEED,
            n_jobs=-1,
        )

        param_grid = {"C": Config.LOG_REG_C_GRID}

        # Refit=False because we want to manually handle the final training
        gs = GridSearchCV(
            lr, param_grid, cv=ps, scoring="neg_log_loss", n_jobs=-1, refit=False
        )
        gs.fit(X_comb_val, y_comb_val)

        self.best_C = gs.best_params_["C"]
        # best_score_ is negative log loss
        print(f"Best C found: {self.best_C}")
        print(f"Best Validation Log Loss (LR GridSearch): {-gs.best_score_}")

        # 3. Component Evaluation on Validation Set

        def safe_log_loss(y_true, y_pred, classes_seen):
            """
            Calculates log loss only on samples where the true label is in the model's classes.
            Useful for debug runs where training/validation sets might have disjoint classes.
            """
            y_true = np.array(y_true)
            mask = np.isin(y_true, classes_seen)
            if not np.any(mask):
                return np.nan
            # Pass explicit labels to ensure column mapping is correct
            return log_loss(y_true[mask], y_pred[mask], labels=classes_seen)

        # Train fresh LR on Train-only with best C to get clean predictions for ensemble
        lr_clean = LogisticRegression(
            C=self.best_C,
            solver=Config.LOG_REG_SOLVER,
            multi_class="multinomial",
            max_iter=Config.LOG_REG_MAX_ITER,
            random_state=Config.SEED,
            n_jobs=-1,
        )
        lr_clean.fit(X_train_fused, y_train)
        lr_preds_val = lr_clean.predict_proba(X_val_fused)
        lr_loss_val = safe_log_loss(y_val, lr_preds_val, lr_clean.classes_)
        print(f"Logistic Regression Validation Log Loss: {lr_loss_val}")

        # Train LDA on Train-only
        lda_clean = LinearDiscriminantAnalysis(
            solver=Config.LDA_SOLVER, shrinkage=Config.LDA_SHRINKAGE
        )
        lda_clean.fit(X_train_fused, y_train)
        lda_preds_val = lda_clean.predict_proba(X_val_fused)
        lda_loss_val = safe_log_loss(y_val, lda_preds_val, lda_clean.classes_)
        print(f"LDA Validation Log Loss: {lda_loss_val}")

        # Ensemble Evaluation & Weight Optimization
        print("Optimizing ensemble weights...")
        best_loss = float("inf")
        best_w = 0.5

        # Search range [0, 1] with step 0.01
        for w in np.linspace(0, 1, 101):
            ens_preds = w * lda_preds_val + (1 - w) * lr_preds_val
            loss = safe_log_loss(y_val, ens_preds, lr_clean.classes_)
            if loss < best_loss:
                best_loss = loss
                best_w = w

        self.lda_weight = best_w
        print(f"Best LDA Weight: {self.lda_weight:.2f}")
        print(f"Optimized Hybrid Ensemble Validation Log Loss: {best_loss}")

        # 4. Final Retraining (Train + Val)
        print("--- Starting Final Retraining Phase (Train + Val) ---")

        # Combine Raw Data
        X_all_tab = np.vstack([X_train_tab, X_val_tab])
        X_all_img = np.vstack([X_train_img, X_val_img])
        y_all = np.concatenate([y_train, y_val])

        # Fit Scaler on Combined Tabular Data
        self.scaler.fit(X_all_tab)
        X_all_tab_sc = self.scaler.transform(X_all_tab)

        # Fuse Combined Data
        X_all_fused = np.hstack([X_all_tab_sc, X_all_img])

        # Fit Final Logistic Regression
        self.lr_model = LogisticRegression(
            C=self.best_C,
            solver=Config.LOG_REG_SOLVER,
            multi_class="multinomial",
            max_iter=Config.LOG_REG_MAX_ITER,
            random_state=Config.SEED,
            n_jobs=-1,
        )
        self.lr_model.fit(X_all_fused, y_all)

        # Fit Final LDA
        self.lda_model = LinearDiscriminantAnalysis(
            solver=Config.LDA_SOLVER, shrinkage=Config.LDA_SHRINKAGE
        )
        self.lda_model.fit(X_all_fused, y_all)

        # Store classes for prediction mapping
        self.classes_ = self.lr_model.classes_
        print("Final models retrained successfully.")

    def predict_ensemble(self, X_test_tab, X_test_img):
        """
        Generates ensemble predictions for the test set.

        Args:
            X_test_tab (np.ndarray): Test tabular features.
            X_test_img (np.ndarray): Test image PCA features.

        Returns:
            tuple: (predictions, class_names)
                predictions (np.ndarray): Probability matrix (N, n_classes).
                class_names (np.ndarray): Array of class names.
        """
        # Preprocess Test Data
        # Scale tabular features using the scaler fitted on (Train + Val)
        X_test_tab_sc = self.scaler.transform(X_test_tab)

        # Fuse
        X_test_fused = np.hstack([X_test_tab_sc, X_test_img])

        # Generate Predictions
        lr_preds = self.lr_model.predict_proba(X_test_fused)
        lda_preds = self.lda_model.predict_proba(X_test_fused)

        # Average Predictions (Weighted Ensemble)
        ensemble_preds = self.lda_weight * lda_preds + (1 - self.lda_weight) * lr_preds

        return ensemble_preds, self.classes_
