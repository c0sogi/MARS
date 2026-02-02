import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from library.utils import set_seed, score_predictions
from library.data_loader import LeafDataManager
from library.models import LGBMWrapper, LDAWrapper, SVMWrapper
from library.ensemble import WeightOptimizer


class CrossValidationPipeline:
    """
    Manages the Stratified K-Fold Cross-Validation, Ensemble Optimization,
    and Final Submission generation for the Leaf Classification task.
    """

    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        set_seed(random_state)
        self.data_manager = LeafDataManager(seed=random_state)

    def run(self, load_cached_data=True):
        """
        Executes the full pipeline:
        1. Load and combine data.
        2. Run CV to train models and generate OOF predictions.
        3. Optimize ensemble weights.
        4. Generate test predictions using CV-bagging.
        5. Save submission.
        """
        print("Initializing Pipeline...")

        # 1. Load Data
        # We load both train and val splits provided by the metadata and combine them
        # to perform our own Stratified K-Fold CV on the entire available labeled data.

        # Tree Data (Raw)
        X_train_tree, y_train = self.data_manager.get_train_data(
            model_type="tree", load_cached_data=load_cached_data
        )
        X_val_tree, y_val = self.data_manager.get_val_data(
            model_type="tree", load_cached_data=load_cached_data
        )
        X_test_tree, test_ids = self.data_manager.get_test_data(
            model_type="tree", load_cached_data=load_cached_data
        )

        # Linear/Kernel Data (Transformed)
        X_train_lin, _ = self.data_manager.get_train_data(
            model_type="linear_kernel", load_cached_data=load_cached_data
        )
        X_val_lin, _ = self.data_manager.get_val_data(
            model_type="linear_kernel", load_cached_data=load_cached_data
        )
        X_test_lin, _ = self.data_manager.get_test_data(
            model_type="linear_kernel", load_cached_data=load_cached_data
        )

        # Classes
        classes = self.data_manager.get_classes(load_cached_data=load_cached_data)

        # Combine Train + Val
        X_full_tree = np.vstack((X_train_tree, X_val_tree))
        X_full_lin = np.vstack((X_train_lin, X_val_lin))
        y_full = np.concatenate((y_train, y_val))

        print(
            f"Data Loaded. Full Training Set: {X_full_tree.shape[0]} samples. Test Set: {X_test_tree.shape[0]} samples."
        )

        # 2. Initialize Arrays
        n_samples = X_full_tree.shape[0]
        n_classes = len(classes)
        n_test = len(test_ids)

        # OOF Predictions
        oof_lgbm = np.zeros((n_samples, n_classes))
        oof_lda = np.zeros((n_samples, n_classes))
        oof_svm = np.zeros((n_samples, n_classes))

        # Test Predictions (Accumulators for averaging)
        pred_test_lgbm = np.zeros((n_test, n_classes))
        pred_test_lda = np.zeros((n_test, n_classes))
        pred_test_svm = np.zeros((n_test, n_classes))

        # 3. Cross-Validation Loop
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        print(f"Starting {self.n_folds}-Fold Stratified Cross-Validation...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_full_tree, y_full)):
            print(f"\nProcessing Fold {fold + 1}...")

            # Split Data
            # Tree
            X_tr_tree, X_va_tree = X_full_tree[train_idx], X_full_tree[val_idx]
            # Linear
            X_tr_lin, X_va_lin = X_full_lin[train_idx], X_full_lin[val_idx]
            # Target
            y_tr, y_va = y_full[train_idx], y_full[val_idx]

            # --- Model 1: LightGBM ---
            # Uses raw features (Tree data)
            model_lgbm = LGBMWrapper(random_state=self.random_state)
            model_lgbm.fit(X_tr_tree, y_tr, X_val=X_va_tree, y_val=y_va)

            # Predict
            oof_lgbm[val_idx] = model_lgbm.predict_proba(X_va_tree)
            pred_test_lgbm += model_lgbm.predict_proba(X_test_tree) / self.n_folds

            # --- Model 2: LDA ---
            # Uses transformed features (Linear data)
            model_lda = LDAWrapper(random_state=self.random_state)
            model_lda.fit(X_tr_lin, y_tr, X_val=X_va_lin, y_val=y_va)

            # Predict
            oof_lda[val_idx] = model_lda.predict_proba(X_va_lin)
            pred_test_lda += model_lda.predict_proba(X_test_lin) / self.n_folds

            # --- Model 3: SVM ---
            # Uses transformed features (Linear data)
            model_svm = SVMWrapper(random_state=self.random_state)
            model_svm.fit(X_tr_lin, y_tr, X_val=X_va_lin, y_val=y_va)

            # Predict
            oof_svm[val_idx] = model_svm.predict_proba(X_va_lin)
            pred_test_svm += model_svm.predict_proba(X_test_lin) / self.n_folds

        # 4. Evaluate OOF Performance
        print("\n--- OOF Validation Scores (Log Loss) ---")
        # y_full is integer-encoded, so we must pass integer labels (indices) to log_loss
        class_indices = np.arange(len(classes))
        score_lgbm = score_predictions(y_full, oof_lgbm, classes=class_indices)
        score_lda = score_predictions(y_full, oof_lda, classes=class_indices)
        score_svm = score_predictions(y_full, oof_svm, classes=class_indices)

        print(f"LightGBM: {score_lgbm}")
        print(f"LDA:      {score_lda}")
        print(f"SVM:      {score_svm}")

        # 5. Optimize Weights
        print("\nOptimizing Ensemble Weights...")
        optimizer = WeightOptimizer(random_state=self.random_state)
        optimizer.fit([oof_lgbm, oof_lda, oof_svm], y_full, classes=class_indices)

        # 6. Generate Final Predictions
        print("\nGenerating Final Submission...")
        final_test_preds = optimizer.predict(
            [pred_test_lgbm, pred_test_lda, pred_test_svm]
        )

        # 7. Save Submission
        self._save_submission(test_ids, final_test_preds, classes)

    def _save_submission(self, test_ids, preds, classes):
        """
        Saves the predictions to a CSV file in the required format.
        """
        # Create DataFrame
        submission_df = pd.DataFrame(preds, columns=classes)

        # Insert ID column
        submission_df.insert(0, "id", test_ids)

        # Ensure directory exists
        os.makedirs("./submission", exist_ok=True)
        output_path = "./submission/submission.csv"

        # Save
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
