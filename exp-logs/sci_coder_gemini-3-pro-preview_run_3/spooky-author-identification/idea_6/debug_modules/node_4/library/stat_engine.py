import numpy as np
import scipy.sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import os

from library.config import Config
from library.features import get_data
from library.utils import seed_everything, calculate_log_loss


class StatisticalTrainer:
    """
    Trainer for the Statistical Branch (TF-IDF + Stylometric Features + Logistic Regression).
    Handles Cross-Validation, OOF generation, and Test prediction.
    """

    def __init__(self, n_folds=Config.N_FOLDS, random_state=Config.SEED):
        """
        Args:
            n_folds (int): Number of cross-validation folds.
            random_state (int): Random seed for reproducibility.
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.model_params = {
            "solver": "saga",
            "multi_class": "multinomial",
            "C": 1.0,
            "max_iter": 1000,
            "random_state": random_state,
            "class_weight": "balanced",
            "n_jobs": -1,
        }

    def run_cv(
        self,
        X_train=None,
        y_train=None,
        X_test=None,
        test_ids=None,
        load_cached_data=True,
        debug=Config.DEBUG,
    ):
        """
        Executes the Stratified K-Fold Cross-Validation pipeline.

        Args:
            X_train (scipy.sparse.csr_matrix, optional): Training features.
            y_train (np.array, optional): Training labels.
            X_test (scipy.sparse.csr_matrix, optional): Test features.
            test_ids (np.array, optional): Test IDs.
            load_cached_data (bool): Whether to load features from cache if inputs are None.
            debug (bool): Whether to run in debug mode.

        Returns:
            tuple: (oof_preds, test_preds_avg, y_train_sorted, test_ids)
                oof_preds: Out-Of-Fold predictions for the training set.
                test_preds_avg: Averaged predictions for the test set.
                y_train_sorted: Labels corresponding to oof_preds (if shuffling occurred).
                test_ids: IDs corresponding to test_preds.
        """
        seed_everything(self.random_state)

        # 1. Load Data if not provided
        if X_train is None or y_train is None or X_test is None:
            (
                X_train_part,
                y_train_part,
                X_val_part,
                y_val_part,
                X_test_loaded,
                test_ids_loaded,
            ) = get_data(load_cached_data=load_cached_data, debug=debug)

            # Combine provided Train and Val splits to perform our own K-Fold
            # This ensures we use all available labeled data for CV
            print("Combining pre-split Train and Validation sets for K-Fold CV...")
            X_train = scipy.sparse.vstack([X_train_part, X_val_part]).tocsr()
            y_train = np.concatenate([y_train_part, y_val_part])
            X_test = X_test_loaded
            test_ids = test_ids_loaded

        # 2. Initialize Arrays
        n_samples = X_train.shape[0]
        n_classes = Config.NUM_CLASSES
        n_test = X_test.shape[0]

        oof_preds = np.zeros((n_samples, n_classes))
        test_preds_sum = np.zeros((n_test, n_classes))

        # 3. Stratified K-Fold
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        print(f"Starting {self.n_folds}-Fold Stratified Cross-Validation...")

        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            # Split Data
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            # Initialize Model
            model = LogisticRegression(**self.model_params)

            # Train
            model.fit(X_tr, y_tr)

            # Predict (Validation)
            val_probs = model.predict_proba(X_val)
            oof_preds[val_idx] = val_probs

            # Predict (Test)
            test_probs = model.predict_proba(X_test)
            test_preds_sum += test_probs

            # Evaluate Fold
            fold_loss = calculate_log_loss(y_val, val_probs)
            fold_metrics.append(fold_loss)

            print(f"Fold {fold + 1}/{self.n_folds} | Log Loss: {fold_loss}")

        # 4. Aggregate Results
        test_preds_avg = test_preds_sum / self.n_folds
        overall_loss = calculate_log_loss(y_train, oof_preds)

        print("\n=== Statistical Model CV Results ===")
        print(f"Overall Log Loss: {overall_loss}")
        print(f"Average Fold Log Loss: {np.mean(fold_metrics)}")

        return oof_preds, test_preds_avg, y_train, test_ids

    def train_on_augmented(
        self,
        X_train_aug,
        y_train_aug,
        X_val_orig,
        y_val_orig,
        X_test,
        test_ids,
        load_cached_data=True,
    ):
        """
        Trains the model on an augmented dataset (original + pseudo-labeled) and
        evaluates on the original validation set. Used for Stage 2.

        Args:
            X_train_aug: Augmented training features.
            y_train_aug: Augmented training labels.
            X_val_orig: Original validation features (for metric tracking).
            y_val_orig: Original validation labels.
            X_test: Test features.
            test_ids: Test IDs.

        Returns:
            tuple: (val_preds, test_preds, val_loss)
        """
        seed_everything(self.random_state)

        print("Training Statistical Model on Augmented Dataset...")

        # Initialize Model
        model = LogisticRegression(**self.model_params)

        # Train on full augmented data
        model.fit(X_train_aug, y_train_aug)

        # Predict on Original Validation Set (to check for drift/performance)
        val_preds = model.predict_proba(X_val_orig)
        val_loss = calculate_log_loss(y_val_orig, val_preds)

        # Predict on Test Set
        test_preds = model.predict_proba(X_test)

        print(f"Augmented Training Complete. Validation Log Loss: {val_loss}")

        return val_preds, test_preds, val_loss
