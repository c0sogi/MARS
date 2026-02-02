import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from library.config import Config
from library.data import DataManager
from library.model import XGBModelWrapper


class SemiSupervisedPipeline:
    """
    Orchestrates the semi-supervised learning pipeline:
    1. Teacher Training (CV)
    2. Pseudo-Labeling
    3. Student Training (CV with Augmented Data)
    4. Submission Generation
    """

    def __init__(self):
        self.config = Config
        self.data_manager = DataManager()
        self.working_dir = self.config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def run_pipeline(self):
        """Executes the full pipeline."""
        print("Initializing Semi-Supervised Pipeline...")

        # 1. Load Data
        # We load the processed data. df_train here refers to the labeled training set.
        df_train, df_val, df_test = self.data_manager.load_and_preprocess(
            load_cached_data=True
        )

        # 2. Teacher Stage
        print("\n=== Stage 1: Teacher Training ===")
        teacher_probs_path = os.path.join(self.working_dir, "teacher_test_probs.npy")

        if os.path.exists(teacher_probs_path):
            print(f"Loading cached Teacher predictions from {teacher_probs_path}")
            teacher_test_probs = np.load(teacher_probs_path)
        else:
            # Train teacher ensemble on df_train
            teacher_test_probs = self.run_cv_stage(
                df_train, df_test, stage_name="Teacher"
            )
            np.save(teacher_probs_path, teacher_test_probs)

        # 3. Pseudo-Labeling
        print("\n=== Stage 2: Pseudo-Labeling ===")
        # Merge pseudo-labels. df_augmented contains original train + pseudo samples.
        df_augmented = self.data_manager.merge_pseudo_labels(
            df_train,
            df_test,
            teacher_test_probs,
            threshold=self.config.PSEUDO_LABEL_THRESHOLD,
        )

        # 4. Student Stage
        print("\n=== Stage 3: Student Training ===")
        student_probs_path = os.path.join(self.working_dir, "student_test_probs.npy")

        if os.path.exists(student_probs_path):
            print(f"Loading cached Student predictions from {student_probs_path}")
            student_test_probs = np.load(student_probs_path)
        else:
            # Train student ensemble.
            # We pass df_train to ensure validation folds are strictly from the original data.
            student_test_probs = self.run_student_stage(df_train, df_augmented, df_test)
            np.save(student_probs_path, student_test_probs)

        # 5. Submission
        print("\n=== Stage 4: Submission Generation ===")
        self.create_submission(student_test_probs, df_test)
        print("Pipeline completed successfully.")

    def _get_features_and_target(self, df):
        """Separates features and target. Excludes Id."""
        exclude_cols = [self.config.ID_COL, self.config.TARGET_COL]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols]
        y = None
        if self.config.TARGET_COL in df.columns:
            y = df[self.config.TARGET_COL]

        return X, y

    def run_cv_stage(self, df_train, df_test, stage_name="Teacher"):
        """
        Runs Stratified K-Fold CV.
        Returns averaged test probabilities.
        """
        folds = self.data_manager.get_folds(df_train, n_splits=self.config.CV_FOLDS)

        # Prepare Test Features
        X_test, _ = self._get_features_and_target(df_test)

        # Array to store test predictions from each fold
        test_probs_sum = np.zeros((len(df_test), self.config.NUM_CLASSES))

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"\n[{stage_name}] Fold {fold_idx + 1}/{self.config.CV_FOLDS}")

            # Split Data
            fold_train = df_train.iloc[train_idx]
            fold_val = df_train.iloc[val_idx]

            X_train, y_train = self._get_features_and_target(fold_train)
            X_val, y_val = self._get_features_and_target(fold_val)

            # Initialize and Train Model
            model = XGBModelWrapper()
            model.train(X_train, y_train, X_val, y_val)

            # Evaluate OOF
            val_probs = model.predict_proba(X_val)
            val_pred_labels = np.argmax(val_probs, axis=1)

            acc = accuracy_score(y_val, val_pred_labels)
            ll = log_loss(y_val, val_probs, labels=list(range(self.config.NUM_CLASSES)))
            print(
                f"[{stage_name}] Fold {fold_idx + 1} - Accuracy: {acc}, Log Loss: {ll}"
            )

            # Predict on Test
            test_probs = model.predict_proba(X_test)
            test_probs_sum += test_probs

            # Cleanup
            del model, X_train, y_train, X_val, y_val, fold_train, fold_val

        # Average predictions
        avg_test_probs = test_probs_sum / self.config.CV_FOLDS
        return avg_test_probs

    def run_student_stage(self, df_orig_train, df_augmented, df_test):
        """
        Runs Student training.
        Uses folds from df_orig_train for validation to prevent leakage.
        Adds pseudo-labeled data (from df_augmented) to training set.
        """
        # 1. Get folds based on ORIGINAL training data
        folds = self.data_manager.get_folds(
            df_orig_train, n_splits=self.config.CV_FOLDS
        )

        # 2. Identify indices of pseudo-labeled data in df_augmented
        # df_augmented is constructed as concat([df_orig_train, df_pseudo])
        n_orig = len(df_orig_train)
        n_augmented = len(df_augmented)
        pseudo_indices = np.arange(n_orig, n_augmented)

        # Prepare Test Features
        X_test, _ = self._get_features_and_target(df_test)

        test_probs_sum = np.zeros((len(df_test), self.config.NUM_CLASSES))

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"\n[Student] Fold {fold_idx + 1}/{self.config.CV_FOLDS}")

            # Construct Training Indices: Original Train Indices + All Pseudo Indices
            full_train_indices = np.concatenate([train_idx, pseudo_indices])

            # Validation Indices: Strictly Original Validation Indices
            full_val_indices = val_idx

            # Select Data
            fold_train = df_augmented.iloc[full_train_indices]
            fold_val = df_augmented.iloc[full_val_indices]

            X_train, y_train = self._get_features_and_target(fold_train)
            X_val, y_val = self._get_features_and_target(fold_val)

            print(
                f"Train Size: {len(X_train)} (Orig: {len(train_idx)} + Pseudo: {len(pseudo_indices)})"
            )
            print(f"Val Size: {len(X_val)}")

            # Train Model
            model = XGBModelWrapper()
            model.train(X_train, y_train, X_val, y_val)

            # Evaluate
            val_probs = model.predict_proba(X_val)
            val_pred_labels = np.argmax(val_probs, axis=1)

            acc = accuracy_score(y_val, val_pred_labels)
            ll = log_loss(y_val, val_probs, labels=list(range(self.config.NUM_CLASSES)))
            print(f"[Student] Fold {fold_idx + 1} - Accuracy: {acc}, Log Loss: {ll}")

            # Predict on Test
            test_probs = model.predict_proba(X_test)
            test_probs_sum += test_probs

            del model, X_train, y_train, X_val, y_val, fold_train, fold_val

        avg_test_probs = test_probs_sum / self.config.CV_FOLDS
        return avg_test_probs

    def create_submission(self, test_probs, df_test):
        """
        Generates the submission CSV.
        """
        # Get class indices with highest probability
        pred_indices = np.argmax(test_probs, axis=1)

        # Map back to original class labels
        pred_labels = [self.config.INVERSE_CLASS_MAPPING[i] for i in pred_indices]

        # Create DataFrame
        submission = pd.DataFrame(
            {
                self.config.ID_COL: df_test[self.config.ID_COL],
                self.config.TARGET_COL: pred_labels,
            }
        )

        # Save
        save_path = self.config.SUBMISSION_PATH
        print(f"Saving submission to {save_path}...")
        submission.to_csv(save_path, index=False)
        print("Submission saved.")
