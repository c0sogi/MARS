import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score
from library import config, model, data


class EnsemblePipeline:
    """
    Manages the self-training ensemble pipeline, including Cross-Validation,
    Pseudo-Labeling, and Submission generation.
    """

    def __init__(self):
        pass

    def run_cv_training(self, X, y, X_test, n_original_samples=None):
        """
        Runs Stratified K-Fold Cross-Validation using XGBTrainer.

        Supports Semi-Supervised Learning:
        If n_original_samples is provided, the validation set for each fold is
        drawn strictly from the first n_original_samples (the true labeled data).
        Any data beyond n_original_samples (pseudo-labels) is added to the
        training set of every fold.

        Args:
            X (pd.DataFrame): Training features (potentially augmented).
            y (pd.Series): Training targets (potentially augmented).
            X_test (pd.DataFrame): Test features.
            n_original_samples (int, optional): Number of original samples.
                                                Defaults to len(X) (standard CV).

        Returns:
            tuple: (oof_probs, test_probs_avg)
                - oof_probs: OOF predictions for the original samples.
                - test_probs_avg: Averaged predictions for the test set.
        """
        # Determine the split between original and pseudo data
        if n_original_samples is None:
            n_original_samples = len(X)

        # Separate original data (for validation splitting)
        X_orig = X.iloc[:n_original_samples]
        y_orig = y.iloc[:n_original_samples]

        # Check if there are pseudo labels to augment training with
        has_pseudo = n_original_samples < len(X)
        X_pseudo = None
        y_pseudo = None

        if has_pseudo:
            X_pseudo = X.iloc[n_original_samples:]
            y_pseudo = y.iloc[n_original_samples:]
            print(
                f"Training with {len(X_pseudo)} pseudo-labeled samples added to each fold."
            )

        # Initialize storage for predictions
        # OOF predictions are only stored for original samples to calculate valid CV score
        oof_probs = np.zeros((n_original_samples, config.NUM_CLASSES))

        # Accumulator for soft voting on test set
        test_probs_sum = np.zeros((len(X_test), config.NUM_CLASSES))

        # Get Stratified Folds based on ORIGINAL data only
        # This ensures validation sets are pure and distribution matches the original task
        folds = data.get_stratified_folds(y_orig, n_folds=config.N_FOLDS)

        print(f"Starting {config.N_FOLDS}-Fold Cross-Validation...")

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"\n--- Fold {fold_idx + 1} / {config.N_FOLDS} ---")

            # Split original data into Train and Val for this fold
            X_train_fold = X_orig.iloc[train_idx]
            y_train_fold = y_orig.iloc[train_idx]
            X_val_fold = X_orig.iloc[val_idx]
            y_val_fold = y_orig.iloc[val_idx]

            # Augment the training portion with pseudo-labeled data
            if has_pseudo:
                X_train_fold = pd.concat([X_train_fold, X_pseudo], axis=0)
                y_train_fold = pd.concat([y_train_fold, y_pseudo], axis=0)

            # Initialize and Train Model
            trainer = model.XGBTrainer()
            trainer.train(
                X_train_fold,
                y_train_fold,
                X_val=X_val_fold,
                y_val=y_val_fold,
                verbose_eval=False,
            )

            # Predict on Validation Set (OOF)
            val_pred_probs = trainer.predict(X_val_fold)
            oof_probs[val_idx] = val_pred_probs

            # Calculate and print Fold Accuracy
            val_preds = np.argmax(val_pred_probs, axis=1)
            acc = accuracy_score(y_val_fold, val_preds)
            print(f"Fold {fold_idx + 1} Accuracy: {acc}")

            # Predict on Test Set
            test_pred_probs = trainer.predict(X_test)
            test_probs_sum += test_pred_probs

        # Aggregate Test Predictions (Soft Voting)
        avg_test_probs = test_probs_sum / config.N_FOLDS

        # Calculate Overall CV Score
        overall_preds = np.argmax(oof_probs, axis=1)
        overall_acc = accuracy_score(y_orig, overall_preds)
        print(f"\nOverall CV Accuracy: {overall_acc}")

        return oof_probs, avg_test_probs

    def generate_augmented_train_set(self, X_train, y_train, X_test, test_probs):
        """
        Generates an augmented training set by adding high-confidence test predictions.

        Args:
            X_train (pd.DataFrame): Original training features.
            y_train (pd.Series): Original training targets.
            X_test (pd.DataFrame): Test features.
            test_probs (np.ndarray): Predicted probabilities for test set.

        Returns:
            tuple: (X_aug, y_aug) - Augmented features and targets.
        """
        # Combine X and y into a DataFrame as required by library.data.create_augmented_dataset
        train_df = X_train.copy()
        train_df[config.TARGET_COL] = y_train

        # Use library function to create augmented dataset
        # This handles thresholding and concatenation
        augmented_df = data.create_augmented_dataset(
            train_df, X_test, test_probs, threshold=config.PSEUDO_LABEL_THRESHOLD
        )

        # Split back into X and y
        X_aug, y_aug = data.get_X_y(augmented_df)

        return X_aug, y_aug

    def save_submission(self, test_ids, test_probs, output_path=config.SUBMISSION_PATH):
        """
        Generates and saves the submission file in the correct format.

        Args:
            test_ids (pd.Series or array): IDs for the test set.
            test_probs (np.ndarray): Predicted probabilities (or class indices).
            output_path (str): Path to save the CSV.
        """
        # Get class predictions (argmax of probabilities)
        predictions = np.argmax(test_probs, axis=1)

        # Map back to original class labels (e.g., 0 -> 1, 1 -> 2)
        # The model predicts 0..N-1, but submission requires original IDs
        final_preds = [config.INVERSE_TARGET_MAPPING[p] for p in predictions]

        submission = pd.DataFrame(
            {config.ID_COL: test_ids, config.TARGET_COL: final_preds}
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
