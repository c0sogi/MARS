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

    def run_cv_training(self, X, y, X_test):
        """
        Runs Stratified K-Fold Cross-Validation using XGBTrainer.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training targets.
            X_test (pd.DataFrame): Test features.

        Returns:
            tuple: (oof_probs, test_probs_avg)
                - oof_probs: OOF predictions for the training samples.
                - test_probs_avg: Averaged predictions for the test set.
        """
        # Initialize storage for predictions
        oof_probs = np.zeros((len(X), config.NUM_CLASSES))

        # Accumulator for soft voting on test set
        test_probs_sum = np.zeros((len(X_test), config.NUM_CLASSES))

        # Get Stratified Folds
        folds = data.get_stratified_folds(y, n_folds=config.N_FOLDS)

        print(f"Starting {config.N_FOLDS}-Fold Cross-Validation...")

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"\n--- Fold {fold_idx + 1} / {config.N_FOLDS} ---")

            # Split data into Train and Val for this fold
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

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
        overall_acc = accuracy_score(y, overall_preds)
        print(f"\nOverall CV Accuracy: {overall_acc}")

        return oof_probs, avg_test_probs

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
