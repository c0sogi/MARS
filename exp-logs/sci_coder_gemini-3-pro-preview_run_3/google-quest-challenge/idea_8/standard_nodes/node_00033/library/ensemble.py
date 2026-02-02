import os
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor

from library.config import Config
from library.utils import seed_everything, compute_spearman_metric


class MetaStacker:
    """
    Level-2 Stacking Ensemble.
    Aggregates OOF and Test predictions from multiple upstream models (streams),
    trains a Ridge Regression meta-learner, and generates the final submission.
    """

    def __init__(self, model_tags):
        """
        Args:
            model_tags (list of str): List of model identifiers (e.g., ['deberta', 'mpnet'])
                                      corresponding to the file prefixes in the working directory.
        """
        self.model_tags = model_tags
        self.n_folds = Config.N_FOLDS
        self.working_dir = Config.WORKING_DIR
        self.target_cols = Config.TARGET_COLS

        # Define Meta-Learner
        # We use RidgeCV to automatically select the best regularization strength
        self.meta_model = MultiOutputRegressor(
            RidgeCV(alphas=Config.RIDGE_ALPHAS, scoring="neg_mean_squared_error")
        )

    def _load_oof_data(self):
        """
        Loads and aggregates OOF predictions and targets from all folds.

        Returns:
            X_meta (np.ndarray): Stacked OOF predictions (Features for meta-learner).
            y_meta (np.ndarray): Stacked Ground Truth targets.
        """
        X_list = []
        y_list = []

        print(f"[MetaStacker] Loading OOF data for tags: {self.model_tags}")

        for fold in range(self.n_folds):
            # 1. Load Targets
            # Targets are consistent across models, so we just load from the first tag
            target_path = os.path.join(
                self.working_dir, f"{self.model_tags[0]}_fold{fold}_val_targets.npy"
            )
            if not os.path.exists(target_path):
                raise FileNotFoundError(f"Missing target file: {target_path}")

            y_fold = np.load(target_path)
            y_list.append(y_fold)

            # 2. Load and Concat Predictions from each model
            fold_preds = []
            for tag in self.model_tags:
                pred_path = os.path.join(
                    self.working_dir, f"{tag}_fold{fold}_oof_preds.npy"
                )
                if not os.path.exists(pred_path):
                    raise FileNotFoundError(f"Missing OOF preds: {pred_path}")

                p = np.load(pred_path)
                fold_preds.append(p)

            # Horizontal Stack: [Preds_ModelA, Preds_ModelB, ...]
            # Shape: (N_Samples_Fold, 30 * N_Models)
            X_fold = np.concatenate(fold_preds, axis=1)
            X_list.append(X_fold)

        # Vertical Stack across folds
        X_meta = np.concatenate(X_list, axis=0)
        y_meta = np.concatenate(y_list, axis=0)

        return X_meta, y_meta

    def _load_test_data(self):
        """
        Loads and aggregates Test predictions.
        Averages test predictions across folds for each model, then concatenates models.

        Returns:
            X_test (np.ndarray): Stacked Test predictions.
            test_ids (np.ndarray): QA IDs for the test set.
        """
        print(f"[MetaStacker] Loading Test data for tags: {self.model_tags}")

        model_avg_preds = []
        test_ids = None

        for tag in self.model_tags:
            fold_test_preds = []

            for fold in range(self.n_folds):
                path = os.path.join(
                    self.working_dir, f"{tag}_fold{fold}_test_preds.npy"
                )
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing Test preds: {path}")

                fold_test_preds.append(np.load(path))

                # Load IDs from the first fold of the first model (static)
                if test_ids is None:
                    id_path = os.path.join(
                        self.working_dir, f"{tag}_fold{fold}_test_ids.npy"
                    )
                    if os.path.exists(id_path):
                        test_ids = np.load(id_path)

            # Average predictions across folds for this model
            # Shape: (N_Test, 30)
            avg_pred = np.mean(fold_test_preds, axis=0)
            model_avg_preds.append(avg_pred)

        # Horizontal Stack: [Avg_Preds_ModelA, Avg_Preds_ModelB, ...]
        # Shape: (N_Test, 30 * N_Models)
        X_test = np.concatenate(model_avg_preds, axis=1)

        if test_ids is None:
            raise FileNotFoundError(
                "Could not locate test_ids.npy in working directory."
            )

        return X_test, test_ids

    def run(self):
        """
        Executes the stacking pipeline:
        1. Load Data
        2. Train Meta-Learner
        3. Evaluate OOF Performance
        4. Generate Submission
        """
        seed_everything(Config.SEED)

        # 1. Load Data
        X_meta, y_meta = self._load_oof_data()
        X_test, test_ids = self._load_test_data()

        print(
            f"[MetaStacker] Meta-Train Shape: {X_meta.shape}, Targets: {y_meta.shape}"
        )
        print(f"[MetaStacker] Meta-Test Shape:  {X_test.shape}")

        # 2. Train Meta-Learner
        print("[MetaStacker] Training RidgeCV Meta-Learner...")
        self.meta_model.fit(X_meta, y_meta)

        # 3. Evaluate OOF
        oof_preds = self.meta_model.predict(X_meta)
        oof_preds = np.clip(oof_preds, 0, 1)

        score = compute_spearman_metric(y_meta, oof_preds)
        print(f"[MetaStacker] Final Ensemble OOF Spearman Correlation: {score:.16f}")

        # 4. Generate Test Predictions
        print("[MetaStacker] Generating Final Test Predictions...")
        final_preds = self.meta_model.predict(X_test)
        final_preds = np.clip(final_preds, 0, 1)

        # 5. Save Submission
        self._save_submission(final_preds, test_ids)

    def _save_submission(self, preds, ids):
        """
        Saves the predictions to a CSV file in the required format.
        """
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        df = pd.DataFrame(preds, columns=self.target_cols)
        df.insert(0, "qa_id", ids)

        df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"[MetaStacker] Submission saved to {Config.SUBMISSION_PATH}")
        print(f"[MetaStacker] Head:\n{df.head()}")
