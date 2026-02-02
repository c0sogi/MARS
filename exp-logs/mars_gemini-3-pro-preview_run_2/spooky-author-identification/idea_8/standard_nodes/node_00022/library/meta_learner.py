import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from library.config import Config
from library.utils import multiclass_log_loss, clip_probabilities, seed_everything
from library.data_factory import DataManager


class MetaLearner:
    """
    Implements the Stacking Ensemble strategy.
    Aggregates predictions from base models and trains a Logistic Regression meta-learner.
    """

    def __init__(self):
        seed_everything(Config.SEED)
        self.submission_dir = Config.SUBMISSION_DIR
        self.submission_path = Config.SUBMISSION_PATH
        os.makedirs(self.submission_dir, exist_ok=True)

    def _load_targets(self):
        """
        Reconstructs the target array corresponding to the concatenated OOF predictions.
        The base models concatenate Train + Val, so we must do the same.
        """
        train_df, val_df, _ = DataManager.load_metadata()

        # Concatenate in the same order as the base models (Train then Val)
        full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

        # Map labels to integers
        y_true = full_df[Config.TARGET_COL].map(Config.LABEL_MAP).values
        return y_true

    def train_meta_learner(self, oof_dict, test_dict):
        """
        Trains the meta-learner on stacked OOF predictions and generates final test predictions.

        Args:
            oof_dict (dict): Dictionary mapping model names to OOF probability arrays (N_samples, N_classes).
            test_dict (dict): Dictionary mapping model names to Test probability arrays (N_test, N_classes).

        Returns:
            final_preds (np.ndarray): The final predicted probabilities for the test set.
        """
        print("\n==== Meta-Learner Training ====")

        # 1. Prepare Data
        y_true = self._load_targets()

        # Ensure consistent ordering of models
        model_names = sorted(oof_dict.keys())
        print(f"Stacking models: {model_names}")

        # Stack features horizontally: (N_samples, N_models * N_classes)
        X_train_meta = []
        X_test_meta = []

        for name in model_names:
            X_train_meta.append(oof_dict[name])
            X_test_meta.append(test_dict[name])

        X_train_meta = np.hstack(X_train_meta)
        X_test_meta = np.hstack(X_test_meta)

        print(f"Meta-Feature Matrix Shape (Train): {X_train_meta.shape}")
        print(f"Meta-Feature Matrix Shape (Test): {X_test_meta.shape}")

        # 2. Train Logistic Regression Meta-Learner
        # We use a simple linear model to calibrate the ensemble weights
        meta_model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=Config.SEED,
            max_iter=1000,
        )

        meta_model.fit(X_train_meta, y_true)

        # 3. Evaluate on Training Data (Ensemble OOF Score)
        # Note: Strictly speaking, this is an 'insample' score for the meta-learner,
        # but since inputs are OOF, it represents the ensemble CV performance.
        ensemble_oof_probs = meta_model.predict_proba(X_train_meta)
        ensemble_score = multiclass_log_loss(y_true, ensemble_oof_probs)

        print(f"Ensemble CV LogLoss: {ensemble_score}")

        # 4. Generate Test Predictions
        final_test_probs = meta_model.predict_proba(X_test_meta)

        # Clip probabilities to safe range
        final_test_probs = clip_probabilities(final_test_probs)

        # 5. Save Submission
        self._save_submission(final_test_probs)

        return final_test_probs

    def _save_submission(self, probs):
        """
        Formats and saves the submission file.
        """
        # Load test metadata to get IDs
        _, _, test_df = DataManager.load_metadata()

        # Check shape consistency
        if len(test_df) != len(probs):
            raise ValueError(
                f"Shape mismatch: Test DF has {len(test_df)} rows, Preds have {len(probs)} rows."
            )

        # Create DataFrame
        submission_df = pd.DataFrame()
        submission_df["id"] = test_df["id"]

        # Map class indices back to column names
        # Config.LABEL_MAP is {'EAP': 0, 'HPL': 1, 'MWS': 2}
        # We need columns in specific order: EAP, HPL, MWS
        inv_map = {v: k for k, v in Config.LABEL_MAP.items()}

        for i in range(Config.NUM_CLASSES):
            col_name = inv_map[i]
            submission_df[col_name] = probs[:, i]

        # Reorder columns to ensure id, EAP, HPL, MWS
        cols = ["id", "EAP", "HPL", "MWS"]
        submission_df = submission_df[cols]

        # Save
        submission_df.to_csv(self.submission_path, index=False)
        print(f"Submission saved to {self.submission_path}")

        # Preview
        print("Submission Preview:")
        print(submission_df.head())
