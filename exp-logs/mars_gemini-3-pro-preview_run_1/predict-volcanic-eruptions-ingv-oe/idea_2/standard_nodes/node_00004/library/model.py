import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from library.config import MODEL_CONFIG, RANDOM_SEED, WORKING_DIR, SUBMISSION_PATH
from library.utils import seed_everything


class CrossValidator:
    """
    Manages the 5-Fold Cross-Validation training of LightGBM models.
    """

    def __init__(self, config=None):
        """
        Args:
            config (dict, optional): Configuration dictionary for LightGBM.
                                     Defaults to MODEL_CONFIG from library.config.
        """
        self.config = config if config else MODEL_CONFIG.copy()
        self.n_folds = 5
        seed_everything(RANDOM_SEED)

    def train(self, X, y):
        """
        Trains the model using K-Fold Cross Validation.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training targets.

        Returns:
            tuple: (fold_scores, overall_mae)
        """
        # Reset indices to ensure safe splitting with KFold
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=RANDOM_SEED)

        oof_preds = np.zeros(len(X))
        scores = []

        # Prepare parameters
        params = self.config.copy()

        # Extract training control parameters that are passed as arguments to lgb.train
        # or used in callbacks, rather than in the params dict.
        n_estimators = params.pop("n_estimators", 5000)
        early_stopping_rounds = params.pop("early_stopping_rounds", 150)

        # Ensure verbosity is controlled
        if "verbosity" not in params:
            params["verbosity"] = -1

        print(f"Starting {self.n_folds}-Fold Cross-Validation...")

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            # Setup callbacks for early stopping and logging
            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds, verbose=False
                ),
                lgb.log_evaluation(period=100),
            ]

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=n_estimators,
                valid_sets=[dtrain, dval],
                valid_names=["train", "valid"],
                callbacks=callbacks,
            )

            # Save the trained model for this fold
            model_filename = f"lgb_model_fold_{fold}.txt"
            model_path = os.path.join(WORKING_DIR, model_filename)
            model.save_model(model_path)

            # Generate predictions for validation set
            # best_iteration is handled automatically by the booster after early stopping
            val_preds = model.predict(X_val, num_iteration=model.best_iteration)
            oof_preds[val_idx] = val_preds

            mae = mean_absolute_error(y_val, val_preds)
            scores.append(mae)
            print(f"Fold {fold} MAE: {mae}")

        overall_mae = mean_absolute_error(y, oof_preds)
        print(f"Overall CV MAE: {overall_mae}")

        return scores, overall_mae


class InferenceModel:
    """
    Loads trained models and generates predictions for test data.
    """

    def __init__(self, model_dir=WORKING_DIR):
        """
        Args:
            model_dir (str): Directory where trained model files are stored.
        """
        self.model_dir = model_dir
        self.models = []
        self._load_models()

    def _load_models(self):
        """
        Loads all available fold models from the working directory.
        """
        self.models = []
        fold = 0
        while True:
            model_path = os.path.join(self.model_dir, f"lgb_model_fold_{fold}.txt")
            if not os.path.exists(model_path):
                break

            try:
                model = lgb.Booster(model_file=model_path)
                self.models.append(model)
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")

            fold += 1

        if not self.models:
            print(
                f"Warning: No models found in {self.model_dir}. Ensure training is complete."
            )
        else:
            print(f"Loaded {len(self.models)} models for inference.")

    def predict(self, X):
        """
        Generates averaged predictions using all loaded models.

        Args:
            X (pd.DataFrame): Features for prediction.

        Returns:
            np.ndarray: Averaged predictions.
        """
        if not self.models:
            raise RuntimeError("No models loaded. Cannot perform inference.")

        # Initialize predictions
        avg_preds = np.zeros(len(X))

        for model in self.models:
            preds = model.predict(X, num_iteration=model.best_iteration)
            avg_preds += preds

        # Average across folds
        avg_preds /= len(self.models)

        return avg_preds

    def generate_submission(self, X_test, segment_ids):
        """
        Generates predictions for the test set and saves the submission file.

        Args:
            X_test (pd.DataFrame): Test features.
            segment_ids (pd.Series): Corresponding segment IDs.
        """
        print("Generating predictions for test set...")
        predictions = self.predict(X_test)

        submission_df = pd.DataFrame(
            {"segment_id": segment_ids, "time_to_eruption": predictions}
        )

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
