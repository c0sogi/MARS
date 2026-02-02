import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import log_message, calculate_mae


class EnsembleTrainer:
    """
    Manages the training and inference of a Homogeneous Ensemble of LightGBM Regressors.
    Implements Stratified K-Fold CV and model persistence.
    """

    def __init__(self):
        """
        Initialize the trainer. Sets up the directory for saving models.
        """
        self.models = []
        self.model_dir = os.path.join(Config.WORKING_DIR, "models")
        os.makedirs(self.model_dir, exist_ok=True)

    def train_ensemble(self, X, y):
        """
        Trains the ensemble using Stratified K-Fold Cross-Validation.
        Saves trained models to disk.

        Args:
            X (pd.DataFrame): Feature matrix. Can contain 'segment_id'.
            y (pd.Series): Target variable 'time_to_eruption'.

        Returns:
            np.ndarray: Out-of-Fold (OOF) predictions for the training set.
        """
        # Reset models list
        self.models = []

        # Prepare data for stratification
        # Bin the continuous target to allow for stratified splitting
        # Using 10 bins as a heuristic for stratification
        num_bins = 10
        y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

        # Initialize Stratified K-Fold
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        oof_preds = np.zeros(len(y))

        # Drop non-feature columns
        features = [col for col in X.columns if col != "segment_id"]
        X_numeric = X[features]

        log_message(f"Starting {Config.N_FOLDS}-Fold Stratified Cross-Validation...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_numeric, y_bins)):
            log_message(f"\n--- Fold {fold + 1} ---")

            # Split data
            X_train, y_train = X_numeric.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X_numeric.iloc[val_idx], y.iloc[val_idx]

            # Create LightGBM Datasets
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            # Train Model
            model = lgb.train(
                Config.LGBM_PARAMS,
                dtrain,
                valid_sets=[dtrain, dval],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
                    lgb.log_evaluation(Config.VERBOSE_EVAL),
                ],
            )

            # Save Model
            model_path = os.path.join(self.model_dir, f"lgbm_fold_{fold}.txt")
            model.save_model(model_path)
            self.models.append(model)
            log_message(f"Model saved to {model_path}")

            # Generate Validation Predictions
            val_preds = model.predict(X_val, num_iteration=model.best_iteration)
            oof_preds[val_idx] = val_preds

            # Calculate Fold Score
            fold_mae = calculate_mae(y_val, val_preds)
            log_message(f"Fold {fold + 1} MAE: {fold_mae}")

        # Calculate and Log Overall Score
        total_mae = calculate_mae(y, oof_preds)
        log_message(f"\nOverall CV MAE: {total_mae}")

        return oof_preds

    def load_models(self):
        """
        Loads trained models from the model directory.
        """
        self.models = []
        log_message(f"Loading models from {self.model_dir}...")

        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(self.model_dir, f"lgbm_fold_{fold}.txt")
            if os.path.exists(model_path):
                model = lgb.Booster(model_file=model_path)
                self.models.append(model)
            else:
                log_message(
                    f"Warning: Model file not found for fold {fold} at {model_path}"
                )

        if not self.models:
            raise FileNotFoundError("No models found. Please train the ensemble first.")

    def predict_ensemble(self, X):
        """
        Generates predictions using the trained ensemble.
        Averages predictions from all loaded models.

        Args:
            X (pd.DataFrame): Feature matrix.

        Returns:
            np.ndarray: Averaged predictions.
        """
        # Ensure models are loaded
        if not self.models:
            self.load_models()

        # Drop non-feature columns
        features = [col for col in X.columns if col != "segment_id"]
        X_numeric = X[features]

        # Initialize predictions
        avg_preds = np.zeros(len(X))

        log_message(f"Generating predictions using {len(self.models)} models...")

        for i, model in enumerate(self.models):
            preds = model.predict(X_numeric, num_iteration=model.best_iteration)
            avg_preds += preds

        # Average results
        avg_preds /= len(self.models)

        return avg_preds
