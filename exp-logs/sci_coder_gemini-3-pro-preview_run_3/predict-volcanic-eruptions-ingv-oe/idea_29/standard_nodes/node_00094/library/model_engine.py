import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.data_manager import DataManager


class ModelEngine:
    """
    Implements the Homogeneous Bagging training strategy using High-Capacity LightGBM.
    Orchestrates 5-Fold Stratified Cross-Validation and Ensemble Inference.
    """

    def train_single_model(self, X_train, y_train, X_val, y_val, fold_id):
        """
        Trains a single LightGBM model with Early Stopping.

        Args:
            X_train, y_train: Training features and target.
            X_val, y_val: Validation features and target.
            fold_id: Integer identifier for the current fold.

        Returns:
            model: The trained LightGBM Booster.
        """
        # Create LightGBM datasets
        train_ds = lgb.Dataset(X_train, label=y_train)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

        # Train with parameters from Config
        # verbose_eval is handled via callbacks
        model = lgb.train(
            params=Config.LGBM_PARAMS,
            train_set=train_ds,
            valid_sets=[train_ds, val_ds],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(period=Config.VERBOSE_EVAL),
            ],
        )

        # Save the model to cache for later inference
        model_filename = f"lgbm_model_fold_{fold_id}.txt"
        model_path = os.path.join(Config.CACHE_DIR, model_filename)
        model.save_model(model_path)
        print(f"Model for fold {fold_id} saved to {model_path}")

        return model

    def train_kfold_ensemble(self, size=None, load_cached_data=True):
        """
        Executes Stratified K-Fold Cross-Validation.
        Combines Train and Val sets from metadata to perform 5-fold CV on the full dataset.

        Args:
            size (int, optional): Limit dataset size for debugging.
            load_cached_data (bool): Whether to use cached feature files.

        Returns:
            list: List of MAE scores for each fold.
        """
        print("Initializing K-Fold Ensemble Training...")

        # Set seed for reproducibility
        np.random.seed(Config.SEED)

        # 1. Load Data
        # We load both train and val splits defined in metadata to combine them for CV
        print("Loading training partition...")
        X_train_part, y_train_part = DataManager.get_train_data(
            size=size, load_cached_data=load_cached_data
        )

        if size is None:
            print("Loading validation partition for full CV...")
            X_val_part, y_val_part = DataManager.get_val_data(
                size=size, load_cached_data=load_cached_data
            )

            # Concatenate to form the full dataset
            X = pd.concat([X_train_part, X_val_part], axis=0).reset_index(drop=True)
            y = pd.concat([y_train_part, y_val_part], axis=0).reset_index(drop=True)
        else:
            # Debug mode: use only the loaded training subset
            print("Debug mode: Using training partition only.")
            X = X_train_part
            y = y_train_part

        print(f"Total samples for Cross-Validation: {len(X)}")

        # 2. Prepare Stratification
        # Bin the continuous target to allow StratifiedKFold
        # We target ~20 bins to ensure good distribution, but handle small debug sizes
        num_bins = 20
        if len(y) < num_bins * 5:
            num_bins = max(2, len(y) // 5)

        y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

        # 3. K-Fold Loop
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        oof_preds = np.zeros(len(X))
        fold_maes = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
            print(f"\n{'='*20} FOLD {fold} {'='*20}")

            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

            # Train
            model = self.train_single_model(X_tr, y_tr, X_va, y_va, fold)

            # Predict on validation fold
            val_preds = model.predict(X_va, num_iteration=model.best_iteration)
            oof_preds[val_idx] = val_preds

            # Metric
            fold_mae = np.mean(np.abs(y_va - val_preds))
            fold_maes.append(fold_mae)
            print(f"Fold {fold} MAE: {fold_mae}")

        # 4. Overall Metric
        total_mae = np.mean(np.abs(y - oof_preds))
        print(f"\nOverall CV MAE: {total_mae}")

        return fold_maes

    def predict_ensemble(self, size=None, load_cached_data=True):
        """
        Generates predictions for the test set using the ensemble of trained models.
        Averages predictions from all available fold models (Bagging).

        Args:
            size (int, optional): Limit dataset size for debugging.
            load_cached_data (bool): Whether to use cached feature files.
        """
        print("\nGenerating Submission...")

        # 1. Load Test Data
        X_test, ids = DataManager.get_test_data(
            size=size, load_cached_data=load_cached_data
        )

        # 2. Inference
        fold_predictions = []

        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(Config.CACHE_DIR, f"lgbm_model_fold_{fold}.txt")

            if not os.path.exists(model_path):
                print(
                    f"Warning: Model file {model_path} not found. Skipping fold {fold}."
                )
                continue

            # Load model
            print(f"Predicting with model fold {fold}...")
            model = lgb.Booster(model_file=model_path)

            # Predict
            preds = model.predict(X_test, num_iteration=model.best_iteration)
            fold_predictions.append(preds)

        if not fold_predictions:
            raise RuntimeError(
                "No models found to generate predictions. Run training first."
            )

        # 3. Average Predictions (Bagging)
        # Shape: (n_samples, n_folds) -> (n_samples,)
        avg_preds = np.mean(fold_predictions, axis=0)

        # 4. Create Submission File
        submission_df = pd.DataFrame({"segment_id": ids, "time_to_eruption": avg_preds})

        # Ensure submission directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(submission_df.head())
