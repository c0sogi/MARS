import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

from library.config import Config
from library.utils import seed_everything, get_score
from library.feature_engineering import FeatureExtractor


class TabularTrainer:
    """
    Manages the training, validation, and inference lifecycle for the Tabular Branch (LightGBM).
    Handles Feature Engineering, 5-Fold CV, OOF generation, and submission creation.
    """

    def __init__(self):
        self.config = Config
        seed_everything(self.config.SEED)

        # Create necessary directories
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)
        os.makedirs(self.config.SUBMISSION_DIR, exist_ok=True)

    def _get_features(self, debug_sample=None, load_cached_data=True):
        """
        Orchestrates feature extraction for Train, Val, and Test sets.
        Uses FeatureExtractor which handles caching logic.
        """
        extractor = FeatureExtractor()

        # 1. Train Features
        print("Processing Train Features...")
        df_train = extractor.process_dataset(
            self.config.TRAIN_METADATA_PATH,
            self.config.TRAIN_FEATURES_PATH,
            load_cached_data=load_cached_data,
            debug_sample=debug_sample,
        )

        # 2. Val Features
        print("Processing Val Features...")
        df_val = extractor.process_dataset(
            self.config.VAL_METADATA_PATH,
            self.config.VAL_FEATURES_PATH,
            load_cached_data=load_cached_data,
            debug_sample=debug_sample,
        )

        # 3. Test Features
        print("Processing Test Features...")
        df_test = extractor.process_dataset(
            self.config.TEST_METADATA_PATH,
            self.config.TEST_FEATURES_PATH,
            load_cached_data=load_cached_data,
            debug_sample=debug_sample,
        )

        return df_train, df_val, df_test

    def run_cv(self, debug_sample=None, load_cached_data=True):
        """
        Executes the 5-Fold Cross-Validation loop for LightGBM.

        Args:
            debug_sample (int, optional): If set, limits the dataset size for debugging.
            load_cached_data (bool): If True, attempts to load features from cache.
        """
        # 1. Load Features
        df_train_part, df_val_part, df_test = self._get_features(
            debug_sample=debug_sample, load_cached_data=load_cached_data
        )

        # Combine Train and Val to form the full development set for CV
        # We ignore the original split for the purpose of 5-Fold CV to maximize data usage
        df_full = pd.concat([df_train_part, df_val_part], ignore_index=True)

        if debug_sample:
            print(f"Debug Mode: Training on {len(df_full)} samples.")

        # 2. Prepare Data Matrices
        drop_cols = [self.config.SEGMENT_ID_COL, self.config.TARGET_COL]
        feature_cols = [c for c in df_full.columns if c not in drop_cols]

        X = df_full[feature_cols]
        y = df_full[self.config.TARGET_COL]
        segment_ids = df_full[self.config.SEGMENT_ID_COL].values

        X_test = df_test[feature_cols]
        test_ids = df_test[self.config.SEGMENT_ID_COL].values

        # 3. Setup KFold
        kf = KFold(
            n_splits=self.config.N_FOLDS, shuffle=True, random_state=self.config.SEED
        )

        # Storage
        oof_preds = np.zeros(len(df_full))
        test_preds_folds = []
        feature_importance = pd.DataFrame(index=feature_cols)
        feature_importance["total_gain"] = 0.0

        # 4. CV Loop
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"\n{'='*20} Tabular Fold {fold} {'='*20}")

            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            # Create LGBM Datasets
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            # Prepare Parameters
            params = self.config.LGBM_PARAMS.copy()

            # Extract training control params that shouldn't be in the params dict for lgb.train
            num_boost_round = params.pop("n_estimators", 2000)
            early_stopping_rounds = params.pop("early_stopping_rounds", 100)

            # Callbacks
            callbacks = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=100),
            ]

            # Train
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dtrain, dval],
                valid_names=["train", "valid"],
                callbacks=callbacks,
            )

            # Predict Val
            val_preds = model.predict(X_val, num_iteration=model.best_iteration)
            oof_preds[val_idx] = val_preds

            # Score
            fold_mae = get_score(y_val, val_preds)
            print(f"Fold {fold} MAE: {fold_mae}")

            # Predict Test
            test_preds = model.predict(X_test, num_iteration=model.best_iteration)
            test_preds_folds.append(test_preds)

            # Feature Importance
            fold_importance = model.feature_importance(importance_type="gain")
            feature_importance["total_gain"] += fold_importance

            # Save Model
            model_path = os.path.join(
                self.config.WORKING_DIR, f"lgb_model_fold_{fold}.txt"
            )
            model.save_model(model_path)

        # 5. Aggregate Results
        overall_mae = get_score(y, oof_preds)
        print(f"\nOverall Tabular CV MAE: {overall_mae}")

        # Save OOF
        oof_df = pd.DataFrame(
            {"segment_id": segment_ids, "time_to_eruption": oof_preds}
        )
        oof_path = os.path.join(self.config.WORKING_DIR, "tabular_oof.csv")
        oof_df.to_csv(oof_path, index=False)
        print(f"Tabular OOF predictions saved to {oof_path}")

        # Average Test Preds
        avg_test_preds = np.mean(test_preds_folds, axis=0)

        # Save Test Preds (Intermediate)
        test_pred_df = pd.DataFrame(
            {"segment_id": test_ids, "time_to_eruption": avg_test_preds}
        )
        test_pred_path = os.path.join(self.config.WORKING_DIR, "tabular_test.csv")
        test_pred_df.to_csv(test_pred_path, index=False)
        print(f"Tabular Test predictions saved to {test_pred_path}")

        # Save Final Submission (Standalone)
        sub_path = os.path.join(self.config.SUBMISSION_DIR, "submission.csv")
        test_pred_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

        # Save Feature Importance
        feature_importance["total_gain"] /= self.config.N_FOLDS
        imp_path = os.path.join(self.config.WORKING_DIR, "feature_importance.csv")
        feature_importance.sort_values(by="total_gain", ascending=False).to_csv(
            imp_path
        )
        print(f"Feature importance saved to {imp_path}")
