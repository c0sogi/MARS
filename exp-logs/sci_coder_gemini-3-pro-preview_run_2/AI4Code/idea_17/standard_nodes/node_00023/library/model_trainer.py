import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from library.config import Config
from library.utils import validate_ranks
from library.feature_generator import FeatureFactory


class Stage1Ridge:
    """
    Wrapper for Stage 1 Ridge Regression Model.
    """

    def __init__(self, alpha=1.0, random_state=42):
        self.model = Ridge(alpha=alpha, random_state=random_state, solver="auto")

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)


class Stage2LGBM:
    """
    Wrapper for Stage 2 LightGBM Model.
    """

    def __init__(self, params):
        self.params = params
        self.model = None

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        early_stopping_rounds=100,
        verbose_eval=100,
    ):
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=verbose_eval),
        ]

        self.model = lgb.train(
            self.params, train_data, valid_sets=valid_sets, callbacks=callbacks
        )
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def save(self, path):
        if self.model is not None:
            self.model.save_model(path)

    def load(self, path):
        self.model = lgb.Booster(model_file=path)


class CrossValidator:
    """
    Handles Group K-Fold Cross Validation for Stage 1 to generate OOF predictions.
    """

    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def get_oof_predictions(self, X, y, groups, model_class, model_params):
        """
        Generates Out-Of-Fold predictions.

        Args:
            X: Sparse feature matrix.
            y: Target array.
            groups: Array of group identifiers (notebook_ids).
            model_class: Class of the model to instantiate (e.g., Stage1Ridge).
            model_params: Dictionary of parameters for the model.

        Returns:
            oof_preds: Array of predictions aligned with input X.
        """
        unique_groups = np.unique(groups)
        oof_preds = np.zeros(len(y))

        print(f"Starting {self.n_splits}-Fold CV on {len(unique_groups)} notebooks...")

        for fold, (train_group_idx, val_group_idx) in enumerate(
            self.kf.split(unique_groups)
        ):
            # Map group indices back to sample indices
            train_groups = unique_groups[train_group_idx]
            val_groups = unique_groups[val_group_idx]

            # Create boolean masks
            # Using np.isin is efficient enough for 100k groups
            train_mask = np.isin(groups, train_groups)
            val_mask = np.isin(groups, val_groups)

            X_train_fold = X[train_mask]
            y_train_fold = y[train_mask]
            X_val_fold = X[val_mask]

            # Train
            model = model_class(**model_params)
            model.fit(X_train_fold, y_train_fold)

            # Predict
            preds = model.predict(X_val_fold)

            # Store
            oof_preds[val_mask] = preds

            # Simple metric print
            mae = np.mean(np.abs(y[val_mask] - preds))
            print(f"Fold {fold+1} MAE: {mae}")

        return oof_preds


class ModelTrainer:
    """
    Orchestrates the Two-Stage Stacked Hybrid Ranking pipeline.
    """

    def __init__(self):
        self.config = Config
        self.feature_factory = FeatureFactory()

        # Model Paths
        self.ridge_path = os.path.join(self.config.WORKING_DIR, "stage1_ridge.joblib")
        self.lgbm_path = os.path.join(self.config.WORKING_DIR, "stage2_lgbm.txt")
        self.oof_path = os.path.join(self.config.WORKING_DIR, "stage1_oof_preds.npy")

    def train(self, load_cached_data=True):
        """
        Executes the training pipeline.
        """
        print("=== Starting Training Pipeline ===")

        # ----------------------------------------------------------------------
        # Stage 1: Ridge Regression & OOF Generation
        # ----------------------------------------------------------------------
        print("\n--- Stage 1: Ridge Regression ---")
        X_train, y_train, groups_train = self.feature_factory.build_stage1_dataset(
            split="train", load_cached_data=load_cached_data
        )

        # Check if OOF preds exist
        if load_cached_data and os.path.exists(self.oof_path):
            print(f"Loading cached OOF predictions from {self.oof_path}")
            oof_preds = np.load(self.oof_path)
        else:
            print("Generating OOF predictions via Cross-Validation...")
            cv = CrossValidator(
                n_splits=self.config.NUM_FOLDS, random_state=self.config.RANDOM_STATE
            )
            oof_preds = cv.get_oof_predictions(
                X_train,
                y_train,
                groups_train,
                model_class=Stage1Ridge,
                model_params={
                    "alpha": self.config.RIDGE_ALPHA,
                    "random_state": self.config.RANDOM_STATE,
                },
            )
            np.save(self.oof_path, oof_preds)

        # Train Final Ridge Model on all data (for inference)
        ridge_model = Stage1Ridge(
            alpha=self.config.RIDGE_ALPHA, random_state=self.config.RANDOM_STATE
        )
        if load_cached_data and os.path.exists(self.ridge_path):
            print(f"Loading cached Final Ridge model from {self.ridge_path}")
            ridge_model.load(self.ridge_path)
        else:
            print("Training Final Ridge model on full training set...")
            ridge_model.fit(X_train, y_train)
            ridge_model.save(self.ridge_path)

        # ----------------------------------------------------------------------
        # Stage 2: LightGBM Stacking
        # ----------------------------------------------------------------------
        print("\n--- Stage 2: LightGBM Stacking ---")

        # 1. Prepare Training Data for Stage 2 (Using OOF preds)
        df_train_s2, y_train_s2, _ = self.feature_factory.build_stage2_dataset(
            split="train", ridge_preds=oof_preds, load_cached_data=load_cached_data
        )

        # 2. Prepare Validation Data for Stage 2
        # We need Ridge predictions for the validation set first
        print("Preparing Validation Data...")
        X_val, y_val, groups_val = self.feature_factory.build_stage1_dataset(
            split="val", load_cached_data=load_cached_data
        )

        # Predict using the Full Ridge Model
        val_ridge_preds = ridge_model.predict(X_val)

        # Build Stage 2 Val Features
        df_val_s2, y_val_s2, _ = self.feature_factory.build_stage2_dataset(
            split="val", ridge_preds=val_ridge_preds, load_cached_data=load_cached_data
        )

        # 3. Train LightGBM
        lgbm_model = Stage2LGBM(params=self.config.LGBM_PARAMS)

        if load_cached_data and os.path.exists(self.lgbm_path):
            print(f"Loading cached LightGBM model from {self.lgbm_path}")
            lgbm_model.load(self.lgbm_path)
        else:
            print("Training LightGBM model...")
            lgbm_model.fit(
                df_train_s2,
                y_train_s2,
                X_val=df_val_s2,
                y_val=y_val_s2,
                early_stopping_rounds=self.config.LGBM_EARLY_STOPPING_ROUNDS,
                verbose_eval=self.config.LGBM_VERBOSE_EVAL,
            )
            lgbm_model.save(self.lgbm_path)

        print("Training Pipeline Completed.")
        return ridge_model, lgbm_model

    def generate_submission(self, load_cached_data=True):
        """
        Generates predictions for the test set and creates the submission file.
        """
        print("\n=== Generating Submission ===")

        # 1. Load Models
        if not os.path.exists(self.ridge_path) or not os.path.exists(self.lgbm_path):
            raise FileNotFoundError("Models not found. Run train() first.")

        ridge_model = Stage1Ridge()
        ridge_model.load(self.ridge_path)

        lgbm_model = Stage2LGBM(params=self.config.LGBM_PARAMS)
        lgbm_model.load(self.lgbm_path)

        # 2. Process Test Data
        # Stage 1
        print("Processing Test Data (Stage 1)...")
        X_test, _, groups_test = self.feature_factory.build_stage1_dataset(
            split="test", load_cached_data=load_cached_data
        )

        test_ridge_preds = ridge_model.predict(X_test)

        # Stage 2
        print("Processing Test Data (Stage 2)...")
        df_test_s2, _, _ = self.feature_factory.build_stage2_dataset(
            split="test",
            ridge_preds=test_ridge_preds,
            load_cached_data=load_cached_data,
        )

        # 3. Final Prediction
        print("Predicting with LightGBM...")
        final_ranks = lgbm_model.predict(df_test_s2)
        final_ranks = validate_ranks(final_ranks)

        # 4. Reconstruct Notebook Orders
        print("Reconstructing Cell Orders...")

        # We need to map predictions back to cell IDs and sort them.
        # We'll load the test metadata to get the structure.
        # Note: FeatureFactory loads data, but we need the mapping of index -> (notebook_id, cell_id).
        # We can reload the dataframe from FeatureFactory's cache or loader.

        # Efficient way: Load the processed test dataframe from cache (created by FeatureFactory)
        # The FeatureFactory caches `test_markdown.parquet` and `test_code.parquet`.
        loader = self.feature_factory.loader
        df_md_test, df_code_test = loader.load_data(split="test", load_cached_data=True)

        # Assign predicted ranks to markdown cells
        df_md_test["pred_rank"] = final_ranks

        # Code cells have fixed relative ranks.
        # In `loader.load_data` for test, code ranks are -1.
        # But `FeatureFactory._get_data` fixes them to [0, 1].
        # We need to replicate that logic or trust that `df_code_test` has the correct structure.
        # The safest way is to re-calculate code ranks exactly as FeatureFactory did.

        def calc_rank(g):
            n = len(g)
            if n <= 1:
                return pd.Series([0.0] * n, index=g.index)
            return pd.Series(np.arange(n) / (n - 1), index=g.index)

        code_ranks = df_code_test.groupby("notebook_id", group_keys=False)[
            "cell_id"
        ].apply(calc_rank)
        df_code_test["pred_rank"] = code_ranks

        # Combine Markdown and Code
        df_combined = pd.concat(
            [
                df_md_test[["notebook_id", "cell_id", "pred_rank"]],
                df_code_test[["notebook_id", "cell_id", "pred_rank"]],
            ],
            axis=0,
        )

        # Sort
        # Sort by notebook_id, then by pred_rank
        df_sorted = df_combined.sort_values(["notebook_id", "pred_rank"])

        # Group by notebook and join cell_ids
        submission_series = df_sorted.groupby("notebook_id")["cell_id"].apply(
            lambda x: " ".join(x)
        )

        # 5. Save Submission
        submission_df = submission_series.reset_index()
        submission_df.columns = ["id", "cell_order"]

        output_path = self.config.SUBMISSION_PATH
        print(f"Saving submission to {output_path}")
        submission_df.to_csv(output_path, index=False)

        return submission_df
