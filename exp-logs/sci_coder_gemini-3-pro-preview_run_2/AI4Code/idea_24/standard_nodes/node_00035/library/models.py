import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from library.config import (
    CACHE_DIR,
    SEED,
    RIDGE_ALPHA,
    NUM_FOLDS,
    LGBM_PARAMS,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
)


class StackedHybridRanker:
    """
    Implements the Two-Stage Stacked Regression Pipeline.
    Stage 1: Ridge Regression (Sparse Lexical) - The 'Signpost' Model
    Stage 2: LightGBM (Gated Multi-Resolution) - The 'Refinement' Model
    """

    def __init__(self):
        self.ridge_model = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)
        self.lgbm_model = None

    def _get_model_path(self, model_name: str) -> str:
        """Helper to construct model file paths."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        return os.path.join(CACHE_DIR, f"{model_name}_model.joblib")

    def train_stage1_ridge(
        self,
        df: pd.DataFrame,
        pipeline,
        load_cached_data: bool = True,
    ) -> pd.DataFrame:
        """
        Trains the Stage 1 Ridge Regressor using K-Fold CV to generate OOF predictions.
        Also fits the final Ridge model on the entire provided dataset.

        Args:
            df: DataFrame containing training data (must have 'cell_type', 'rank', 'total_cells', 'source').
            pipeline: Fitted VectorizationPipeline.
            load_cached_data: Whether to load OOF predictions and model from cache.

        Returns:
            pd.DataFrame: DataFrame containing ['cell_id', 'ridge_pred'] for the input df.
        """
        oof_path = os.path.join(CACHE_DIR, "stage1_oof_preds.parquet")
        model_path = self._get_model_path("stage1_ridge")

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(oof_path) and os.path.exists(model_path):
            print("Loading Stage 1 OOF predictions and model from cache...")
            self.ridge_model = joblib.load(model_path)
            return pd.read_parquet(oof_path)

        print("Training Stage 1 Ridge and generating OOF predictions...")

        # Filter for markdown cells (we only predict rank for markdown)
        md_mask = df["cell_type"] == "markdown"
        df_md = df[md_mask].copy().reset_index(drop=True)

        if len(df_md) == 0:
            raise ValueError("No markdown cells found for Stage 1 training.")

        # Prepare Target: Normalized Rank [0, 1]
        # Formula: rank / (total_cells - 1)
        # We ensure denominator is at least 1 to avoid division by zero
        denominators = df_md["total_cells"].values - 1
        denominators = np.maximum(denominators, 1)
        y = df_md["rank"].values / denominators

        # Prepare Features: Lexical View (TF-IDF)
        # pipeline.transform returns (tfidf_matrix, svd_matrix). Ridge uses TF-IDF.
        print("Transforming text data for Ridge...")
        X, _ = pipeline.transform(df_md["source"].astype(str).fillna(""))

        # Initialize OOF array
        oof_preds = np.zeros(len(df_md))

        # K-Fold Cross-Validation for OOF generation
        kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

        print(f"Running {NUM_FOLDS}-Fold CV...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            model = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)
            model.fit(X_train, y_train)
            oof_preds[val_idx] = model.predict(X_val)

        # Create OOF DataFrame
        df_oof = pd.DataFrame({"cell_id": df_md["cell_id"], "ridge_pred": oof_preds})

        # Fit Final Model on Full Dataset
        print("Fitting final Ridge model on full dataset...")
        self.ridge_model.fit(X, y)

        # Save to Cache
        print(f"Saving Stage 1 artifacts to {CACHE_DIR}")
        df_oof.to_parquet(oof_path, index=False)
        joblib.dump(self.ridge_model, model_path)

        return df_oof

    def predict_stage1_ridge(
        self,
        df: pd.DataFrame,
        pipeline,
    ) -> pd.DataFrame:
        """
        Generates Ridge predictions for a given dataframe (e.g., validation or test set).
        Uses the pre-trained self.ridge_model.

        Args:
            df: DataFrame containing data.
            pipeline: Fitted VectorizationPipeline.

        Returns:
            pd.DataFrame: DataFrame containing ['cell_id', 'ridge_pred'].
        """
        if self.ridge_model is None:
            # Try loading
            model_path = self._get_model_path("stage1_ridge")
            if os.path.exists(model_path):
                self.ridge_model = joblib.load(model_path)
            else:
                raise ValueError("Ridge model not trained or found in cache.")

        # Filter for markdown cells
        md_mask = df["cell_type"] == "markdown"
        df_md = df[md_mask].copy().reset_index(drop=True)

        if len(df_md) == 0:
            return pd.DataFrame(columns=["cell_id", "ridge_pred"])

        # Transform
        X, _ = pipeline.transform(df_md["source"].astype(str).fillna(""))

        # Predict
        preds = self.ridge_model.predict(X)

        return pd.DataFrame({"cell_id": df_md["cell_id"], "ridge_pred": preds})

    def train_stage2_lgbm(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        load_cached_data: bool = True,
    ):
        """
        Trains the Stage 2 LightGBM Regressor.

        Args:
            train_df: Stage 2 features for training (includes target 'rank' and metadata).
            val_df: Stage 2 features for validation.
            load_cached_data: Whether to load the model from cache.
        """
        model_path = self._get_model_path("stage2_lgbm")

        if load_cached_data and os.path.exists(model_path):
            print("Loading Stage 2 LightGBM model from cache...")
            self.lgbm_model = joblib.load(model_path)
            return

        print("Training Stage 2 LightGBM...")

        # Define Features and Target
        # Exclude identifiers, raw text, and ground truth columns from features
        exclude_cols = [
            "cell_id",
            "notebook_id",
            "rank",
            "source",
            "cell_type",
            "ancestor_id",
            "filepath",
            "cell_order",
            "parent_id",
        ]
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]

        print(f"Training with {len(feature_cols)} features.")

        # Helper to compute normalized rank target
        def get_target(df):
            denoms = df["total_cells"].values - 1
            denoms = np.maximum(denoms, 1)
            return df["rank"].values / denoms

        X_train = train_df[feature_cols]
        y_train = get_target(train_df)

        X_val = val_df[feature_cols]
        y_val = get_target(val_df)

        # Create LGBM Datasets
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # Setup Parameters
        params = LGBM_PARAMS.copy()
        num_boost_round = params.pop("n_estimators", 5000)

        # Train with Early Stopping
        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=VERBOSE_EVAL),
        ]

        self.lgbm_model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Save
        print(f"Saving Stage 2 model to {model_path}")
        joblib.dump(self.lgbm_model, model_path)

    def predict_stage2_lgbm(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates final rank predictions using the Stage 2 model.

        Args:
            test_df: Stage 2 features for test set.

        Returns:
            pd.DataFrame: DataFrame containing ['cell_id', 'pred_rank'].
        """
        if self.lgbm_model is None:
            model_path = self._get_model_path("stage2_lgbm")
            if os.path.exists(model_path):
                self.lgbm_model = joblib.load(model_path)
            else:
                raise ValueError("LightGBM model not trained or found in cache.")

        exclude_cols = [
            "cell_id",
            "notebook_id",
            "rank",
            "source",
            "cell_type",
            "ancestor_id",
            "filepath",
            "cell_order",
            "parent_id",
        ]
        feature_cols = [c for c in test_df.columns if c not in exclude_cols]

        X_test = test_df[feature_cols]
        preds = self.lgbm_model.predict(X_test)

        return pd.DataFrame({"cell_id": test_df["cell_id"], "pred_rank": preds})
