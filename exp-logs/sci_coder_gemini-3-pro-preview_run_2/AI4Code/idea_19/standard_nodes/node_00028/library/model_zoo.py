import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from library.config import Config
from library.feature_engineering import GlobalVectorizer


class Stage1Ridge:
    """
    Stage 1: Sparse Lexical Regressor (The "Signpost" Model).
    Uses Ridge Regression on TF-IDF features to predict normalized rank.
    Generates Out-Of-Fold (OOF) predictions for Stage 2 training.
    """

    def __init__(self):
        self.vectorizer = GlobalVectorizer()
        self.model_path = Config.PATH_RIDGE_MODEL
        self.oof_path = Config.CACHE_STAGE1_OOF_PREDS

    def fit_oof(self, df_train, load_cached_data=True):
        """
        Generates OOF predictions using 5-Fold Group CV.
        Fits the final model on the entire df_train.

        Args:
            df_train (pd.DataFrame): Training data containing 'source', 'cell_type', 'ancestor_id', 'pct_rank'.
            load_cached_data (bool): If True, attempts to load OOF preds from cache.

        Returns:
            pd.DataFrame: DataFrame containing ['id', 'cell_id', 'pred_ridge'] for markdown cells.
        """
        # Filter for markdown cells only
        md_mask = df_train["cell_type"] == "markdown"
        df_md = df_train[md_mask].reset_index(drop=True)

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(self.oof_path):
            # We also check if the final model exists, as they are generated together
            if os.path.exists(self.model_path):
                print(f"Loading Stage 1 OOF predictions from cache: {self.oof_path}")
                return pd.read_parquet(self.oof_path)

        print("Generating Stage 1 OOF predictions...")

        # 2. Fit Vectorizer on the markdown corpus
        # Note: GlobalVectorizer handles persistence internally
        self.vectorizer.fit(df_md["source"])

        # 3. Transform text to Sparse TF-IDF
        # vectorizer.transform returns (sparse, dense). We use sparse for Ridge.
        X_sparse, _ = self.vectorizer.transform(df_md["source"])
        y = df_md["pct_rank"].values
        groups = df_md["ancestor_id"].values

        # 4. Cross-Validation
        kfold = GroupKFold(n_splits=5)
        oof_preds = np.zeros(len(df_md))

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_sparse, y, groups)):
            X_tr, y_tr = X_sparse[train_idx], y[train_idx]
            X_val = X_sparse[val_idx]

            model = Ridge(
                alpha=Config.RIDGE_ALPHA, random_state=Config.RIDGE_RANDOM_STATE
            )
            model.fit(X_tr, y_tr)
            oof_preds[val_idx] = model.predict(X_val)

        # 5. Save OOF predictions
        df_res = df_md[["id", "cell_id"]].copy()
        df_res["pred_ridge"] = oof_preds

        os.makedirs(os.path.dirname(self.oof_path), exist_ok=True)
        df_res.to_parquet(self.oof_path, index=False)

        # 6. Fit Final Model on all data
        print("Fitting final Stage 1 Ridge model on full training data...")
        final_model = Ridge(
            alpha=Config.RIDGE_ALPHA, random_state=Config.RIDGE_RANDOM_STATE
        )
        final_model.fit(X_sparse, y)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(final_model, self.model_path)

        return df_res

    def predict(self, df_test):
        """
        Predicts normalized rank for test data using the final trained model.

        Args:
            df_test (pd.DataFrame): Test data.

        Returns:
            pd.DataFrame: DataFrame containing ['id', 'cell_id', 'pred_ridge'].
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Stage 1 model not found. Run fit_oof first.")

        model = joblib.load(self.model_path)

        md_mask = df_test["cell_type"] == "markdown"
        df_md = df_test[md_mask].reset_index(drop=True)

        if df_md.empty:
            return pd.DataFrame(columns=["id", "cell_id", "pred_ridge"])

        # Transform
        X_sparse, _ = self.vectorizer.transform(df_md["source"])

        # Predict
        preds = model.predict(X_sparse)

        df_res = df_md[["id", "cell_id"]].copy()
        df_res["pred_ridge"] = preds

        return df_res


class Stage2LGBM:
    """
    Stage 2: Multi-View Instance Gradient Booster (The "Refinement" Model).
    Uses LightGBM to refine predictions using Ridge OOF, Multi-View Instance features,
    and smoothed neighborhood aggregates.
    """

    def __init__(self):
        self.model_path = Config.PATH_LGBM_MODEL

    def _prepare_data(self, df, ridge_preds, multi_view_feats):
        """
        Merges base dataframe, Ridge predictions, and Multi-View features.
        Prepares X (features) and y (target).
        """
        # Filter for markdown
        df_md = df[df["cell_type"] == "markdown"].reset_index(drop=True)

        # Merge Ridge Predictions
        # ridge_preds: ['id', 'cell_id', 'pred_ridge']
        # We cast 'id' to string to ensure consistent merging if one is category
        df_md["id_str"] = df_md["id"].astype(str)
        ridge_preds_temp = ridge_preds.copy()
        ridge_preds_temp["id_str"] = ridge_preds_temp["id"].astype(str)

        df_merged = df_md.merge(
            ridge_preds_temp[["id_str", "cell_id", "pred_ridge"]],
            on=["id_str", "cell_id"],
            how="left",
        )

        # Merge Multi-View Features
        # multi_view_feats: ['id', 'cell_id', ... features ...]
        feats_temp = multi_view_feats.copy()
        feats_temp["id_str"] = feats_temp["id"].astype(str)

        df_merged = df_merged.merge(
            feats_temp.drop(columns=["id"]),  # Drop original id to avoid conflict
            on=["id_str", "cell_id"],
            how="left",
        )

        # Define columns to exclude from features
        exclude_cols = {
            "id",
            "id_str",
            "cell_id",
            "cell_type",
            "source",
            "ancestor_id",
            "rank",
            "pct_rank",
        }

        feature_cols = [c for c in df_merged.columns if c not in exclude_cols]

        X = df_merged[feature_cols]
        y = df_merged["pct_rank"] if "pct_rank" in df_merged.columns else None

        return X, y, feature_cols

    def train(self, df_train, ridge_train, feats_train, df_val, ridge_val, feats_val):
        """
        Trains the LightGBM model with Early Stopping.

        Args:
            df_train, df_val: Base cell-level dataframes.
            ridge_train, ridge_val: Stage 1 prediction dataframes.
            feats_train, feats_val: Multi-View feature dataframes.
        """
        print("Preparing Stage 2 datasets for LightGBM...")
        X_train, y_train, feat_names = self._prepare_data(
            df_train, ridge_train, feats_train
        )
        X_val, y_val, _ = self._prepare_data(df_val, ridge_val, feats_val)

        # Create LightGBM Datasets
        train_set = lgb.Dataset(X_train, y_train, feature_name=list(feat_names))
        val_set = lgb.Dataset(
            X_val, y_val, feature_name=list(feat_names), reference=train_set
        )

        print("Training LightGBM...")
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=True
            ),
            lgb.log_evaluation(period=Config.VERBOSE_EVAL),
        ]

        model = lgb.train(
            params=Config.LGBM_PARAMS,
            train_set=train_set,
            num_boost_round=Config.NUM_BOOST_ROUND,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        model.save_model(self.model_path)

        # Evaluate
        val_preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, val_preds)
        print(f"Stage 2 Validation MAE: {mae}")

    def predict(self, df_test, ridge_test, feats_test):
        """
        Generates final rank predictions for test data.

        Returns:
            pd.DataFrame: DataFrame containing ['id', 'cell_id', 'pred_rank'].
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Stage 2 model not found. Run train() first.")

        model = lgb.Booster(model_file=self.model_path)

        X_test, _, _ = self._prepare_data(df_test, ridge_test, feats_test)

        preds = model.predict(X_test)

        # Construct result dataframe
        df_res = df_test[df_test["cell_type"] == "markdown"][["id", "cell_id"]].copy()
        df_res["pred_rank"] = preds

        return df_res
