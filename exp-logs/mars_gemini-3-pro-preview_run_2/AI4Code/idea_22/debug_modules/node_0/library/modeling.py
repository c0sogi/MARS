import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from library.config import Config
from library.utils import save_artifacts


class Stage1Ridge:
    """
    Stage 1: Sparse Lexical Regressor using Ridge Regression.
    Generates OOF predictions for stacking and fits a final model for inference.
    """

    def __init__(self):
        self.model_params = Config.RIDGE_PARAMS
        self.model_path = Config.RIDGE_MODEL_PATH
        self.oof_path = Config.STAGE1_OOF_PATH
        self.n_folds = 5

    def get_oof_predictions(self, df, text_pipeline, load_cached_data=True):
        """
        Generates Out-Of-Fold predictions for the training set.
        Also fits the final model on the entire dataset.
        """
        # 1. Check Cache
        if load_cached_data and os.path.exists(self.oof_path):
            print(f"Loading cached Stage 1 OOF predictions from {self.oof_path}")
            return pd.read_parquet(self.oof_path)

        print("Generating Stage 1 OOF predictions...")

        # Filter for markdown cells only
        mask_md = df["cell_type"] == "markdown"
        df_md = df[mask_md].copy().reset_index(drop=True)

        # Vectorize
        print("Vectorizing text for Stage 1...")
        X = text_pipeline.transform_tfidf(df_md["source"])
        y = df_md["pct_rank"].values
        groups = df_md["ancestor_id"].values

        # Initialize OOF array
        oof_preds = np.zeros(len(df_md))

        # GroupKFold
        gkf = GroupKFold(n_splits=self.n_folds)

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val = X[val_idx]

            model = Ridge(**self.model_params)
            model.fit(X_train, y_train)
            oof_preds[val_idx] = model.predict(X_val)

        # Create result DataFrame
        df_oof = df_md[["id", "cell_id"]].copy()
        df_oof["stage1_pred"] = oof_preds

        # Save OOF cache
        print(f"Saving Stage 1 OOF predictions to {self.oof_path}")
        df_oof.to_parquet(self.oof_path, index=False)

        # Fit Final Model on All Data
        print("Fitting final Stage 1 model on full dataset...")
        final_model = Ridge(**self.model_params)
        final_model.fit(X, y)

        print(f"Saving final Stage 1 model to {self.model_path}")
        save_artifacts(final_model, self.model_path)

        return df_oof

    def predict(self, df, text_pipeline):
        """
        Generates predictions using the fitted Stage 1 model.
        Used for Validation and Test sets.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Stage 1 model not found at {self.model_path}. Run get_oof_predictions first."
            )

        # Load model
        model = joblib.load(self.model_path)

        # Filter markdown
        mask_md = df["cell_type"] == "markdown"
        df_md = df[mask_md].copy().reset_index(drop=True)

        if len(df_md) == 0:
            return pd.DataFrame(columns=["id", "cell_id", "stage1_pred"])

        # Vectorize
        X = text_pipeline.transform_tfidf(df_md["source"])

        # Predict
        preds = model.predict(X)

        # Result DataFrame
        df_res = df_md[["id", "cell_id"]].copy()
        df_res["stage1_pred"] = preds

        return df_res


class Stage2LGBM:
    """
    Stage 2: Content-Aware Gradient Booster using LightGBM.
    Refines predictions using neighbor features and Stage 1 outputs.
    """

    def __init__(self):
        self.params = Config.LGBM_PARAMS
        self.model_path = Config.LGBM_MODEL_PATH
        self.early_stopping_rounds = Config.EARLY_STOPPING_ROUNDS
        self.verbose_eval = Config.VERBOSE_EVAL

    def train(self, train_feat, val_feat, train_s1, val_s1):
        """
        Trains the LightGBM model using dense features and Stage 1 predictions.
        """
        print("Preparing Stage 2 datasets...")

        # Merge Stage 1 predictions into features
        # Inner join ensures we only train on markdown cells (since S1 only predicts MD)
        train_data = train_feat.merge(train_s1, on=["id", "cell_id"], how="inner")
        val_data = val_feat.merge(val_s1, on=["id", "cell_id"], how="inner")

        # Define Feature Columns (exclude IDs and Target)
        drop_cols = ["id", "cell_id", "target_rank"]
        feature_cols = [c for c in train_data.columns if c not in drop_cols]

        print(f"Training with {len(feature_cols)} features: {feature_cols}")

        X_train = train_data[feature_cols]
        y_train = train_data["target_rank"]

        X_val = val_data[feature_cols]
        y_val = val_data["target_rank"]

        # Create LGBM Datasets
        ds_train = lgb.Dataset(X_train, label=y_train)
        ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train)

        # Train
        print("Starting LightGBM training...")
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self.early_stopping_rounds, verbose=False
            ),
            lgb.log_evaluation(period=self.verbose_eval),
        ]

        model = lgb.train(
            self.params,
            ds_train,
            valid_sets=[ds_train, ds_val],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Log Metrics
        best_score = model.best_score["valid"]["mae"]
        print(f"Training completed. Best Validation MAE: {best_score}")

        # Save Model
        print(f"Saving Stage 2 model to {self.model_path}")
        save_artifacts(model, self.model_path)

        return model

    def predict(self, test_feat, test_s1):
        """
        Generates final predictions using the Stage 2 model.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Stage 2 model not found at {self.model_path}.")

        # Load model
        # LightGBM models saved via save_model are loaded via Booster
        model = lgb.Booster(model_file=self.model_path)

        # Merge
        test_data = test_feat.merge(test_s1, on=["id", "cell_id"], how="inner")

        if len(test_data) == 0:
            return pd.DataFrame(columns=["id", "cell_id", "pred_rank"])

        # Prepare features
        # Ensure columns match training order.
        # Note: In production, strictly enforce column order. Here we infer from dataframe.
        drop_cols = [
            "id",
            "cell_id",
            "target_rank",
        ]  # target_rank might not exist in test, but safe to list
        feature_cols = [c for c in test_data.columns if c not in drop_cols]

        # Verify feature count matches model
        if len(feature_cols) != model.num_feature():
            # Attempt to align if columns are missing or extra
            # This is a basic safety check
            model_feature_names = model.feature_name()
            # Check if all model features are present
            missing = set(model_feature_names) - set(test_data.columns)
            if missing:
                raise ValueError(f"Missing features for prediction: {missing}")
            feature_cols = model_feature_names

        X_test = test_data[feature_cols]

        # Predict
        preds = model.predict(X_test)

        # Result
        df_res = test_data[["id", "cell_id"]].copy()
        df_res["pred_rank"] = preds

        return df_res
