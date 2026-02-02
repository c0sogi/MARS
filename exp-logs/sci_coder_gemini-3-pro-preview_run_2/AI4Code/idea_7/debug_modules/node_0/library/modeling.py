import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from library.config import Config
from library.feature_engineering import FeatureEngineer, VectorizationPipeline
from library.metrics import score_dataframe


class StackedRanker:
    """
    Implements the Two-Stage Stacked Regression Pipeline.
    Stage 1: Ridge Regression on TF-IDF vectors.
    Stage 2: LightGBM on stacked features (Ridge Preds + LSA + Anchors + Meta).
    """

    def __init__(self):
        self.fe = FeatureEngineer(load_cached_data=True)
        self.pipeline = VectorizationPipeline()
        self.models_dir = Config.WORKING_DIR
        self.submission_dir = Config.SUBMISSION_DIR

        # Ensure directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

    def _get_tfidf_matrix(self, df):
        """Transforms dataframe source text to TF-IDF matrix using the pre-fitted pipeline."""
        texts = df["source"].fillna("").astype(str).tolist()
        return self.pipeline.transform_tfidf(texts)

    def train_stage1(self, df_train, df_val, load_cached_preds=True):
        """
        Stage 1: Ridge Regression.
        - Generates OOF predictions for Train (via CV).
        - Trains final model on full Train.
        - Generates predictions for Val.
        """
        oof_cache_path = os.path.join(self.models_dir, "train_ridge_oof.npy")
        val_pred_cache_path = os.path.join(self.models_dir, "val_ridge_preds.npy")
        model_path = os.path.join(self.models_dir, "ridge_model.joblib")

        # Check cache
        if (
            load_cached_preds
            and os.path.exists(oof_cache_path)
            and os.path.exists(val_pred_cache_path)
            and os.path.exists(model_path)
        ):
            print("[Stage 1] Loading cached predictions and model...")
            train_oof = np.load(oof_cache_path)
            val_preds = np.load(val_pred_cache_path)
            ridge_model = joblib.load(model_path)
            return train_oof, val_preds, ridge_model

        print("[Stage 1] Training Ridge Regression with CV...")

        # Prepare Data
        X_train = self._get_tfidf_matrix(df_train)
        y_train = df_train["rank"].values
        groups = df_train["ancestor_id"].values

        X_val = self._get_tfidf_matrix(df_val)

        # Initialize containers
        train_oof = np.zeros(len(df_train))

        # 5-Fold Group CV
        gkf = GroupKFold(n_splits=Config.N_FOLDS)

        for fold, (train_idx, valid_idx) in enumerate(
            gkf.split(X_train, y_train, groups)
        ):
            X_tr_fold, y_tr_fold = X_train[train_idx], y_train[train_idx]
            X_val_fold = X_train[valid_idx]

            model = Ridge(alpha=Config.RIDGE_ALPHA, random_state=Config.RANDOM_STATE)
            model.fit(X_tr_fold, y_tr_fold)

            train_oof[valid_idx] = model.predict(X_val_fold)

        # Train Final Model on Full Data
        print("[Stage 1] Training final Ridge model on full dataset...")
        final_model = Ridge(alpha=Config.RIDGE_ALPHA, random_state=Config.RANDOM_STATE)
        final_model.fit(X_train, y_train)

        # Predict on Validation
        val_preds = final_model.predict(X_val)

        # Cache results
        np.save(oof_cache_path, train_oof)
        np.save(val_pred_cache_path, val_preds)
        joblib.dump(final_model, model_path)

        return train_oof, val_preds, final_model

    def _prepare_stage2_features(self, df, ridge_preds):
        """
        Constructs the feature matrix for LightGBM.
        Features: Ridge Prediction, LSA vectors, Anchor Features, Metadata.
        """
        # 1. Ridge Prediction
        feat_ridge = pd.DataFrame({"ridge_pred": ridge_preds}, index=df.index)

        # 2. LSA Features (already in df)
        lsa_cols = [c for c in df.columns if c.startswith("lsa_")]
        feat_lsa = df[lsa_cols]

        # 3. Anchor Features (already in df)
        anchor_cols = ["anchor_rank", "anchor_sim", "top3_anchor_rank_mean"]
        feat_anchor = df[anchor_cols]

        # 4. Metadata (already in df)
        meta_cols = ["total_cells", "md_ratio"]
        feat_meta = df[meta_cols]

        # Combine
        X = pd.concat([feat_ridge, feat_lsa, feat_anchor, feat_meta], axis=1)
        return X

    def train_stage2(self, df_train, df_val, train_oof, val_preds):
        """
        Stage 2: LightGBM Regressor.
        Trains on stacked features with Early Stopping.
        """
        print("[Stage 2] Preparing datasets for LightGBM...")
        X_train = self._prepare_stage2_features(df_train, train_oof)
        y_train = df_train["rank"].values

        X_val = self._prepare_stage2_features(df_val, val_preds)
        y_val = df_val["rank"].values

        print(f"[Stage 2] Training LightGBM (Train shape: {X_train.shape})...")

        # Create LGBM Datasets
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        callbacks = [
            lgb.early_stopping(stopping_rounds=Config.LGBM_EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=Config.LGBM_VERBOSE_EVAL),
        ]

        model = lgb.train(
            Config.LGBM_PARAMS,
            dtrain,
            num_boost_round=Config.LGBM_NUM_BOOST_ROUND,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Save model
        model.save_model(os.path.join(self.models_dir, "lgbm_model.txt"))

        # Validation Score
        val_final_preds = model.predict(X_val)
        mae = np.mean(np.abs(val_final_preds - y_val))
        print(f"[Stage 2] Validation MAE: {mae:.16f}")

        return model

    def run(self):
        """
        Executes the full training and inference pipeline.
        """
        # 1. Load Data & Pipeline
        # This will fit the pipeline on train data if not cached
        df_train, df_nb_train = self.fe.process_split("train")
        df_val, df_nb_val = self.fe.process_split("val")

        # Load the fitted pipeline for transforming text
        self.pipeline.load(self.models_dir)

        # 2. Train Stage 1 (Ridge)
        train_oof, val_preds, ridge_model = self.train_stage1(df_train, df_val)

        # 3. Train Stage 2 (LightGBM)
        lgbm_model = self.train_stage2(df_train, df_val, train_oof, val_preds)

        # 4. Inference on Test Set
        print("\n[Inference] Processing Test Set...")
        df_test, df_nb_test = self.fe.process_split("test")

        # Stage 1 Prediction
        print("[Inference] Generating Stage 1 predictions...")
        X_test = self._get_tfidf_matrix(df_test)
        test_ridge_preds = ridge_model.predict(X_test)

        # Stage 2 Prediction
        print("[Inference] Generating Stage 2 predictions...")
        X_test_stk = self._prepare_stage2_features(df_test, test_ridge_preds)
        test_final_preds = lgbm_model.predict(X_test_stk)

        # 5. Generate Submission
        print("[Inference] constructing cell orders...")
        submission_df = self._generate_submission(df_test, df_nb_test, test_final_preds)

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    def _generate_submission(self, df_md, df_nb, md_preds):
        """
        Merges predicted markdown ranks with fixed code ranks to produce final ordering.
        """
        # Add predictions to dataframe
        df_md = df_md.copy()
        df_md["pred_rank"] = md_preds

        # Map notebook_id to code cells
        nb_code_map = dict(zip(df_nb["notebook_id"], df_nb["code_ids"]))

        submission_data = []

        # Group by notebook
        grouped = df_md.groupby("notebook_id")

        # Iterate over all test notebooks (from df_nb to ensure we cover those with 0 md cells if any)
        for nb_id in df_nb["notebook_id"].unique():
            # Get code cells
            code_ids = nb_code_map.get(nb_id, [])

            # Get markdown cells
            if nb_id in grouped.groups:
                md_group = grouped.get_group(nb_id)
                md_ids = md_group["cell_id"].tolist()
                md_ranks = md_group["pred_rank"].tolist()
            else:
                md_ids = []
                md_ranks = []

            # Create unified list with ranks
            cells = []

            # Assign ranks to code cells: equidistant in [0, 1]
            n_code = len(code_ids)
            if n_code > 0:
                # e.g., 0/(N-1), 1/(N-1), ...
                # If N=1, rank=0.
                if n_code == 1:
                    code_ranks = [0.0]
                else:
                    code_ranks = [i / (n_code - 1) for i in range(n_code)]
            else:
                code_ranks = []

            # Add Code Cells
            for cid, rank in zip(code_ids, code_ranks):
                cells.append((cid, rank))

            # Add Markdown Cells
            for cid, rank in zip(md_ids, md_ranks):
                cells.append((cid, rank))

            # Sort by rank
            cells.sort(key=lambda x: x[1])

            # Extract IDs
            sorted_ids = [c[0] for c in cells]
            order_str = " ".join(sorted_ids)

            submission_data.append({"id": nb_id, "cell_order": order_str})

        return pd.DataFrame(submission_data)
