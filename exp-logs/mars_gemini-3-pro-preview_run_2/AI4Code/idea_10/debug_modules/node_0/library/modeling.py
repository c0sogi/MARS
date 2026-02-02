import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from library.config import Config
from library.feature_engine import DualViewVectorizer
from library.utils import kendall_tau_metric, convert_ranks_to_order


class StackedRanker:
    """
    Implements a two-stage stacked ranking pipeline:
    Stage 1: Ridge Regression on TF-IDF vectors (Lexical Signpost).
    Stage 2: LightGBM on Ridge Preds + Anchor Features (Refinement).
    """

    def __init__(self):
        self.config = Config
        self.vectorizer = DualViewVectorizer()

        # Paths for caching
        self.oof_cache_path = os.path.join(
            self.config.CACHE_DIR, "stage1_oof_preds.parquet"
        )

    def _get_markdown_data(self, df):
        """Filters DataFrame for markdown cells and returns relevant parts."""
        mask = df["cell_type"] == "markdown"
        return df[mask].copy().reset_index(drop=True)

    def train_stage1_ridge_oof(self, df_train, load_cached_data=True):
        """
        Trains Stage 1 Ridge Regression using GroupKFold CV to generate OOF predictions.
        Also trains a final Ridge model on the full dataset.

        Args:
            df_train (pd.DataFrame): Raw training data containing 'source', 'cell_type', etc.
            load_cached_data (bool): Whether to load OOF predictions from cache.

        Returns:
            pd.DataFrame: DataFrame containing ['id', 'cell_id', 'ridge_pred'].
        """
        # 1. Check Cache for OOF predictions
        if (
            load_cached_data
            and os.path.exists(self.oof_cache_path)
            and os.path.exists(self.config.RIDGE_MODEL_PATH)
        ):
            print(f"Loading Stage 1 OOF predictions from {self.oof_cache_path}")
            return pd.read_parquet(self.oof_cache_path)

        print("Starting Stage 1: Ridge Regression OOF Training...")

        # 2. Prepare Data
        # Filter for markdown cells only
        df_md = self._get_markdown_data(df_train)

        # Ensure vectorizer is ready
        if not self.vectorizer.load_models():
            print("Fitting vectorizer on markdown corpus...")
            self.vectorizer.fit(df_md["source"].astype(str).fillna("").tolist())

        # Transform text to TF-IDF (Sparse)
        print("Transforming training text...")
        # We only need the sparse matrix for Ridge
        X_sparse, _ = self.vectorizer.transform(
            df_md["source"].astype(str).fillna("").tolist()
        )
        y = df_md["rank"].values
        groups = df_md["ancestor_id"].values

        # 3. Group K-Fold CV
        gkf = GroupKFold(n_splits=self.config.N_FOLDS)
        oof_preds = np.zeros(len(df_md))

        # Initialize Ridge with config params
        model_params = self.config.RIDGE_PARAMS

        print(f"Running {self.config.N_FOLDS}-Fold CV...")
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_sparse, y, groups)):
            X_train_fold = X_sparse[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X_sparse[val_idx]

            model = Ridge(**model_params)
            model.fit(X_train_fold, y_train_fold)

            fold_preds = model.predict(X_val_fold)
            # Clip predictions to [0, 1] range logic (optional but good for rank)
            oof_preds[val_idx] = fold_preds

            # Simple metric print
            mae = np.mean(np.abs(fold_preds - y[val_idx]))
            print(f"Fold {fold+1} MAE: {mae}")

        # 4. Create OOF DataFrame
        df_oof = df_md[["id", "cell_id"]].copy()
        df_oof["ridge_pred"] = oof_preds

        # 5. Train Final Model on Full Data
        print("Training final Stage 1 model on full dataset...")
        final_model = Ridge(**model_params)
        final_model.fit(X_sparse, y)

        # 6. Save Artifacts
        print(f"Saving OOF preds to {self.oof_cache_path}")
        df_oof.to_parquet(self.oof_cache_path, index=False)

        print(f"Saving Ridge model to {self.config.RIDGE_MODEL_PATH}")
        joblib.dump(final_model, self.config.RIDGE_MODEL_PATH)

        return df_oof

    def train_stage2_lgbm(
        self, df_train_features, oof_preds_df, df_val_features, df_val_raw
    ):
        """
        Trains Stage 2 LightGBM Regressor.

        Args:
            df_train_features (pd.DataFrame): Feature engineered training data.
            oof_preds_df (pd.DataFrame): OOF predictions from Stage 1.
            df_val_features (pd.DataFrame): Feature engineered validation data.
            df_val_raw (pd.DataFrame): Raw validation data (needed for GT orders for metric calc).
        """
        print("Starting Stage 2: LightGBM Training...")

        # 1. Merge Ridge Predictions into Features
        # Train set: Use OOF predictions
        train_data = df_train_features.merge(
            oof_preds_df, on=["id", "cell_id"], how="left"
        )

        # Validation set: Must predict using the saved Ridge model
        print("Generating Ridge predictions for validation set...")
        ridge_model = joblib.load(self.config.RIDGE_MODEL_PATH)

        # We need raw text for validation to vectorize
        # Assuming df_val_features aligns with df_val_raw markdown cells,
        # but safer to re-extract text from raw based on ids in features
        # However, df_val_features contains metadata but not source text usually.
        # Let's assume we need to load text.
        # Strategy: Use df_val_raw to get text, vectorize, predict, merge.

        df_val_md = self._get_markdown_data(df_val_raw)
        X_val_sparse, _ = self.vectorizer.transform(
            df_val_md["source"].astype(str).fillna("").tolist()
        )
        val_ridge_preds = ridge_model.predict(X_val_sparse)

        df_val_ridge = df_val_md[["id", "cell_id"]].copy()
        df_val_ridge["ridge_pred"] = val_ridge_preds

        val_data = df_val_features.merge(df_val_ridge, on=["id", "cell_id"], how="left")

        # 2. Prepare LGBM Datasets
        # Drop non-feature columns
        drop_cols = ["id", "cell_id", "rank", "ancestor_id"]
        features = [c for c in train_data.columns if c not in drop_cols]
        target = "rank"

        print(f"Training features: {features}")

        X_train = train_data[features]
        y_train = train_data[target]
        X_val = val_data[features]
        y_val = val_data[target]

        # Create LGBM Datasets
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # 3. Train
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self.config.LGBM_PARAMS["early_stopping_rounds"]
            ),
            lgb.log_evaluation(period=100),
        ]

        # Remove early_stopping_rounds from params as it's passed via callbacks
        params = self.config.LGBM_PARAMS.copy()
        if "early_stopping_rounds" in params:
            del params["early_stopping_rounds"]

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # 4. Save Model
        print(f"Saving LightGBM model to {self.config.LGBM_MODEL_PATH}")
        model.save_model(self.config.LGBM_MODEL_PATH)

        # 5. Evaluate Kendall Tau on Validation
        print("Evaluating Kendall Tau on Validation Set...")
        # Predict on validation
        val_preds = model.predict(X_val)

        # Construct prediction dataframe
        df_pred_rows = val_data[["id", "cell_id"]].copy()
        df_pred_rows["pred_rank"] = val_preds

        # We need to reconstruct the full notebook order (code + markdown)
        # Get code cells from raw validation data
        val_code = (
            df_val_raw[df_val_raw["cell_type"] == "code"]
            .groupby("id")["cell_id"]
            .apply(list)
            .to_dict()
        )

        # Group predicted markdown ranks
        val_md_preds = (
            df_pred_rows.groupby("id")
            .apply(lambda x: dict(zip(x["cell_id"], x["pred_rank"])))
            .to_dict()
        )

        # Build submission format for validation
        submission_rows = []
        for nb_id in df_val_raw["id"].unique():
            code_cells = val_code.get(nb_id, [])
            md_ranks = val_md_preds.get(nb_id, {})

            ordered_str = convert_ranks_to_order(md_ranks, code_cells)
            submission_rows.append({"id": nb_id, "cell_order": ordered_str})

        df_val_pred_final = pd.DataFrame(submission_rows)

        # Get Ground Truth
        df_val_gt = df_val_raw[["id", "cell_order"]].drop_duplicates()

        score = kendall_tau_metric(df_val_gt, df_val_pred_final)
        print(f"Validation Kendall Tau: {score}")

        return model

    def predict(self, df_test_raw, df_test_features):
        """
        Runs the full inference pipeline on test data and generates submission file.

        Args:
            df_test_raw (pd.DataFrame): Raw test data.
            df_test_features (pd.DataFrame): Feature engineered test data.
        """
        print("Starting Inference...")

        # 1. Load Models
        if not os.path.exists(self.config.RIDGE_MODEL_PATH):
            raise FileNotFoundError("Ridge model not found. Train Stage 1 first.")
        if not os.path.exists(self.config.LGBM_MODEL_PATH):
            raise FileNotFoundError("LGBM model not found. Train Stage 2 first.")

        ridge_model = joblib.load(self.config.RIDGE_MODEL_PATH)
        lgbm_model = lgb.Booster(model_file=self.config.LGBM_MODEL_PATH)

        # 2. Stage 1 Inference (Ridge)
        print("Generating Stage 1 predictions...")
        df_test_md = self._get_markdown_data(df_test_raw)

        # Ensure vectorizer is loaded
        if not self.vectorizer.load_models():
            raise FileNotFoundError("Vectorizers not found.")

        X_test_sparse, _ = self.vectorizer.transform(
            df_test_md["source"].astype(str).fillna("").tolist()
        )
        ridge_preds = ridge_model.predict(X_test_sparse)

        df_test_ridge = df_test_md[["id", "cell_id"]].copy()
        df_test_ridge["ridge_pred"] = ridge_preds

        # 3. Merge Features
        test_data = df_test_features.merge(
            df_test_ridge, on=["id", "cell_id"], how="left"
        )

        # 4. Stage 2 Inference (LGBM)
        print("Generating Stage 2 predictions...")
        drop_cols = ["id", "cell_id", "rank", "ancestor_id"]
        features = [c for c in test_data.columns if c not in drop_cols]

        # Ensure feature order matches training
        # (LGBM Booster handles this by name usually, but good to be safe)
        X_test = test_data[features]

        final_preds = lgbm_model.predict(X_test)

        # 5. Construct Submission
        print("Constructing submission file...")
        df_pred_rows = test_data[["id", "cell_id"]].copy()
        df_pred_rows["pred_rank"] = final_preds

        # Get code cells
        test_code = (
            df_test_raw[df_test_raw["cell_type"] == "code"]
            .groupby("id")["cell_id"]
            .apply(list)
            .to_dict()
        )

        # Group predicted markdown ranks
        test_md_preds = (
            df_pred_rows.groupby("id")
            .apply(lambda x: dict(zip(x["cell_id"], x["pred_rank"])))
            .to_dict()
        )

        submission_rows = []
        # Iterate over all test IDs to ensure we cover notebooks with no markdown or no code
        all_test_ids = df_test_raw["id"].unique()

        for nb_id in all_test_ids:
            code_cells = test_code.get(nb_id, [])
            md_ranks = test_md_preds.get(nb_id, {})

            ordered_str = convert_ranks_to_order(md_ranks, code_cells)
            submission_rows.append({"id": nb_id, "cell_order": ordered_str})

        df_submission = pd.DataFrame(submission_rows)

        # Save
        print(f"Saving submission to {self.config.SUBMISSION_PATH}")
        df_submission.to_csv(self.config.SUBMISSION_PATH, index=False)
        print("Submission generated successfully.")
