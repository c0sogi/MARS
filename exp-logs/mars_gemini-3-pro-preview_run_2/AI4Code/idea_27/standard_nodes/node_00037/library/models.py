import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

from library.config import Config
from library.utils import set_seed, format_submission
from library.data_manager import NotebookLoader
from library.vectorization import TextProcessor
from library.feature_extraction import FeatureEngineer


class Stage1Ridge:
    def __init__(self, config):
        self.config = config
        self.model_path = os.path.join(config.WORKING_DIR, "stage1_ridge_model.joblib")
        self.oof_path = os.path.join(config.WORKING_DIR, "stage1_oof_preds.parquet")
        self.model = None

    def fit_oof(self, df_corpus, tfidf_mat, df_features, load_cached_data=True):
        """
        Trains Ridge Regression using GroupKFold to generate OOF predictions.
        Retrains final model on full data.
        """
        # Check cache
        if (
            load_cached_data
            and os.path.exists(self.model_path)
            and os.path.exists(self.oof_path)
        ):
            print("Loading cached Stage 1 model and OOF predictions...")
            self.model = joblib.load(self.model_path)
            df_oof = pd.read_parquet(self.oof_path)
            if len(df_oof) == len(df_features):
                return df_oof["ridge_pred"].values
            else:
                print("Cached OOF shape mismatch. Retraining...")

        print("Training Stage 1 (Ridge) with 5-Fold CV...")

        # Align tfidf_mat (all cells) to df_features (markdown cells)
        # Create a map from cell_id to matrix index
        cell_id_to_idx = pd.Series(data=df_corpus.index, index=df_corpus["cell_id"])

        # Get indices for feature rows
        feature_indices = cell_id_to_idx.loc[df_features["cell_id"]].values

        X = tfidf_mat[feature_indices]
        y = df_features["target"].values
        groups = df_features["id"].values

        oof_preds = np.zeros(len(y))
        gkf = GroupKFold(n_splits=self.config.N_FOLDS)

        scores = []
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = Ridge(alpha=self.config.RIDGE_ALPHA)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            preds = np.clip(preds, 0, 1)

            oof_preds[val_idx] = preds
            score = mean_absolute_error(y_val, preds)
            scores.append(score)
            print(f"Fold {fold+1} Ridge MAE: {score}")

        print(f"Average Ridge MAE: {np.mean(scores)}")

        # Train final model on full dataset
        print("Retraining Stage 1 Ridge on full dataset...")
        final_model = Ridge(alpha=self.config.RIDGE_ALPHA)
        final_model.fit(X, y)
        self.model = final_model

        # Save results
        joblib.dump(self.model, self.model_path)
        pd.DataFrame({"ridge_pred": oof_preds}).to_parquet(self.oof_path)

        return oof_preds

    def predict(self, df_corpus, tfidf_mat, df_features):
        """
        Predicts using the trained Ridge model.
        """
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
            else:
                raise ValueError("Stage 1 model not trained.")

        # Map features to sparse matrix rows
        cell_id_to_idx = pd.Series(data=df_corpus.index, index=df_corpus["cell_id"])
        feature_indices = cell_id_to_idx.loc[df_features["cell_id"]].values

        X = tfidf_mat[feature_indices]
        preds = self.model.predict(X)
        return np.clip(preds, 0, 1)


class Stage2LGBM:
    def __init__(self, config):
        self.config = config
        self.model_path = os.path.join(config.WORKING_DIR, "stage2_lgbm_model.txt")
        self.model = None
        self.feature_cols = None

    def fit(self, df_features, load_cached_data=True):
        """
        Trains LightGBM with Early Stopping.
        """
        # Define feature columns (exclude IDs and Target)
        exclude_cols = ["id", "cell_id", "target"]
        self.feature_cols = [c for c in df_features.columns if c not in exclude_cols]

        if load_cached_data and os.path.exists(self.model_path):
            print("Loading cached Stage 2 model...")
            self.model = lgb.Booster(model_file=self.model_path)
            return

        print("Training Stage 2 (LightGBM)...")

        X = df_features[self.feature_cols]
        y = df_features["target"]
        groups = df_features["id"]

        # Use 1 fold of GroupKFold for validation to respect notebook groups
        gkf = GroupKFold(n_splits=self.config.N_FOLDS)
        train_idx, val_idx = next(gkf.split(X, y, groups))

        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

        params = self.config.LGBM_PARAMS.copy()

        self.model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=params.get("early_stopping_rounds", 50)
                ),
                lgb.log_evaluation(period=100),
            ],
        )

        self.model.save_model(self.model_path)

    def predict(self, df_features):
        """
        Predicts using the trained LightGBM model.
        """
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = lgb.Booster(model_file=self.model_path)
            else:
                raise ValueError("Stage 2 model not trained.")

        # Ensure feature columns match training
        if self.feature_cols is None:
            exclude_cols = ["id", "cell_id", "target", "pred_rank"]
            self.feature_cols = [
                c for c in df_features.columns if c not in exclude_cols
            ]

        return self.model.predict(df_features[self.feature_cols])


def run_task(load_cached_data=True):
    """
    Orchestrates the full pipeline: Loading -> Processing -> Training -> Inference.
    """
    config = Config()
    set_seed(42)

    # --------------------------------------------------------------------------
    # 1. Load and Prepare Training Data
    # --------------------------------------------------------------------------
    loader = NotebookLoader(config)
    df_train_corpus, df_val_corpus = loader.prepare_datasets(
        load_cached_data=load_cached_data
    )

    # Combine train and val for full dataset training (as per strategy)
    df_full_corpus = pd.concat([df_train_corpus, df_val_corpus], ignore_index=True)

    # --------------------------------------------------------------------------
    # 2. Vectorization (TF-IDF & SVD)
    # --------------------------------------------------------------------------
    processor = TextProcessor(config)
    processor.fit_pipeline(df_full_corpus, load_cached_models=load_cached_data)
    tfidf_mat, svd_mat = processor.transform_cells(
        df_full_corpus, mode="train", load_cached_data=load_cached_data
    )

    # --------------------------------------------------------------------------
    # 3. Feature Extraction
    # --------------------------------------------------------------------------
    engineer = FeatureEngineer(config)
    df_features = engineer.extract_features(
        df_full_corpus,
        tfidf_mat,
        svd_mat,
        mode="train",
        load_cached_data=load_cached_data,
    )

    # --------------------------------------------------------------------------
    # 4. Stage 1: Ridge Regression (OOF)
    # --------------------------------------------------------------------------
    stage1 = Stage1Ridge(config)
    oof_preds = stage1.fit_oof(
        df_full_corpus, tfidf_mat, df_features, load_cached_data=load_cached_data
    )
    df_features["ridge_pred"] = oof_preds

    # --------------------------------------------------------------------------
    # 5. Stage 2: LightGBM (Stacking)
    # --------------------------------------------------------------------------
    stage2 = Stage2LGBM(config)
    stage2.fit(df_features, load_cached_data=load_cached_data)

    # --------------------------------------------------------------------------
    # 6. Inference on Test Set
    # --------------------------------------------------------------------------
    print("Running Inference on Test Set...")
    df_test_corpus = loader.load_test_data(load_cached_data=load_cached_data)
    tfidf_test, svd_test = processor.transform_cells(
        df_test_corpus, mode="test", load_cached_data=load_cached_data
    )
    df_test_features = engineer.extract_features(
        df_test_corpus,
        tfidf_test,
        svd_test,
        mode="test",
        load_cached_data=load_cached_data,
    )

    # Stage 1 Inference
    ridge_test_preds = stage1.predict(df_test_corpus, tfidf_test, df_test_features)
    df_test_features["ridge_pred"] = ridge_test_preds

    # Stage 2 Inference
    final_preds = stage2.predict(df_test_features)
    df_test_features["pred_rank"] = final_preds

    # --------------------------------------------------------------------------
    # 7. Post-Processing and Submission
    # --------------------------------------------------------------------------
    submission_rows = []

    # Map predictions back to cell IDs for fast lookup
    pred_map = dict(zip(df_test_features["cell_id"], df_test_features["pred_rank"]))

    # Process each test notebook to reconstruct order
    for nb_id, group in df_test_corpus.groupby("id"):
        cells = group.copy()

        # Separate code and markdown masks
        code_mask = cells["cell_type"] == "code"
        md_mask = cells["cell_type"] == "markdown"

        # Assign Ranks
        # Code cells: Fixed anchors evenly distributed [0, 1]
        n_code = code_mask.sum()
        if n_code > 0:
            cells.loc[code_mask, "rank"] = np.linspace(0, 1, n_code)

        # Markdown cells: Predicted ranks
        # Use map to fill ranks; if any MD cell missing from features (unlikely), it gets NaN -> handle if needed
        cells.loc[md_mask, "rank"] = cells.loc[md_mask, "cell_id"].map(pred_map)

        # Sort by rank
        cells = cells.sort_values("rank")
        cell_order = " ".join(cells["cell_id"].tolist())

        submission_rows.append({"id": nb_id, "cell_order": cell_order})

    df_sub = pd.DataFrame(submission_rows)
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
