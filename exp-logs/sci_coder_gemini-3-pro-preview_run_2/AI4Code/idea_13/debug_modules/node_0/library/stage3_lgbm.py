import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from library.config import Config
from library.utils import seed_everything, kendall_tau_metric
from library.stage2_metric import Stage2Metric
from library.feature_engineering import generate_anchor_features
from library.data_loader import NotebookProcessor, load_metadata


class Stage3LGBM:
    """
    Stage 3: Neighborhood Gradient Booster (LightGBM).
    Refines rank predictions using Ridge OOF scores and Metric Anchor features.
    """

    def __init__(self, config=Config):
        self.config = config
        self.stage2 = Stage2Metric(config)

    def _load_ridge_predictions(self, split):
        """
        Loads the Ridge predictions for the specified split.
        """
        if split == "train":
            path = self.config.TRAIN_RIDGE_OOF_PATH
        elif split == "val":
            path = self.config.VAL_RIDGE_PREDS_PATH
        elif split == "test":
            path = self.config.TEST_RIDGE_PREDS_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Ridge predictions not found at {path}. Run Stage 1 first."
            )

        return pd.read_parquet(path)

    def _prepare_features(self, split, load_cached_features=True):
        """
        Generates or loads the feature set for Stage 3.
        Features: [Ridge_Pred, Anchor_Mean, Anchor_Weighted, Anchor_MinDist, Anchor_Nearest]
        """
        # Determine cache path
        if split == "train":
            feat_path = self.config.TRAIN_FEATURES_PATH
        elif split == "val":
            feat_path = self.config.VAL_FEATURES_PATH
        elif split == "test":
            feat_path = self.config.TEST_FEATURES_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        # 1. Try Load Cache
        if load_cached_features and os.path.exists(feat_path):
            print(f"Loading cached Stage 3 features for {split} from {feat_path}...")
            return pd.read_parquet(feat_path)

        print(f"Generating Stage 3 features for {split}...")

        # 2. Load Base Data
        processor = NotebookProcessor(self.config)
        df = processor.load_data(split)

        # 3. Load Ridge Predictions
        # Ridge preds dataframe has columns [id, cell_id, ridge_pred]
        df_ridge = self._load_ridge_predictions(split)

        # Merge Ridge Preds
        # Ensure alignment. We merge on id and cell_id.
        df = df.merge(df_ridge, on=["id", "cell_id"], how="left")

        # 4. Generate Stage 2 Embeddings
        # We need text source
        texts = df["source"].fillna("").astype(str).tolist()
        # Get embeddings (this handles SVD + Siamese projection)
        # We use the trained Stage 2 model
        embeddings = self.stage2.get_projected_embeddings(texts, load_cached_model=True)

        # 5. Generate Anchor Features
        # This returns a dataframe with index aligned to df
        # We need to ensure df index is clean (0..N-1)
        df = df.reset_index(drop=True)

        # Pass subset of df required for grouping
        anchor_feats = generate_anchor_features(
            df[["id", "cell_type"]],
            embeddings,
            top_k=self.config.TOP_K_ANCHORS,
            n_jobs=self.config.NUM_WORKERS,
        )

        # 6. Combine
        # anchor_feats index matches df index
        df_features = pd.concat([df, anchor_feats], axis=1)

        # 7. Save to Cache
        os.makedirs(os.path.dirname(feat_path), exist_ok=True)
        df_features.to_parquet(feat_path, index=False)
        print(f"Saved features to {feat_path}")

        return df_features

    def train(self, load_cached_features=True):
        """
        Trains the LightGBM model on markdown cells.
        """
        seed_everything(self.config.SEED)

        # 1. Prepare Data
        df_train = self._prepare_features("train", load_cached_features)
        df_val = self._prepare_features("val", load_cached_features)

        # Filter for Markdown cells only
        train_mask = df_train["cell_type"] == "markdown"
        val_mask = df_val["cell_type"] == "markdown"

        feature_cols = [
            "ridge_pred",
            "anchor_mean_rank",
            "anchor_weighted_rank",
            "anchor_min_dist",
            "anchor_nearest_rank",
        ]

        X_train = df_train.loc[train_mask, feature_cols]
        y_train = df_train.loc[train_mask, "norm_rank"]

        X_val = df_val.loc[val_mask, feature_cols]
        y_val = df_val.loc[val_mask, "norm_rank"]

        print(
            f"Training LightGBM on {len(X_train)} samples, validating on {len(X_val)} samples..."
        )

        # 2. Train Model
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [
            lgb.early_stopping(stopping_rounds=self.config.LGBM_EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=100),
        ]

        model = lgb.train(
            self.config.LGBM_PARAMS,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # 3. Evaluation (Kendall Tau)
        print("Evaluating Kendall Tau on Validation Set...")

        # Predict on validation markdown
        val_preds = model.predict(X_val)

        # Create a working copy for ranking
        df_val_eval = df_val.copy()

        # Assign predictions to markdown cells
        df_val_eval.loc[val_mask, "pred_rank"] = val_preds

        # Assign linear ranks to code cells (0.0 to 1.0)
        # We define a helper to apply per notebook
        def assign_code_ranks(group):
            code_mask = group["cell_type"] == "code"
            n_code = code_mask.sum()
            if n_code > 0:
                ranks = np.linspace(0.0, 1.0, n_code)
                group.loc[code_mask, "pred_rank"] = ranks
            return group

        # Apply ranking logic
        df_val_eval = df_val_eval.groupby("id", group_keys=False).apply(
            assign_code_ranks
        )

        # Fill any remaining NaNs (e.g., notebooks with no code cells)
        df_val_eval["pred_rank"] = df_val_eval["pred_rank"].fillna(0.0)

        # Sort by predicted rank
        df_val_eval = df_val_eval.sort_values(["id", "pred_rank"])

        # Create ordering string
        pred_orders = (
            df_val_eval.groupby("id")["cell_id"]
            .apply(lambda x: " ".join(x))
            .reset_index()
        )
        pred_orders.columns = ["id", "cell_order"]

        # Load Ground Truth
        val_meta = load_metadata("val")

        score = kendall_tau_metric(pred_orders, val_meta)
        print(f"Stage 3 Validation Kendall Tau: {score}")

        # 4. Save Model
        model_path = os.path.join(self.config.WORKING_DIR, "lgbm_model.txt")
        model.save_model(model_path)
        print(f"Model saved to {model_path}")

        return model

    def predict(self, load_cached_features=True):
        """
        Generates predictions for the test set and saves the submission file.
        """
        seed_everything(self.config.SEED)

        # 1. Load Model
        model_path = os.path.join(self.config.WORKING_DIR, "lgbm_model.txt")
        if not os.path.exists(model_path):
            raise FileNotFoundError("LGBM model not found. Train first.")

        model = lgb.Booster(model_file=model_path)

        # 2. Prepare Test Features
        df_test = self._prepare_features("test", load_cached_features)

        # 3. Predict Markdown
        test_mask = df_test["cell_type"] == "markdown"

        feature_cols = [
            "ridge_pred",
            "anchor_mean_rank",
            "anchor_weighted_rank",
            "anchor_min_dist",
            "anchor_nearest_rank",
        ]

        if test_mask.sum() > 0:
            X_test = df_test.loc[test_mask, feature_cols]
            preds = model.predict(X_test)
            df_test.loc[test_mask, "pred_rank"] = preds
        else:
            df_test["pred_rank"] = 0.0

        # 4. Assign Code Ranks
        def assign_code_ranks_test(group):
            code_mask = group["cell_type"] == "code"
            n_code = code_mask.sum()
            if n_code > 0:
                ranks = np.linspace(0.0, 1.0, n_code)
                group.loc[code_mask, "pred_rank"] = ranks
            return group

        df_test = df_test.groupby("id", group_keys=False).apply(assign_code_ranks_test)
        df_test["pred_rank"] = df_test["pred_rank"].fillna(0.0)

        # 5. Sort and Submission
        df_test = df_test.sort_values(["id", "pred_rank"])

        submission = (
            df_test.groupby("id")["cell_id"].apply(lambda x: " ".join(x)).reset_index()
        )
        submission.columns = ["id", "cell_order"]

        submission.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")


def run_stage3(load_cached_features=True):
    """
    Helper function to run Stage 3 training and inference.
    """
    stage3 = Stage3LGBM(Config)
    stage3.train(load_cached_features=load_cached_features)
    stage3.predict(load_cached_features=load_cached_features)
