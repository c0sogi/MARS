import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Dict, Union, Optional
from library.config import Config
from library.utils import get_logger, Timer


class Stage1Ridge:
    """
    Wrapper for the Stage 1 Ridge Regression model.
    The model is trained and saved by the FeaturePipeline in features.py.
    This class provides an interface to load the model and make predictions if needed
    outside the feature pipeline context.
    """

    def __init__(self):
        self.logger = get_logger("Stage1Ridge")
        self.model_path = os.path.join(Config.WORKING_DIR, "ridge_model.joblib")
        self.model = None

    def load(self):
        """Loads the pre-trained Ridge model from disk."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.logger.info(f"Loaded Ridge model from {self.model_path}")
        else:
            self.logger.warning(f"Ridge model not found at {self.model_path}")

    def predict(self, X) -> np.ndarray:
        """
        Makes predictions using the loaded Ridge model.
        Args:
            X: Sparse matrix or array-like features (TF-IDF).
        Returns:
            np.ndarray: Predicted ranks.
        """
        if self.model is None:
            self.load()

        if self.model is None:
            raise RuntimeError("Ridge model is not loaded. Ensure features.py has run.")

        return self.model.predict(X)


class Stage2LGBM:
    """
    Stage 2 LightGBM Regressor.
    Refines the ranking by combining Stage 1 Ridge predictions with
    distributional anchor features and metadata.
    """

    def __init__(self):
        self.logger = get_logger("Stage2LGBM")
        self.working_dir = Config.WORKING_DIR
        self.model_path = os.path.join(self.working_dir, "lgbm_model.txt")
        self.model = None
        self.feature_cols = []

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Identifies feature columns by excluding metadata and target columns.
        """
        exclude_cols = {
            "notebook_id",
            "cell_id",
            "target",
            "original_index",
            "source",
            "rank",
            "code_ids",
            "code_sources",
        }
        return [c for c in df.columns if c not in exclude_cols]

    def train(self, df_train: pd.DataFrame, df_val: pd.DataFrame):
        """
        Trains the LightGBM model with early stopping.
        Args:
            df_train (pd.DataFrame): Training data with features and 'target'.
            df_val (pd.DataFrame): Validation data with features and 'target'.
        """
        with Timer("Stage 2 Training"):
            self.feature_cols = self._get_feature_columns(df_train)
            self.logger.info(
                f"Training with {len(self.feature_cols)} features: {self.feature_cols}"
            )

            X_train = df_train[self.feature_cols]
            y_train = df_train["target"]
            X_val = df_val[self.feature_cols]
            y_val = df_val["target"]

            # Create LGBM Datasets
            ds_train = lgb.Dataset(X_train, label=y_train)
            ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train)

            params = Config.get_lgbm_params()

            # Callbacks
            callbacks = [
                lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(period=50),
            ]

            self.model = lgb.train(
                params,
                ds_train,
                valid_sets=[ds_train, ds_val],
                valid_names=["train", "valid"],
                callbacks=callbacks,
            )

            # Save model
            self.model.save_model(self.model_path)
            self.logger.info(f"LGBM model saved to {self.model_path}")

            # Log validation score
            val_preds = self.model.predict(X_val)
            val_mae = np.mean(np.abs(val_preds - y_val))
            self.logger.info(f"Final Validation MAE: {val_mae:.8f}")

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for the test set.
        Args:
            df_test (pd.DataFrame): Test data with features.
        Returns:
            np.ndarray: Predicted ranks.
        """
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = lgb.Booster(model_file=self.model_path)
                self.logger.info(f"Loaded LGBM model from {self.model_path}")
            else:
                raise RuntimeError("LGBM model not found. Train the model first.")

        # Ensure features match training
        # If feature_cols is empty (loaded from disk without training), infer from df
        # Note: In production, strictly loading feature names from metadata is safer,
        # but here we assume df_test has same schema as train.
        cols = self._get_feature_columns(df_test)
        X_test = df_test[cols]

        return self.model.predict(X_test)


class SubmissionGenerator:
    """
    Handles the reconstruction of cell order and generation of the submission file.
    """

    def __init__(self):
        self.logger = get_logger("SubmissionGenerator")
        self.submission_path = Config.SUBMISSION_PATH

    def generate(self, df_md_pred: pd.DataFrame, df_nb: pd.DataFrame):
        """
        Generates the submission CSV.

        Args:
            df_md_pred (pd.DataFrame): DataFrame containing 'notebook_id', 'cell_id', and 'pred_rank'.
            df_nb (pd.DataFrame): DataFrame containing 'notebook_id' and 'code_ids' (list of strings).
        """
        with Timer("Submission Generation"):
            self.logger.info("Reconstructing cell orders...")

            # Convert notebooks to a dictionary for fast lookup
            # nb_id -> list of code_ids
            nb_code_map = df_nb.set_index("notebook_id")["code_ids"].to_dict()

            # Group predictions by notebook
            grouped_preds = df_md_pred.groupby("notebook_id")

            submission_rows = []

            # Iterate over all test notebooks found in the predictions
            # Note: We must ensure we cover all notebooks in the test set.
            # df_nb contains all notebooks in the split.

            all_nb_ids = df_nb["notebook_id"].unique()

            for nb_id in all_nb_ids:
                # 1. Get Code Cells and assign fixed ranks
                code_ids = nb_code_map.get(nb_id, [])
                n_code = len(code_ids)

                cells_with_ranks = []

                if n_code > 0:
                    # Ranks: 0.0 to 1.0
                    if n_code == 1:
                        code_ranks = [0.0]
                    else:
                        code_ranks = np.linspace(0, 1, n_code)

                    for cid, r in zip(code_ids, code_ranks):
                        cells_with_ranks.append((cid, r))

                # 2. Get Markdown Cells and their predicted ranks
                if nb_id in grouped_preds.groups:
                    group = grouped_preds.get_group(nb_id)
                    for _, row in group.iterrows():
                        cells_with_ranks.append((row["cell_id"], row["pred_rank"]))

                # 3. Sort by rank
                cells_with_ranks.sort(key=lambda x: x[1])

                # 4. Extract ordered IDs
                ordered_ids = [c[0] for c in cells_with_ranks]
                cell_order_str = " ".join(ordered_ids)

                submission_rows.append({"id": nb_id, "cell_order": cell_order_str})

            # Create DataFrame and save
            df_sub = pd.DataFrame(submission_rows)
            df_sub.to_csv(self.submission_path, index=False)
            self.logger.info(
                f"Submission saved to {self.submission_path} with {len(df_sub)} rows."
            )
