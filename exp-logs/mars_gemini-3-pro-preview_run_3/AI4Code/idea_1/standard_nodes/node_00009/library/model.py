import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from library.config import Config
from library.features import SemanticFeatureExtractor
from library.data_loader import load_notebook


class PositionRegressor:
    """
    Wraps a Gradient Boosting Regressor (LightGBM) to predict the relative position
    of markdown cells within a notebook.
    """

    def __init__(self):
        self.model = None
        self.extractor = SemanticFeatureExtractor()
        self.model_path = os.path.join(Config.CACHE_DIR, "lgbm_model.txt")

    def train(self, df_train_meta, df_val_meta):
        """
        Trains the LightGBM model using features extracted from the notebooks.

        Args:
            df_train_meta (pd.DataFrame): Training metadata.
            df_val_meta (pd.DataFrame): Validation metadata.
        """
        print("Starting model training pipeline...")

        # 1. Fit/Load Vectorizer
        # We use the training set to define the vocabulary
        self.extractor.fit_vectorizer(df_train_meta, load_cached=True)

        # 2. Generate Features
        # These methods handle caching internally (parquet files)
        print("Preparing training features...")
        df_train_feats = self.extractor.generate_dataset(df_train_meta, mode="train")
        print("Preparing validation features...")
        df_val_feats = self.extractor.generate_dataset(df_val_meta, mode="val")

        # 3. Prepare LightGBM Datasets
        # Exclude non-feature columns
        exclude_cols = {"id", "cell_id", "target"}
        feature_cols = [c for c in df_train_feats.columns if c not in exclude_cols]

        print(f"Training with {len(feature_cols)} features: {feature_cols}")

        X_train = df_train_feats[feature_cols]
        y_train = df_train_feats["target"]
        X_val = df_val_feats[feature_cols]
        y_val = df_val_feats["target"]

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        # 4. Train Model
        params = Config.MODEL_PARAMS.copy()
        n_estimators = params.pop("n_estimators", 1000)

        # Callbacks for early stopping and logging
        callbacks = [
            lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=Config.VERBOSE_EVAL),
        ]

        print("Fitting LightGBM model...")
        self.model = lgb.train(
            params,
            train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            num_boost_round=n_estimators,
            callbacks=callbacks,
        )

        # 5. Save Model
        print(f"Saving model to {self.model_path}...")
        self.model.save_model(self.model_path)
        print("Training complete.")

    def predict(self, df_metadata, mode="test"):
        """
        Generates predictions for the provided dataset.

        Args:
            df_metadata (pd.DataFrame): Metadata for the notebooks to predict.
            mode (str): The mode ('val' or 'test') to ensure correct feature caching/loading.

        Returns:
            pd.DataFrame: DataFrame containing 'id', 'cell_id', and 'pred_pos'.
        """
        # Load model if not present
        if self.model is None:
            if os.path.exists(self.model_path):
                print(f"Loading model from {self.model_path}...")
                self.model = lgb.Booster(model_file=self.model_path)
            else:
                raise RuntimeError("Model not found. Please train the model first.")

        # Ensure vectorizer is ready (extractor handles loading from cache)
        if self.extractor.model is None:
            # Trigger loading by calling fit_vectorizer logic implicitly via generate_dataset
            # or explicitly if needed. generate_dataset handles the check.
            pass

        print(f"Generating features for {mode} set...")
        df_feats = self.extractor.generate_dataset(df_metadata, mode=mode)

        if df_feats.empty:
            print("Warning: No features generated (possibly no markdown cells).")
            return pd.DataFrame(columns=["id", "cell_id", "pred_pos"])

        exclude_cols = {"id", "cell_id", "target"}
        feature_cols = [c for c in df_feats.columns if c not in exclude_cols]

        print("Running inference...")
        preds = self.model.predict(df_feats[feature_cols])

        df_res = df_feats[["id", "cell_id", "n_code"]].copy()
        df_res["pred_pos"] = preds

        return df_res

    def generate_submission(self, df_test_meta):
        """
        Runs the full inference pipeline and saves the submission file.

        Args:
            df_test_meta (pd.DataFrame): Test metadata.
        """
        # 1. Get Predictions
        df_preds = self.predict(df_test_meta)

        # 2. Reconstruct Cell Order
        print("Reconstructing cell orders...")

        # Group predictions by notebook ID for fast access
        if not df_preds.empty:
            pred_groups = df_preds.groupby("id")
        else:
            pred_groups = None

        submission_rows = []

        # Iterate through each test notebook to build the ordered list
        for _, row in df_test_meta.iterrows():
            nb_id = row["id"]
            file_path = row["file_path"]

            # Load notebook to get code cells (anchors)
            try:
                nb_data = load_notebook(file_path)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                submission_rows.append({"id": nb_id, "cell_order": ""})
                continue

            # Code cells are anchors with fixed integer ranks: 0, 1, 2, ...
            # In the test set, code cells are provided in the correct order.
            code_cells = list(nb_data["code_cells"].keys())
            n_code = len(code_cells)

            cells_with_ranks = []

            # Add code cells
            for i, cid in enumerate(code_cells):
                cells_with_ranks.append((float(i), cid))

            # Add markdown cells
            if pred_groups is not None and nb_id in pred_groups.groups:
                md_df = pred_groups.get_group(nb_id)

                for _, r in md_df.iterrows():
                    cid = r["cell_id"]
                    pred_rel = r["pred_pos"]

                    # Convert relative position [0, 1] to rank scale [0, n_code]
                    # We clip to ensure it stays within reasonable bounds
                    pred_rel = max(0.0, min(1.0, pred_rel))
                    rank = pred_rel * n_code

                    cells_with_ranks.append((rank, cid))

            # Sort all cells by rank
            cells_with_ranks.sort(key=lambda x: x[0])

            # Extract IDs
            ordered_ids = [x[1] for x in cells_with_ranks]
            cell_order_str = " ".join(ordered_ids)

            submission_rows.append({"id": nb_id, "cell_order": cell_order_str})

        # 3. Save Submission
        df_submission = pd.DataFrame(submission_rows)
        df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
