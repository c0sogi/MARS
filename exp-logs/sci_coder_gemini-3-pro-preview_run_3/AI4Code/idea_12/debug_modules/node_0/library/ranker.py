import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from library.config import Config
from library.feature_extraction import FeatureExtractor
from library.data_utils import get_notebook_cells, get_metadata


class LGBMRanker:
    """
    Trains a LightGBM regressor to predict the relative position of markdown cells
    and generates the final submission file.
    """

    def __init__(self):
        self.params = Config.Training.LGBM_PARAMS
        self.working_dir = Config.Paths.WORKING_DIR
        self.model_path = os.path.join(self.working_dir, "lgbm_model.txt")
        self.submission_path = Config.Paths.SUBMISSION_PATH
        self.feature_extractor = FeatureExtractor()

    def _prepare_data(self, df, is_training=True):
        """
        Separates features (X) and target (y) from the DataFrame.
        Drops metadata columns.
        """
        # Identify feature columns
        # Exclude metadata and target
        exclude_cols = ["notebook_id", "markdown_id", "target"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols]
        y = df["target"] if is_training and "target" in df.columns else None

        return X, y, feature_cols

    def train(self):
        """
        Loads features, trains the LightGBM model, and saves it.
        """
        print("Starting LightGBM training pipeline...")

        # 1. Load Features
        # FeatureExtractor handles caching internally
        print("Loading training features...")
        df_train = self.feature_extractor.extract_features(
            "train", load_cached_data=True
        )

        print("Loading validation features...")
        df_val = self.feature_extractor.extract_features("val", load_cached_data=True)

        if df_train.empty or df_val.empty:
            raise ValueError("Training or Validation features are empty.")

        # 2. Sampling (if configured)
        n_samples = Config.Training.NUM_NOTEBOOKS_LGBM
        if n_samples is not None:
            print(f"Sampling {n_samples} notebooks for training...")
            unique_ids = df_train["notebook_id"].unique()
            if len(unique_ids) > n_samples:
                # Set seed for reproducible sampling
                np.random.seed(Config.SEED)
                selected_ids = np.random.choice(unique_ids, n_samples, replace=False)
                df_train = df_train[df_train["notebook_id"].isin(selected_ids)].copy()

        # 3. Prepare Datasets
        X_train, y_train, feat_names = self._prepare_data(df_train, is_training=True)
        X_val, y_val, _ = self._prepare_data(df_val, is_training=True)

        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        # 4. Train
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=Config.Training.LGBM_EARLY_STOPPING_ROUNDS
            ),
            lgb.log_evaluation(period=100),
        ]

        print("Training LightGBM model...")
        model = lgb.train(
            self.params,
            train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # 5. Save Model
        print(f"Saving model to {self.model_path}")
        model.save_model(self.model_path)

        # 6. Validation Metrics
        # Predict on validation set to get final MSE
        val_preds = model.predict(X_val)
        mse = mean_squared_error(y_val, val_preds)
        print(f"Final Validation MSE: {mse}")

        return model

    def predict(self, df_test):
        """
        Loads the model and generates predictions for the test set.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Train first."
            )

        print(f"Loading model from {self.model_path}")
        model = lgb.Booster(model_file=self.model_path)

        X_test, _, _ = self._prepare_data(df_test, is_training=False)

        print(f"Predicting on {len(X_test)} samples...")
        predictions = model.predict(X_test)

        return predictions

    def generate_submission(self):
        """
        Orchestrates the prediction and submission generation process.
        """
        # 1. Load Test Features
        print("Loading test features...")
        df_test_features = self.feature_extractor.extract_features(
            "test", load_cached_data=True
        )

        if df_test_features.empty:
            print("Warning: Test features empty.")
            # We continue to generate a submission file even if features are empty
            # to ensure the file exists, though it might just contain code cells.

        # 2. Generate Predictions (if features exist)
        if not df_test_features.empty:
            preds = self.predict(df_test_features)
            df_test_features["pred_rank"] = preds
        else:
            df_test_features = pd.DataFrame(
                columns=["notebook_id", "markdown_id", "pred_rank"]
            )

        # 3. Reconstruct Order
        print("Reconstructing cell orders...")

        # Load test metadata to get file paths
        df_test_meta = get_metadata("test")
        # Create a map for quick lookup
        id_to_path = pd.Series(
            df_test_meta.file_path.values, index=df_test_meta.id
        ).to_dict()

        submission_rows = []

        # Group predictions by notebook
        if not df_test_features.empty:
            grouped = df_test_features.groupby("notebook_id")
        else:
            grouped = None

        # Iterate over all test notebooks (ensure we cover those with no markdown too)
        test_ids = df_test_meta["id"].unique()

        for nb_id in test_ids:
            rel_path = id_to_path.get(nb_id)
            if not rel_path:
                continue

            # Get notebook structure
            try:
                nb_data = get_notebook_cells(nb_id, rel_path)
            except Exception:
                # Fallback if file read fails
                submission_rows.append({"id": nb_id, "cell_order": ""})
                continue

            code_cells = nb_data["code_cells"]
            markdown_cells = nb_data["markdown_cells"]

            code_ids = [c["id"] for c in code_cells]

            # If no markdown cells, order is just code cells
            if not markdown_cells:
                submission_rows.append({"id": nb_id, "cell_order": " ".join(code_ids)})
                continue

            # If no code cells, order is arbitrary for markdown (or based on prediction if we had a model for that)
            # Our model assumes code context. If n_code=0, features are 0.
            if not code_cells:
                md_ids = [m["id"] for m in markdown_cells]
                submission_rows.append({"id": nb_id, "cell_order": " ".join(md_ids)})
                continue

            # Get predictions for this notebook
            md_pred_map = {}
            if grouped is not None and nb_id in grouped.groups:
                nb_preds = grouped.get_group(nb_id)
                # Map markdown_id to prediction
                md_pred_map = dict(zip(nb_preds["markdown_id"], nb_preds["pred_rank"]))

            # Assign scores
            # Code cells: index + 0.5 (places them at 0.5, 1.5, 2.5...)
            cells_with_scores = []
            for i, cid in enumerate(code_ids):
                cells_with_scores.append((cid, i + 0.5))

            # Markdown cells: prediction * n_code
            # Prediction is in [0, 1], so this maps to [0, n_code]
            n_code = len(code_ids)
            for md in markdown_cells:
                mid = md["id"]
                pred = md_pred_map.get(mid, 0.0)  # Default to 0 if missing
                score = pred * n_code
                cells_with_scores.append((mid, score))

            # Sort by score
            cells_with_scores.sort(key=lambda x: x[1])

            # Extract IDs
            final_order = [x[0] for x in cells_with_scores]
            submission_rows.append({"id": nb_id, "cell_order": " ".join(final_order)})

        # 4. Save Submission
        df_submission = pd.DataFrame(submission_rows)
        print(f"Saving submission to {self.submission_path}")
        df_submission.to_csv(self.submission_path, index=False)

    def run(self):
        """
        Main execution entry point.
        """
        # Train
        self.train()

        # Generate Submission
        self.generate_submission()
