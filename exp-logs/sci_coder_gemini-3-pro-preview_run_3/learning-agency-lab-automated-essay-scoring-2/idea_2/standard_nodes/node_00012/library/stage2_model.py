import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from library.config import Config
from library.utils import seed_everything, compute_qwk
from library.data import get_data


class LGBMHandler:
    """
    Handles the training and inference of the Stage 2 LightGBM model.
    Combines dense embeddings from Stage 1 with explicit meta-features.
    """

    def __init__(self):
        """
        Initialize the handler.
        Sets random seeds for reproducibility.
        """
        seed_everything()
        self.model = None

    def _load_dataset(
        self, embedding_path, meta_features_path, metadata_path, is_test=False
    ):
        """
        Loads and combines embeddings and meta-features.

        Args:
            embedding_path (str): Path to the .npy embedding file.
            meta_features_path (str): Path to the .parquet meta-features file.
            metadata_path (str): Path to the metadata CSV (for targets/ids).
            is_test (bool): Whether this is the test set (no targets).

        Returns:
            tuple: (X, y, ids)
                - X (np.ndarray): Combined feature matrix.
                - y (np.ndarray or None): Target scores.
                - ids (np.ndarray): Essay IDs.
        """
        # 1. Load Embeddings (Dense Vectors)
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embeddings not found at {embedding_path}")
        embeddings = np.load(embedding_path)

        # 2. Load Meta-Features (Structural Scalars)
        if not os.path.exists(meta_features_path):
            raise FileNotFoundError(f"Meta-features not found at {meta_features_path}")
        meta_df = pd.read_parquet(meta_features_path)

        # Convert to numpy
        meta_features = meta_df.values

        # 3. Load Metadata (Targets and IDs)
        # We use get_data to leverage its caching logic if applicable,
        # though here we mainly need the raw dataframe for alignment check
        df = get_data(metadata_path)

        # 4. Validation Checks
        # Ensure row counts match across all sources
        if not (len(embeddings) == len(meta_features) == len(df)):
            raise ValueError(
                f"Dimension mismatch! Embeddings: {len(embeddings)}, "
                f"Meta: {len(meta_features)}, DF: {len(df)}"
            )

        # 5. Concatenate Features
        # Horizontal stack: [Embedding Vector | Meta Features]
        X = np.hstack([embeddings, meta_features])

        # 6. Extract Targets and IDs
        ids = df["essay_id"].values
        y = None
        if not is_test:
            if "score" not in df.columns:
                raise KeyError(f"Column 'score' not found in {metadata_path}")
            y = df["score"].values

        return X, y, ids

    def train_model(self):
        """
        Trains the LightGBM model using features from Train and Validation sets.
        Evaluates performance using QWK.
        """
        print("Starting Stage 2: LightGBM Training...")

        # 1. Load Data
        print("Loading Training Data...")
        X_train, y_train, _ = self._load_dataset(
            Config.TRAIN_EMBEDDINGS_PATH,
            Config.TRAIN_META_FEATS_PATH,
            Config.TRAIN_DATA_PATH,
            is_test=False,
        )

        print("Loading Validation Data...")
        X_val, y_val, _ = self._load_dataset(
            Config.VAL_EMBEDDINGS_PATH,
            Config.VAL_META_FEATS_PATH,
            Config.VAL_DATA_PATH,
            is_test=False,
        )

        # 2. Initialize Model
        # We use the parameters defined in Config
        self.model = lgb.LGBMRegressor(**Config.LGBM_PARAMS)

        # 3. Train
        # Use early stopping to prevent overfitting
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=Config.LGBM_PARAMS["early_stopping_rounds"],
                verbose=False,
            ),
            lgb.log_evaluation(period=100),
        ]

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=callbacks,
        )

        # 4. Evaluate
        print("Evaluating on Validation Set...")
        val_preds = self.model.predict(X_val)

        # Calculate RMSE
        mse = np.mean((y_val - val_preds) ** 2)
        rmse = np.sqrt(mse)
        print(f"Validation RMSE: {rmse}")

        # Calculate QWK
        qwk = compute_qwk(y_val, val_preds)
        print(f"Validation QWK: {qwk}")

        # Save model text dump (optional, for debugging/persistence)
        model_save_path = os.path.join(Config.WORKING_DIR, "lgbm_model.txt")
        self.model.booster_.save_model(model_save_path)
        print(f"Model saved to {model_save_path}")

    def predict_and_submit(self):
        """
        Generates predictions for the Test set and creates the submission file.
        """
        print("Starting Stage 2: Inference and Submission...")

        if self.model is None:
            # Try to load if not in memory (e.g. if running separate steps)
            model_save_path = os.path.join(Config.WORKING_DIR, "lgbm_model.txt")
            if os.path.exists(model_save_path):
                print("Loading trained model from disk...")
                self.model = lgb.Booster(model_file=model_save_path)
            else:
                raise RuntimeError("Model not trained! Call train_model() first.")

        # 1. Load Test Data
        print("Loading Test Data...")
        X_test, _, test_ids = self._load_dataset(
            Config.TEST_EMBEDDINGS_PATH,
            Config.TEST_META_FEATS_PATH,
            Config.TEST_DATA_PATH,
            is_test=True,
        )

        # 2. Predict
        # Note: If loaded from Booster, use self.model.predict.
        # If LGBMRegressor, use self.model.predict. Both work similarly for simple arrays.
        raw_preds = self.model.predict(X_test)

        # 3. Post-processing
        # Clip to valid range [1, 6] and round to nearest integer
        final_preds = np.clip(np.round(raw_preds), 1, 6).astype(int)

        # 4. Create Submission DataFrame
        submission_df = pd.DataFrame({"essay_id": test_ids, "score": final_preds})

        # 5. Save
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

        print("Submission file created successfully.")
        print(submission_df.head())
