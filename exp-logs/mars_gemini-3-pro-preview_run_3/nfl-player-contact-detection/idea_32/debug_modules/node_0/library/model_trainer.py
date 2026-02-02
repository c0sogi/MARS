import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from library.config import Config
from library.utils import get_config_hash


class ModelTrainer:
    """
    Handles the training and persistence of the Scale-Aligned Dual-Stream GBDT models.
    Implements Targeted Majority Undersampling and Stream-specific hyperparameter configuration.
    """

    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.config_hash = get_config_hash()

    def train(self, train_data, val_data, force_retrain=False):
        """
        Trains the Stream A and Stream B models.

        Args:
            train_data (dict): Dictionary containing 'stream_a' and 'stream_b' training data.
                               Each value is a dict with 'X', 'y', 'ids'.
            val_data (dict): Dictionary containing 'stream_a' and 'stream_b' validation data.
            force_retrain (bool): If True, ignores cached models and retrains.

        Returns:
            dict: Dictionary containing the trained XGBClassifier objects.
        """
        models = {}

        for stream_name in ["stream_a", "stream_b"]:
            model_path = self._get_model_path(stream_name)

            # Check cache
            if not force_retrain and os.path.exists(model_path):
                print(f"[{stream_name.upper()}] Loading model from cache: {model_path}")
                models[stream_name] = joblib.load(model_path)
                continue

            print(f"[{stream_name.upper()}] Training model...")

            # 1. Prepare Data
            X_train = train_data[stream_name]["X"]
            y_train = train_data[stream_name]["y"]
            X_val = val_data[stream_name]["X"]
            y_val = val_data[stream_name]["y"]

            if len(X_train) == 0:
                print(f"[{stream_name.upper()}] Warning: No training data available.")
                models[stream_name] = None
                continue

            # 2. Apply Targeted Majority Undersampling
            X_res, y_res = self._undersample(
                X_train, y_train, ratio=Config.UNDERSAMPLE_RATIO
            )
            print(
                f"[{stream_name.upper()}] Undersampled Train Shape: {X_res.shape}, Positive Rate: {y_res.mean():.4f}"
            )

            # 3. Configure Model
            if stream_name == "stream_a":
                params = Config.XGB_PARAMS_STREAM_A.copy()
            else:
                params = Config.XGB_PARAMS_STREAM_B.copy()

            # Extract fitting params that shouldn't be in __init__ for some sklearn wrapper versions,
            # but usually XGBClassifier handles them in kwargs.
            # We will pass them to fit() explicitly if they are specific to the training process.
            # However, Config defines them in the dict. We'll pass the dict to __init__.

            model = xgb.XGBClassifier(**params)

            # 4. Train with Early Stopping
            # Note: eval_set is used for early stopping
            model.fit(X_res, y_res, eval_set=[(X_val, y_val)], verbose=False)

            # 5. Log Metrics
            best_score = model.best_score
            print(
                f"[{stream_name.upper()}] Training Complete. Best Iteration: {model.best_iteration}"
            )
            print(f"[{stream_name.upper()}] Validation LogLoss: {best_score}")

            # 6. Save Model
            print(f"[{stream_name.upper()}] Saving model to {model_path}")
            joblib.dump(model, model_path)
            models[stream_name] = model

        return models

    def predict(self, models, test_data):
        """
        Generates predictions for the test set using the dual-stream architecture.

        Args:
            models (dict): Dictionary of trained models.
            test_data (dict): Dictionary containing 'stream_a' and 'stream_b' test data.

        Returns:
            pd.DataFrame: DataFrame with 'contact_id' and 'score' (probability).
        """
        results = []

        for stream_name in ["stream_a", "stream_b"]:
            model = models.get(stream_name)
            data = test_data.get(stream_name)

            if model is None or data is None or len(data["X"]) == 0:
                continue

            X_test = data["X"]
            ids = data["ids"]

            # Predict probabilities (class 1)
            probs = model.predict_proba(X_test)[:, 1]

            # Create DataFrame
            df_pred = pd.DataFrame({"contact_id": ids, "score": probs})
            results.append(df_pred)

        if not results:
            return pd.DataFrame(columns=["contact_id", "score"])

        # Combine results from both streams
        final_df = pd.concat(results, axis=0, ignore_index=True)
        return final_df

    def _undersample(self, X, y, ratio=10.0):
        """
        Performs random undersampling of the majority class (0) to achieve
        the specified negative:positive ratio.
        """
        # Ensure inputs are consistent
        if isinstance(X, pd.DataFrame):
            X = X.reset_index(drop=True)

        # Identify indices
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]

        n_pos = len(pos_indices)
        n_neg = len(neg_indices)

        # Calculate target number of negatives
        n_neg_keep = int(n_pos * ratio)

        # If we have fewer negatives than the target, keep all negatives
        if n_neg_keep >= n_neg:
            return X, y

        # Randomly sample negatives
        np.random.seed(Config.SEED)
        neg_indices_sampled = np.random.choice(
            neg_indices, size=n_neg_keep, replace=False
        )

        # Combine indices
        final_indices = np.concatenate([pos_indices, neg_indices_sampled])
        np.random.shuffle(final_indices)

        # Subset data
        if isinstance(X, pd.DataFrame):
            X_res = X.iloc[final_indices].copy()
        else:
            X_res = X[final_indices]

        y_res = y[final_indices]

        return X_res, y_res

    def _get_model_path(self, stream_name):
        """
        Generates the file path for saving/loading the model based on config hash.
        """
        filename = f"model_{stream_name}_{self.config_hash}.joblib"
        return os.path.join(self.cache_dir, filename)
