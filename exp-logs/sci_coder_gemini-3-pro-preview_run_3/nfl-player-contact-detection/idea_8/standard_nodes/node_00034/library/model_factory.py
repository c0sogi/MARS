import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef
from library.config import Config
from library.utils import set_seed


class DualStreamModel:
    """
    Manages the training, optimization, and inference of the Dual-Stream GBDT solution.
    Stream A: Player-Player Interaction
    Stream B: Player-Ground Interaction
    """

    def __init__(self):
        self.model_a = None
        self.model_b = None
        self.threshold_a = 0.5
        self.threshold_b = 0.5
        self.working_dir = Config.WORKING_DIR

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

    def fit(self, train_data, val_data):
        """
        Trains both Stream A and Stream B models, then optimizes thresholds based on validation MCC.

        Args:
            train_data (dict): Dictionary containing 'stream_a' and 'stream_b' training data.
                               Each value is a dict with 'X', 'y', 'ids'.
            val_data (dict): Dictionary containing 'stream_a' and 'stream_b' validation data.
        """
        set_seed(Config.SEED)

        print("\n=== Training Stream A (Player-Player) ===")
        self.model_a, val_probs_a = self._train_single_stream(
            train_data["stream_a"],
            val_data["stream_a"],
            Config.XGB_PARAMS_A,
            "Stream A",
        )

        print("\n=== Training Stream B (Player-Ground) ===")
        self.model_b, val_probs_b = self._train_single_stream(
            train_data["stream_b"],
            val_data["stream_b"],
            Config.XGB_PARAMS_B,
            "Stream B",
        )

        print("\n=== Optimizing Thresholds ===")
        # Optimize Stream A
        y_val_a = val_data["stream_a"]["y"]
        self.threshold_a, best_mcc_a = self._optimize_threshold(y_val_a, val_probs_a)
        print(
            f"Stream A Best Threshold: {self.threshold_a:.4f} | Val MCC: {best_mcc_a:.8f}"
        )

        # Optimize Stream B
        y_val_b = val_data["stream_b"]["y"]
        self.threshold_b, best_mcc_b = self._optimize_threshold(y_val_b, val_probs_b)
        print(
            f"Stream B Best Threshold: {self.threshold_b:.4f} | Val MCC: {best_mcc_b:.8f}"
        )

        # Save models and thresholds
        self.save_models()

    def predict(self, test_data):
        """
        Generates predictions for the test set using the trained models and optimized thresholds.

        Args:
            test_data (dict): Dictionary containing 'stream_a' and 'stream_b' test data.

        Returns:
            pd.DataFrame: Submission dataframe with 'contact_id' and 'contact'.
        """
        print("\n=== Running Inference ===")

        # Predict Stream A
        print("Predicting Stream A...")
        probs_a = self._predict_proba(self.model_a, test_data["stream_a"]["X"])
        preds_a = (probs_a >= self.threshold_a).astype(int)
        ids_a = test_data["stream_a"]["ids"]

        # Predict Stream B
        print("Predicting Stream B...")
        probs_b = self._predict_proba(self.model_b, test_data["stream_b"]["X"])
        preds_b = (probs_b >= self.threshold_b).astype(int)
        ids_b = test_data["stream_b"]["ids"]

        # Combine results
        df_a = pd.DataFrame({"contact_id": ids_a, "contact": preds_a})
        df_b = pd.DataFrame({"contact_id": ids_b, "contact": preds_b})

        submission = pd.concat([df_a, df_b], axis=0).reset_index(drop=True)

        # Ensure no duplicates (though streams should be disjoint)
        submission = submission.drop_duplicates(subset=["contact_id"])

        return submission

    def _train_single_stream(self, train_dict, val_dict, params, name):
        """
        Helper to train a single XGBoost model.
        """
        X_train, y_train = train_dict["X"], train_dict["y"]
        X_val, y_val = val_dict["X"], val_dict["y"]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Train
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params["n_estimators"],
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=100,
        )

        # Generate validation probabilities for threshold optimization
        # Use iteration_range to use best iteration
        best_iteration = model.best_iteration + 1
        val_probs = model.predict(dval, iteration_range=(0, best_iteration))

        return model, val_probs

    def _predict_proba(self, model, X):
        """
        Helper to predict probabilities. Handles empty inputs gracefully.
        """
        if model is None:
            raise ValueError("Model has not been trained or loaded.")

        if len(X) == 0:
            return np.array([])

        dtest = xgb.DMatrix(X)
        # Use best_iteration if available, else default
        try:
            limit = model.best_iteration + 1
            return model.predict(dtest, iteration_range=(0, limit))
        except AttributeError:
            return model.predict(dtest)

    def _optimize_threshold(self, y_true, y_proba):
        """
        Performs a linear search to find the probability threshold that maximizes MCC.
        """
        best_mcc = -1.0
        best_thresh = 0.5

        # Search range: 0.01 to 0.99
        thresholds = np.linspace(0.01, 0.99, 99)

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred)

            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = thresh

        return best_thresh, best_mcc

    def save_models(self):
        """Saves models and thresholds to disk."""
        path_a = os.path.join(self.working_dir, "model_a.json")
        path_b = os.path.join(self.working_dir, "model_b.json")
        path_meta = os.path.join(self.working_dir, "model_meta.json")

        if self.model_a:
            self.model_a.save_model(path_a)
        if self.model_b:
            self.model_b.save_model(path_b)

        meta = {"threshold_a": self.threshold_a, "threshold_b": self.threshold_b}
        with open(path_meta, "w") as f:
            json.dump(meta, f)

        print(f"Models saved to {self.working_dir}")

    def load_models(self):
        """Loads models and thresholds from disk."""
        path_a = os.path.join(self.working_dir, "model_a.json")
        path_b = os.path.join(self.working_dir, "model_b.json")
        path_meta = os.path.join(self.working_dir, "model_meta.json")

        if os.path.exists(path_a):
            self.model_a = xgb.Booster()
            self.model_a.load_model(path_a)

        if os.path.exists(path_b):
            self.model_b = xgb.Booster()
            self.model_b.load_model(path_b)

        if os.path.exists(path_meta):
            with open(path_meta, "r") as f:
                meta = json.load(f)
                self.threshold_a = meta.get("threshold_a", 0.5)
                self.threshold_b = meta.get("threshold_b", 0.5)

        print(f"Models loaded from {self.working_dir}")
