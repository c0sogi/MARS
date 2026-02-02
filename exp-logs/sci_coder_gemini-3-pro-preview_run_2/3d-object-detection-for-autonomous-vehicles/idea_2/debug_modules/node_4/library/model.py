import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, mean_squared_error

from library.config import Config


class ClusterClassifier:
    """
    A Cluster-and-Classify model for 3D Object Detection.
    Consists of:
    1. XGBClassifier: Classifies a cluster as one of the object classes or background.
    2. XGBRegressor: Regresses 3D bounding box parameters (refinements) for identified objects.
    """

    def __init__(self):
        # Feature columns expected in the input DataFrame
        self.feature_cols = [
            "point_count",
            "x_min",
            "y_min",
            "z_min",
            "x_max",
            "y_max",
            "z_max",
            "x_mean",
            "y_mean",
            "z_mean",
            "x_std",
            "y_std",
            "z_std",
            "cluster_width",
            "cluster_length",
            "cluster_height",
            "cluster_volume",
            "eig_1",
            "eig_2",
            "eig_3",
            "norm_eig_1",
            "norm_eig_2",
            "norm_eig_3",
            "density",
        ]

        # Background class index (last index after all object classes)
        self.bg_class_id = Config.NUM_CLASSES

        # Initialize models with config parameters
        self.clf = XGBClassifier(**Config.XGB_CLF_PARAMS)
        self.reg = XGBRegressor(**Config.XGB_REG_PARAMS)
        self.le = LabelEncoder()

    def _prepare_data(self, df, is_training=True):
        """
        Extracts features (X) and targets (y_class, y_reg) from DataFrame.
        """
        # 1. Extract Features
        # Ensure all columns exist, fill missing with 0
        X = df[self.feature_cols].fillna(0.0).values

        if not is_training:
            return X, None, None

        # 2. Extract Classification Targets
        # Map -1 (Background) to self.bg_class_id
        y_raw = df["target_class"].values.astype(int)
        y_class = np.where(y_raw == -1, self.bg_class_id, y_raw)

        # 3. Extract Regression Targets
        # We predict offsets relative to cluster geometric properties
        # Target: [dx, dy, dz, width, length, height, yaw]

        # Center offsets: Target Center - Cluster Mean
        dx = df["target_center_x"] - df["x_mean"]
        dy = df["target_center_y"] - df["y_mean"]
        dz = df["target_center_z"] - df["z_mean"]

        # Dimensions and Yaw are predicted directly
        # (Could also regress log(dim) - log(cluster_dim), but direct is simpler for baseline)
        w = df["target_width"]
        l = df["target_length"]
        h = df["target_height"]
        yaw = df["target_yaw"]

        y_reg = np.column_stack([dx, dy, dz, w, l, h, yaw])

        return X, y_class, y_reg

    def fit(self, df_train, df_val=None):
        """
        Train the classifier and regressor.
        """
        print("Preparing training data...")
        X_train, y_clf_train, y_reg_train = self._prepare_data(
            df_train, is_training=True
        )

        eval_set_clf = [(X_train, y_clf_train)]
        eval_set_reg = None

        X_val, y_clf_val, y_reg_val = None, None, None

        if df_val is not None:
            print("Preparing validation data...")
            X_val, y_clf_val, y_reg_val = self._prepare_data(df_val, is_training=True)
            eval_set_clf.append((X_val, y_clf_val))

        # ==========================================
        # 1. Train Classifier
        # ==========================================
        print(f"Training Classifier on {len(X_train)} samples...")
        # Note: XGBoost 2.0+ uses 'early_stopping_rounds' in constructor or fit.
        # Config params are passed in __init__.
        self.clf.fit(X_train, y_clf_train, eval_set=eval_set_clf, verbose=False)

        # Log Metrics
        train_loss = log_loss(y_clf_train, self.clf.predict_proba(X_train))
        print(f"Classifier Train LogLoss: {train_loss}")
        if X_val is not None:
            val_loss = log_loss(y_clf_val, self.clf.predict_proba(X_val))
            print(f"Classifier Val LogLoss: {val_loss}")

        # ==========================================
        # 2. Train Regressor
        # ==========================================
        # Filter only positive samples (objects) for regression
        mask_train = y_clf_train != self.bg_class_id
        X_reg_train = X_train[mask_train]
        y_reg_train = y_reg_train[mask_train]

        if len(X_reg_train) > 0:
            print(f"Training Regressor on {len(X_reg_train)} positive samples...")

            eval_set_reg = [(X_reg_train, y_reg_train)]

            if X_val is not None:
                mask_val = y_clf_val != self.bg_class_id
                X_reg_val = X_val[mask_val]
                y_reg_val = y_reg_val[mask_val]
                if len(X_reg_val) > 0:
                    eval_set_reg.append((X_reg_val, y_reg_val))

            self.reg.fit(X_reg_train, y_reg_train, eval_set=eval_set_reg, verbose=False)

            # Log Metrics
            train_preds = self.reg.predict(X_reg_train)
            train_rmse = np.sqrt(mean_squared_error(y_reg_train, train_preds))
            print(f"Regressor Train RMSE: {train_rmse}")

            if X_val is not None and len(X_reg_val) > 0:
                val_preds = self.reg.predict(X_reg_val)
                val_rmse = np.sqrt(mean_squared_error(y_reg_val, val_preds))
                print(f"Regressor Val RMSE: {val_rmse}")
        else:
            print(
                "Warning: No positive samples found in training data. Regressor not trained."
            )

    def predict(self, df):
        """
        Predict bounding boxes for the given clusters.
        Returns a list of dictionaries.
        """
        if df.empty:
            return []

        X, _, _ = self._prepare_data(df, is_training=False)

        # 1. Class Probabilities
        probs = self.clf.predict_proba(X)
        pred_labels_local = np.argmax(probs, axis=1)
        pred_labels = self.le.inverse_transform(pred_labels_local)
        max_probs = np.max(probs, axis=1)

        # 2. Regression Predictions (for all samples)
        # Shape: (N, 7) -> [dx, dy, dz, w, l, h, yaw]
        reg_preds = self.reg.predict(X)

        results = []

        # 3. Construct Final Boxes
        for i in range(len(df)):
            label_id = pred_labels[i]
            confidence = max_probs[i]

            # Filter Background and Low Confidence
            if label_id == self.bg_class_id:
                continue
            if confidence < Config.CONF_THRESHOLD:
                continue

            # Reconstruct Absolute Coordinates
            # Center = Cluster Mean + Predicted Offset
            c_x = df.iloc[i]["x_mean"] + reg_preds[i, 0]
            c_y = df.iloc[i]["y_mean"] + reg_preds[i, 1]
            c_z = df.iloc[i]["z_mean"] + reg_preds[i, 2]

            # Dimensions
            w = max(0.1, reg_preds[i, 3])  # Ensure positive
            l = max(0.1, reg_preds[i, 4])
            h = max(0.1, reg_preds[i, 5])

            yaw = reg_preds[i, 6]

            class_name = Config.ID_TO_CLASS[label_id]

            results.append(
                {
                    "sample_token": df.iloc[i]["sample_token"],
                    "confidence": float(confidence),
                    "center_x": float(c_x),
                    "center_y": float(c_y),
                    "center_z": float(c_z),
                    "width": float(w),
                    "length": float(l),
                    "height": float(h),
                    "yaw": float(yaw),
                    "class_name": class_name,
                }
            )

        return results

    def save(self, path):
        """Save the entire model object."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Model saved to {path}")

    @staticmethod
    def load(path):
        """Load the model object."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)
