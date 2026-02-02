import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from library.config import Config
from library.data_processing import create_training_dataset
from library.model import ClusterClassifier


class Trainer:
    """
    Orchestrates the training of the Cluster-and-Classify model.
    """

    def __init__(self):
        self.train_meta_path = os.path.join(Config.METADATA_DIR, "train_metadata.csv")
        self.val_meta_path = os.path.join(Config.METADATA_DIR, "val_metadata.csv")
        self.model_save_path = os.path.join(Config.WORKING_DIR, "model.joblib")
        self.train_cache_path = os.path.join(
            Config.WORKING_DIR, "train_features.parquet"
        )
        self.val_cache_path = os.path.join(Config.WORKING_DIR, "val_features.parquet")

    def _load_or_generate_data(self, meta_path, target_cache_path, force_gen=False):
        """
        Helper to safely generate data using the library function which hardcodes output path.
        """
        # If we have the target cache and don't need to force gen, just load it
        if not force_gen and os.path.exists(target_cache_path):
            print(f"Loading cached data from {target_cache_path}...")
            return pd.read_parquet(target_cache_path)

        # The library function always writes to Config.WORKING_DIR/train_features.parquet
        # We need to manage this file to avoid conflicts between train and val generation.

        # 1. Back up existing train cache if it exists and is not the target we are currently generating
        temp_backup = self.train_cache_path + ".bak"
        train_cache_exists = os.path.exists(self.train_cache_path)

        # If we are generating validation data, we must move the train cache out of the way
        # because create_training_dataset will overwrite it.
        if train_cache_exists and target_cache_path != self.train_cache_path:
            os.rename(self.train_cache_path, temp_backup)

        try:
            # 2. Generate data (this creates/overwrites train_features.parquet)
            # We pass load_cached_data=False to force generation if we are in this block
            # effectively, unless we are generating train data and didn't back it up.
            print(f"Generating data from {meta_path}...")
            df = create_training_dataset(meta_path, load_cached_data=False)

            # 3. Rename the output to our desired target path
            if target_cache_path != self.train_cache_path:
                if os.path.exists(self.train_cache_path):
                    os.rename(self.train_cache_path, target_cache_path)

            return df

        finally:
            # 4. Restore backup if we moved it
            if os.path.exists(temp_backup):
                # If train_features.parquet exists now (and we aren't generating train),
                # it's a leftover or the result we just renamed.
                # If we renamed it successfully, this path shouldn't exist.
                # If it does exist, we delete it to restore the backup safely.
                if (
                    os.path.exists(self.train_cache_path)
                    and target_cache_path != self.train_cache_path
                ):
                    os.remove(self.train_cache_path)

                os.rename(temp_backup, self.train_cache_path)

    def train(self, load_cached_data=True):
        print("=== Starting Training Pipeline ===")

        # 1. Load Datasets
        # Load Train
        # We can use the library function directly for train if we want, but our helper handles logic
        if load_cached_data and os.path.exists(self.train_cache_path):
            print(f"Loading training data from {self.train_cache_path}...")
            df_train = pd.read_parquet(self.train_cache_path)
        else:
            df_train = create_training_dataset(
                self.train_meta_path, load_cached_data=False
            )

        # Load Val
        df_val = self._load_or_generate_data(
            self.val_meta_path, self.val_cache_path, force_gen=not load_cached_data
        )

        if df_train.empty:
            raise ValueError("Training dataset is empty.")
        if df_val.empty:
            raise ValueError("Validation dataset is empty.")

        # 2. Configure Model
        # Inject early stopping parameters into Config before initializing model
        # This works because XGBClassifier/Regressor accept **kwargs in __init__
        Config.XGB_CLF_PARAMS["early_stopping_rounds"] = 10
        Config.XGB_REG_PARAMS["early_stopping_rounds"] = 10

        model = ClusterClassifier()

        # 3. Fit Model
        print("Fitting model...")
        model.fit(df_train, df_val)

        # 4. Evaluate
        print("=== Validation Evaluation ===")
        # We use the internal helper to get the exact X and y used for training/validation
        X_val, y_cls_val, y_reg_val = model._prepare_data(df_val, is_training=True)

        # A. Classification Metrics
        # Note: y_cls_val contains the background class ID
        y_cls_pred_enc = model.clf.predict(X_val)
        y_cls_pred = model.le.inverse_transform(y_cls_pred_enc)

        acc = accuracy_score(y_cls_val, y_cls_pred)
        print(f"Validation Accuracy: {acc}")

        target_names = Config.CLASSES + ["Background"]
        # Ensure target names match the unique labels present if subset is used
        unique_labels = sorted(list(set(y_cls_val) | set(y_cls_pred)))
        # Filter target names to match indices
        active_target_names = [
            target_names[i] for i in unique_labels if i < len(target_names)
        ]

        print("Classification Report:")
        print(
            classification_report(
                y_cls_val,
                y_cls_pred,
                labels=unique_labels,
                target_names=active_target_names,
            )
        )

        # B. Regression Metrics
        # Evaluate only on objects (exclude background)
        mask_obj = y_cls_val != model.bg_class_id
        if np.sum(mask_obj) > 0:
            X_val_obj = X_val[mask_obj]
            y_reg_val_obj = y_reg_val[mask_obj]

            y_reg_pred = model.reg.predict(X_val_obj)

            # MSE
            mse = mean_squared_error(y_reg_val_obj, y_reg_pred)
            print(f"Validation MSE (Objects Only): {mse}")

            # RMSE for interpretability
            rmse = np.sqrt(mse)
            print(f"Validation RMSE (Objects Only): {rmse}")
        else:
            print("No object samples in validation set for regression evaluation.")

        # 5. Save Model
        print(f"Saving model to {self.model_save_path}...")
        model.save(self.model_save_path)
        print("Training complete.")
