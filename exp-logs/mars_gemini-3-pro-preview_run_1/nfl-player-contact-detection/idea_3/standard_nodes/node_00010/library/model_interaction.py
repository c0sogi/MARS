import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import matthews_corrcoef, log_loss
from library.config import Config
from library.utils import print_metric, seed_everything


class InteractionGBM:
    """
    Stream A: Interaction Model.
    Wraps a LightGBM model configured for high-capacity learning on flattened
    temporal kinematic features to detect player-player contacts.
    """

    def __init__(self, config=Config):
        self.config = config
        self.model = None
        self.model_path = os.path.join(
            self.config.WORKING_DIR, "lgbm_interaction_model.txt"
        )
        seed_everything(self.config.SEED)

    def train(self, train_df, val_df, feature_cols):
        """
        Trains the LightGBM model using the provided training and validation DataFrames.

        Args:
            train_df (pd.DataFrame): Training data containing features and 'contact' target.
            val_df (pd.DataFrame): Validation data containing features and 'contact' target.
            feature_cols (list): List of column names to use as features.
        """
        print(
            f"Initializing InteractionGBM training with {len(feature_cols)} features..."
        )

        # Prepare Features and Targets
        X_train = train_df[feature_cols]
        y_train = train_df["contact"]
        X_val = val_df[feature_cols]
        y_val = val_df["contact"]

        # Create LightGBM Datasets
        # free_raw_data=False ensures we can use the data for metric calc later if needed,
        # though usually we pass explicit valid sets.
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # Retrieve parameters from Config
        params = self.config.LGBM_PARAMS.copy()
        train_params = self.config.LGBM_TRAIN_PARAMS.copy()

        # Setup callbacks for logging and early stopping
        callbacks = [
            lgb.log_evaluation(period=train_params.get("verbose_eval", 100)),
            lgb.early_stopping(
                stopping_rounds=train_params.get("early_stopping_rounds", 50)
            ),
        ]

        print("Starting LightGBM training...")
        self.model = lgb.train(
            params,
            dtrain,
            num_boost_round=train_params.get("num_boost_round", 1000),
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Save the model
        print(f"Saving model to {self.model_path}...")
        self.model.save_model(self.model_path)

        # Evaluation on Validation Set
        print("Calculating detailed validation metrics...")
        # Predict using best iteration
        val_preds_prob = self.model.predict(
            X_val, num_iteration=self.model.best_iteration
        )

        # Calculate Log Loss
        val_log_loss = log_loss(y_val, val_preds_prob)
        print_metric("Validation LogLoss", val_log_loss)

        # Calculate MCC (Threshold optimization could be done, but we use 0.5 or a heuristic)
        # Here we use 0.5 as a standard baseline, or we could find optimal.
        # Given the prompt asks for MCC, we'll stick to standard threshold for reporting.
        val_preds_binary = (val_preds_prob > 0.5).astype(int)
        val_mcc = matthews_corrcoef(y_val, val_preds_binary)
        print_metric("Validation MCC (Thresh=0.5)", val_mcc)

        # Feature Importance Analysis
        importance = self.model.feature_importance(importance_type="gain")
        feature_imp = pd.DataFrame(
            sorted(zip(importance, feature_cols)), columns=["Value", "Feature"]
        )
        print("\nTop 10 Features by Gain:")
        print(
            feature_imp.sort_values(by="Value", ascending=False)
            .head(10)
            .to_string(index=False)
        )

    def predict(self, df, feature_cols):
        """
        Generates predictions for the given DataFrame.

        Args:
            df (pd.DataFrame): Data containing features.
            feature_cols (list): List of feature column names.

        Returns:
            np.ndarray: Predicted probabilities of contact.
        """
        # Load model if not in memory
        if self.model is None:
            if os.path.exists(self.model_path):
                print(f"Loading model from {self.model_path}...")
                self.model = lgb.Booster(model_file=self.model_path)
            else:
                raise ValueError("Model has not been trained and no saved model found.")

        # Predict
        # Note: LightGBM handles missing columns gracefully usually, but we assume schema matches
        X = df[feature_cols]
        preds = self.model.predict(X, num_iteration=self.model.best_iteration)

        return preds
