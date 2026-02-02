import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from library.config import Config


class ObjectDetector:
    """
    Wrapper for the Two-Stage Object Detection Model (LightGBM).
    Stage 1: Classification (Object vs Background + Class Type)
    Stage 2: Regression (Bounding Box Refinement)
    """

    def __init__(self):
        self.classifier = None
        self.regressors = {}
        self.working_dir = Config.WORKING_DIR
        self.model_dir = os.path.join(self.working_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)

        self.cls_path = os.path.join(self.model_dir, "classifier.pkl")
        self.reg_path = os.path.join(self.model_dir, "regressors.pkl")

    def train(self, train_df, val_df):
        """
        Trains the classifier and regressors using the provided DataFrames.

        Args:
            train_df (pd.DataFrame): Training dataset with features and targets.
            val_df (pd.DataFrame): Validation dataset for early stopping.
        """
        print("Starting Model Training...")

        # ---------------------------------------------------------
        # 1. Train Classifier
        # ---------------------------------------------------------
        print("\n--- Training Classifier ---")

        # Filter out ambiguous samples (-1)
        train_cls_mask = train_df["target_class"] != -1
        val_cls_mask = val_df["target_class"] != -1

        X_train_cls = train_df.loc[train_cls_mask, Config.FEATURES]
        y_train_cls = train_df.loc[train_cls_mask, "target_class"]

        X_val_cls = val_df.loc[val_cls_mask, Config.FEATURES]
        y_val_cls = val_df.loc[val_cls_mask, "target_class"]

        self.classifier = lgb.LGBMClassifier(**Config.LGBM_CLS_PARAMS)

        # Callbacks for early stopping and logging
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=True
            ),
            lgb.log_evaluation(period=100),
        ]

        self.classifier.fit(
            X_train_cls,
            y_train_cls,
            eval_set=[(X_val_cls, y_val_cls)],
            eval_metric="multi_logloss",
            callbacks=callbacks,
        )

        # Print final validation score
        val_preds = self.classifier.predict_proba(X_val_cls)
        val_pred_cls = np.argmax(val_preds, axis=1)
        acc = np.mean(val_pred_cls == y_val_cls)
        print(f"Classifier Validation Accuracy: {acc}")

        # ---------------------------------------------------------
        # 2. Train Regressors
        # ---------------------------------------------------------
        print("\n--- Training Regressors ---")

        # Filter for Positive samples only (Class > 0)
        train_reg_mask = train_df["target_class"] > 0
        val_reg_mask = val_df["target_class"] > 0

        if train_reg_mask.sum() == 0:
            print(
                "Warning: No positive samples in training set. Skipping regression training."
            )
        else:
            X_train_reg = train_df.loc[train_reg_mask, Config.FEATURES]
            X_val_reg = val_df.loc[val_reg_mask, Config.FEATURES]

            for target in Config.REGRESSION_TARGETS:
                print(f"Training Regressor for target: {target}")

                y_train_reg = train_df.loc[train_reg_mask, target]
                y_val_reg = val_df.loc[val_reg_mask, target]

                reg = lgb.LGBMRegressor(**Config.LGBM_REG_PARAMS)

                callbacks_reg = [
                    lgb.early_stopping(
                        stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=True
                    ),
                    lgb.log_evaluation(period=100),
                ]

                reg.fit(
                    X_train_reg,
                    y_train_reg,
                    eval_set=[(X_val_reg, y_val_reg)],
                    eval_metric="l2",
                    callbacks=callbacks_reg,
                )

                self.regressors[target] = reg

        # Save Models
        self.save_models()

    def predict(self, test_df):
        """
        Runs inference on the test dataset.
        Returns a DataFrame containing refined bounding boxes and classes.

        Args:
            test_df (pd.DataFrame): Test dataset features.

        Returns:
            pd.DataFrame: DataFrame with prediction columns.
        """
        if test_df.empty:
            return pd.DataFrame()

        X_test = test_df[Config.FEATURES]

        # 1. Classification
        if self.classifier is None:
            self.load_models()

        probs = self.classifier.predict_proba(X_test)
        class_ids = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)

        # 2. Filtering
        # Keep if class is not background (0) and confidence is high enough
        mask = (class_ids > 0) & (confidences >= Config.CONFIDENCE_THRESHOLD)

        if not mask.any():
            return pd.DataFrame()

        survivors = test_df.loc[mask].copy()
        survivors["class_id"] = class_ids[mask]
        survivors["confidence"] = confidences[mask]
        X_survivors = X_test.loc[mask]

        # 3. Regression (Refinement)
        # Apply residuals
        for target in Config.REGRESSION_TARGETS:
            if target in self.regressors:
                model = self.regressors[target]
                pred_residual = model.predict(X_survivors)
                survivors[f"pred_{target}"] = pred_residual
            else:
                survivors[f"pred_{target}"] = 0.0

        # 4. Apply Corrections
        # Center: g = p + d
        survivors["final_x"] = survivors["prop_x"] + survivors["pred_dx"]
        survivors["final_y"] = survivors["prop_y"] + survivors["pred_dy"]
        survivors["final_z"] = survivors["prop_z"] + survivors["pred_dz"]

        # Dimensions (Log space): g = p * exp(d)
        survivors["final_w"] = survivors["prop_w"] * np.exp(survivors["pred_dw"])
        survivors["final_l"] = survivors["prop_l"] * np.exp(survivors["pred_dl"])
        survivors["final_h"] = survivors["prop_h"] * np.exp(survivors["pred_dh"])

        # Yaw: g = p + d
        survivors["final_yaw"] = survivors["prop_yaw"] + survivors["pred_dyaw"]

        # Map Class ID to Name
        survivors["class_name"] = survivors["class_id"].map(Config.ID_TO_CLASS)

        return survivors

    def generate_submission(self, test_df):
        """
        Generates the submission.csv file.

        Args:
            test_df (pd.DataFrame): Test dataset containing features and sample_tokens.
        """
        print("Generating Submission...")

        # Run inference
        predictions = self.predict(test_df)

        # Prepare submission dataframe structure
        # We need one row per sample_token in the test set.
        # Load test metadata to get all sample tokens (even those with no predictions)
        test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
        all_sample_tokens = test_meta["sample_token"].unique()

        submission_map = {token: "" for token in all_sample_tokens}

        if not predictions.empty:
            # Format prediction string: confidence x y z w l h yaw class_name
            # Ensure order matches format

            # Helper to format float
            def fmt(x):
                return f"{x:.4f}"

            # Vectorized string creation
            pred_strs = (
                predictions["confidence"].apply(fmt)
                + " "
                + predictions["final_x"].apply(fmt)
                + " "
                + predictions["final_y"].apply(fmt)
                + " "
                + predictions["final_z"].apply(fmt)
                + " "
                + predictions["final_w"].apply(fmt)
                + " "
                + predictions["final_l"].apply(fmt)
                + " "
                + predictions["final_h"].apply(fmt)
                + " "
                + predictions["final_yaw"].apply(fmt)
                + " "
                + predictions["class_name"]
            )

            predictions["pred_string"] = pred_strs

            # Group by sample_token and join with space
            grouped = predictions.groupby("sample_token")["pred_string"].apply(
                lambda x: " ".join(x)
            )

            # Update map
            for token, string in grouped.items():
                submission_map[token] = string

        # Create final DataFrame
        submission_df = pd.DataFrame(
            list(submission_map.items()), columns=["Id", "PredictionString"]
        )

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    def save_models(self):
        """Saves the trained models to disk."""
        joblib.dump(self.classifier, self.cls_path)
        joblib.dump(self.regressors, self.reg_path)
        print(f"Models saved to {self.model_dir}")

    def load_models(self):
        """Loads models from disk."""
        if os.path.exists(self.cls_path):
            self.classifier = joblib.load(self.cls_path)
        else:
            raise FileNotFoundError("Classifier model not found. Train first.")

        if os.path.exists(self.reg_path):
            self.regressors = joblib.load(self.reg_path)
        else:
            raise FileNotFoundError("Regressor models not found. Train first.")
        print("Models loaded successfully.")
