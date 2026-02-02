import pandas as pd
import numpy as np
import os
import logging
import gc
from library import config, utils, data_loader, model_factory


class InferencePipeline:
    """
    Manages the inference pipeline for the Reference-Anchored Decoupled-Mining Ensemble (RAD-ME).

    Responsibilities:
    1. Load trained Expert models (LGBM, XGB, CatBoost/HGB).
    2. Optimize decision threshold on Validation set (or load cached).
    3. Generate features for Test set.
    4. execute Ensemble Prediction.
    5. Generate valid submission file.
    """

    def __init__(self):
        self.loader = data_loader.NFLDataLoader()
        self.factory = model_factory.ModelFactory()
        self.feature_cols = config.FEATURE_COLUMNS
        self.target_col = "contact"
        self.models_dir = os.path.join(config.WORKING_DIR, "models")
        self.experts = {}

        # Setup logging
        utils.setup_logging(os.path.join(config.WORKING_DIR, "inference.log"))

    def load_models(self):
        """
        Loads the three trained expert models from disk.
        """
        logging.info("Loading Expert models...")
        model_types = ["lgbm", "xgb", "cat"]

        for m_type in model_types:
            path = os.path.join(self.models_dir, f"expert_{m_type}.joblib")
            if os.path.exists(path):
                self.experts[m_type] = utils.load_model(path)
            else:
                logging.warning(f"Model file not found: {path}. Skipping {m_type}.")

        if not self.experts:
            raise RuntimeError(
                "No expert models were loaded. Cannot proceed with inference."
            )

    def optimize_threshold(self, load_cached_data=True, force_recalc=False):
        """
        Determines the optimal decision threshold based on MCC.
        First checks for a cached threshold file. If missing or force_recalc is True,
        loads validation data and computes the threshold.

        Args:
            load_cached_data (bool): Whether to use cached validation features.
            force_recalc (bool): If True, ignores cached threshold file and recalculates.

        Returns:
            float: The optimal threshold.
        """
        threshold_path = os.path.join(self.models_dir, "best_threshold.npy")

        if not force_recalc and os.path.exists(threshold_path):
            best_th = np.load(threshold_path)[0]
            logging.info(f"Loaded cached best threshold: {best_th}")
            return best_th

        logging.info("Optimizing threshold on validation set...")

        # Load Validation Data
        df_val = self.loader.prepare_dataset(
            split="val", load_cached_data=load_cached_data
        )

        if df_val.empty:
            logging.warning("Validation set is empty. Defaulting threshold to 0.5.")
            return 0.5

        X_val = df_val[self.feature_cols]
        y_val = df_val[self.target_col].values

        # Get Ensemble Probabilities
        ensemble_probs = self.ensemble_predict(X_val)

        # Grid Search for best MCC
        thresholds = np.arange(0.1, 0.91, 0.01)
        best_mcc = -1.0
        best_th = 0.5

        for th in thresholds:
            preds = (ensemble_probs >= th).astype(int)
            mcc = utils.calc_mcc(y_val, preds)
            if mcc > best_mcc:
                best_mcc = mcc
                best_th = th

        logging.info(f"Threshold Optimization Complete.")
        logging.info(f"Best Validation MCC: {best_mcc}")
        logging.info(f"Optimal Threshold: {best_th}")

        # Cache the result
        np.save(threshold_path, np.array([best_th]))

        # Clean up
        del df_val, X_val, y_val, ensemble_probs
        gc.collect()

        return best_th

    def generate_test_features(self, load_cached_data=True):
        """
        Generates or loads features for the test set using the data loader.

        Args:
            load_cached_data (bool): Whether to use cached parquet files.

        Returns:
            pd.DataFrame: Processed test dataframe with features.
        """
        logging.info("Generating Test features...")
        return self.loader.prepare_dataset(
            split="test", load_cached_data=load_cached_data
        )

    def ensemble_predict(self, X):
        """
        Generates averaged probabilities from all loaded expert models.

        Args:
            X (pd.DataFrame): Feature matrix.

        Returns:
            np.ndarray: Averaged probabilities of class 1.
        """
        if not self.experts:
            self.load_models()

        ensemble_probs = np.zeros(len(X))

        for m_type, model in self.experts.items():
            probs = self.factory.predict_proba(model, X)
            ensemble_probs += probs

        ensemble_probs /= len(self.experts)
        return ensemble_probs

    def create_submission(self, df_test, probs, threshold):
        """
        Creates the submission file by merging predictions with the sample submission.
        Handles rows that were filtered out during gating (assigns them 0).

        Args:
            df_test (pd.DataFrame): Test dataframe containing 'contact_id'.
            probs (np.ndarray): Predicted probabilities for df_test.
            threshold (float): Decision threshold.
        """
        logging.info("Creating submission file...")

        # Apply threshold
        predictions = (probs >= threshold).astype(int)

        # Create DataFrame from predictions
        pred_df = pd.DataFrame(
            {"contact_id": df_test["contact_id"].values, "contact": predictions}
        )

        # Load Sample Submission to ensure all IDs are covered
        sample_sub_path = os.path.join(config.INPUT_DIR, "sample_submission.csv")
        if not os.path.exists(sample_sub_path):
            raise FileNotFoundError(f"Sample submission not found at {sample_sub_path}")

        sample_sub = pd.read_csv(sample_sub_path)

        # Merge: Left join sample_sub with predictions
        # Rows missing in pred_df (filtered by gating) will be NaN
        final_sub = sample_sub[["contact_id"]].merge(
            pred_df, on="contact_id", how="left"
        )

        # Fill NaNs with 0 (No Contact)
        final_sub["contact"] = final_sub["contact"].fillna(0).astype(int)

        # Save
        save_path = config.SUBMISSION_PATH
        final_sub.to_csv(save_path, index=False)
        logging.info(f"Submission saved to {save_path}. Total Rows: {len(final_sub)}")

    def run(self, load_cached_data=True):
        """
        Executes the full inference pipeline.

        Args:
            load_cached_data (bool): Whether to use cached data for features and threshold.
        """
        utils.seed_everything(config.SEED)

        # 1. Load Models
        self.load_models()

        # 2. Optimize/Load Threshold
        threshold = self.optimize_threshold(load_cached_data=load_cached_data)

        # 3. Generate Test Features
        df_test = self.generate_test_features(load_cached_data=load_cached_data)

        if df_test.empty:
            logging.warning(
                "Test dataset is empty after processing. Generating all-zero submission."
            )
            # Create a dummy dataframe with just contact_id from sample_submission to ensure valid file
            sample_sub = pd.read_csv(
                os.path.join(config.INPUT_DIR, "sample_submission.csv")
            )
            sample_sub["contact"] = 0
            sample_sub.to_csv(config.SUBMISSION_PATH, index=False)
            return

        X_test = df_test[self.feature_cols]

        # 4. Predict
        probs = self.ensemble_predict(X_test)

        # 5. Create Submission
        self.create_submission(df_test, probs, threshold)

        logging.info("Inference pipeline completed successfully.")
