import pandas as pd
import numpy as np
import os
import logging
import gc
from library import config, utils, data_loader, model_factory


class CurriculumTrainer:
    """
    Orchestrates the Reference-Anchored Decoupled-Mining Ensemble (RAD-ME) training pipeline.

    Phases:
    1. Scout Training: Train diverse models on a balanced subset.
    2. Diversity Mining: Use Scouts to identify Hard Negatives in the full training set.
    3. Expert Training: Train final ensemble on Positives + Hard Negatives + Anchors.
    4. Evaluation: Optimize decision threshold on Validation set.
    5. Inference: Generate submission for Test set.
    """

    def __init__(self):
        self.loader = data_loader.NFLDataLoader()
        self.factory = model_factory.ModelFactory()
        self.feature_cols = config.FEATURE_COLUMNS
        self.target_col = "contact"

        # Model storage
        self.scouts = {}
        self.experts = {}
        self.best_threshold = 0.5

        # Paths for saving intermediate artifacts
        self.models_dir = os.path.join(config.WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def train_scouts(self, df_train, df_val):
        """
        Phase 1: Train Scout models on a balanced dataset.
        """
        logging.info(">>> PHASE 1: Training Scouts")

        # Prepare Balanced Training Data
        X_train_scout, y_train_scout = self.loader.get_scout_dataset(df_train)

        # Prepare Validation Data (Full Gated Validation Set)
        X_val = df_val[self.feature_cols]
        y_val = df_val[self.target_col]

        model_types = ["lgbm", "xgb", "cat"]

        for m_type in model_types:
            logging.info(f"Training Scout: {m_type.upper()}")
            model = self.factory.train_model(
                model_type=m_type,
                X_train=X_train_scout,
                y_train=y_train_scout,
                X_val=X_val,
                y_val=y_val,
            )
            self.scouts[m_type] = model

            # Save model
            utils.save_model(
                model, os.path.join(self.models_dir, f"scout_{m_type}.joblib")
            )

        logging.info("Scout training complete.")

    def mine_hard_negatives(self, df_train):
        """
        Phase 2: Diversity Mining.
        Run Scouts on the full gated training set to find False Positives (Hard Negatives).
        """
        logging.info(">>> PHASE 2: Mining Hard Negatives")

        X_full = df_train[self.feature_cols]
        y_full = df_train[self.target_col]

        # Only look at actual negatives
        neg_indices = df_train[y_full == 0].index
        X_neg = X_full.loc[neg_indices]

        if X_neg.empty:
            logging.warning("No negatives found in training set. Skipping mining.")
            return []

        # Accumulate predictions (Union of Scouts)
        # If ANY scout thinks it's contact (prob > threshold), it's a hard negative.
        is_hard = np.zeros(len(X_neg), dtype=bool)

        for m_type, model in self.scouts.items():
            probs = self.factory.predict_proba(model, X_neg)
            # Check against mining threshold
            mask = probs > config.SCOUT_THRESHOLD
            is_hard = is_hard | mask
            logging.info(
                f"Scout {m_type} flagged {np.sum(mask)} potential hard negatives."
            )

        hard_negative_indices = neg_indices[is_hard].tolist()
        logging.info(f"Total Unique Hard Negatives Mined: {len(hard_negative_indices)}")

        return hard_negative_indices

    def train_experts(self, df_train, hard_negative_indices, df_val):
        """
        Phase 3: Train Expert models on the Anchored dataset.
        """
        logging.info(">>> PHASE 3: Training Experts")

        # Prepare Expert Dataset (Positives + Hard Negatives + Anchors)
        X_train_expert, y_train_expert = self.loader.get_expert_dataset(
            df_train, hard_negative_indices
        )

        # Prepare Validation Data
        X_val = df_val[self.feature_cols]
        y_val = df_val[self.target_col]

        model_types = ["lgbm", "xgb", "cat"]

        for m_type in model_types:
            logging.info(f"Training Expert: {m_type.upper()}")
            model = self.factory.train_model(
                model_type=m_type,
                X_train=X_train_expert,
                y_train=y_train_expert,
                X_val=X_val,
                y_val=y_val,
            )
            self.experts[m_type] = model

            # Save model
            utils.save_model(
                model, os.path.join(self.models_dir, f"expert_{m_type}.joblib")
            )

        logging.info("Expert training complete.")

    def evaluate_ensemble(self, df_val):
        """
        Phase 4: Evaluate Ensemble and Optimize Threshold.
        """
        logging.info(">>> PHASE 4: Ensemble Evaluation & Threshold Optimization")

        X_val = df_val[self.feature_cols]
        y_val = df_val[self.target_col].values

        # Generate predictions from all experts
        ensemble_probs = np.zeros(len(X_val))
        for m_type, model in self.experts.items():
            probs = self.factory.predict_proba(model, X_val)
            ensemble_probs += probs

        # Average
        ensemble_probs /= len(self.experts)

        # Optimize Threshold for MCC
        thresholds = np.arange(0.1, 0.91, 0.01)
        best_mcc = -1.0
        best_th = 0.5

        for th in thresholds:
            preds = (ensemble_probs >= th).astype(int)
            mcc = utils.calc_mcc(y_val, preds)
            if mcc > best_mcc:
                best_mcc = mcc
                best_th = th

        self.best_threshold = best_th
        logging.info(f"Best Validation MCC: {best_mcc}")
        logging.info(f"Optimal Threshold: {best_th}")

        # Save threshold
        np.save(
            os.path.join(self.models_dir, "best_threshold.npy"), np.array([best_th])
        )

    def predict_test(self, load_cached_data=True):
        """
        Phase 5: Inference on Test Set and Submission Generation.
        """
        logging.info(">>> PHASE 5: Test Inference")

        # Load Test Data (Features generated/loaded via loader)
        # Note: prepare_dataset handles the feature generation pipeline
        df_test = self.loader.prepare_dataset(
            split="test", load_cached_data=load_cached_data
        )

        if df_test.empty:
            logging.warning("Test dataset is empty. Creating dummy submission.")
            # Fallback to sample submission logic if needed, but usually not expected
            return

        X_test = df_test[self.feature_cols]
        contact_ids = df_test["contact_id"].values

        # Ensemble Prediction
        ensemble_probs = np.zeros(len(X_test))
        for m_type, model in self.experts.items():
            probs = self.factory.predict_proba(model, X_test)
            ensemble_probs += probs

        ensemble_probs /= len(self.experts)

        # Apply Threshold
        predictions = (ensemble_probs >= self.best_threshold).astype(int)

        # Create Submission DataFrame
        sub_df = pd.DataFrame({"contact_id": contact_ids, "contact": predictions})

        # The test set processed via metadata/gating might be a subset of the full sample_submission
        # if gating excluded rows. However, the competition usually requires all rows.
        # We must merge with sample_submission to ensure all IDs are present.
        # Rows filtered by gating are implicitly "No Contact" (0).

        sample_sub = pd.read_csv(
            os.path.join(config.INPUT_DIR, "sample_submission.csv")
        )

        # Merge: Left join sample_sub with predictions
        final_sub = sample_sub[["contact_id"]].merge(
            sub_df, on="contact_id", how="left"
        )

        # Fill NaNs (rows filtered by gating) with 0
        final_sub["contact"] = final_sub["contact"].fillna(0).astype(int)

        # Save
        save_path = config.SUBMISSION_PATH
        final_sub.to_csv(save_path, index=False)
        logging.info(f"Submission saved to {save_path}. Rows: {len(final_sub)}")

    def run(self, load_cached_data=True):
        """
        Main execution flow.
        """
        utils.setup_logging(os.path.join(config.WORKING_DIR, "training.log"))
        utils.seed_everything(config.SEED)

        logging.info("Starting RAD-ME Curriculum Training...")

        # 1. Load Data
        df_train = self.loader.prepare_dataset(
            split="train", load_cached_data=load_cached_data
        )
        df_val = self.loader.prepare_dataset(
            split="val", load_cached_data=load_cached_data
        )

        # 2. Train Scouts
        self.train_scouts(df_train, df_val)

        # 3. Mine Hard Negatives
        hard_neg_indices = self.mine_hard_negatives(df_train)

        # 4. Train Experts
        self.train_experts(df_train, hard_neg_indices, df_val)

        # Free memory
        del df_train
        gc.collect()

        # 5. Evaluate
        self.evaluate_ensemble(df_val)

        # Free memory
        del df_val
        gc.collect()

        # 6. Predict Test
        self.predict_test(load_cached_data=load_cached_data)

        logging.info("Pipeline completed successfully.")
