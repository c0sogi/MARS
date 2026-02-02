import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import (
    Timer,
    save_data,
    load_data,
    save_model,
    load_model,
    optimize_threshold,
    calc_mcc,
    seed_everything,
)
from library.data_factory import DataFactory
from library.model_zoo import TriEnsemble


class Workflow:
    """
    Orchestrates the Orthogonal-Spectral Vector-Anchored Ensemble (OSVA-E) pipeline.
    Manages Scout training, Hard Negative Mining, Expert Training, and Inference.
    """

    def __init__(self):
        self.config = Config
        self.factory = DataFactory()
        seed_everything(self.config.SEED)

    def train_scouts(self, load_cached_data=True):
        """
        Phase 1: Train Scout models on a balanced dataset to identify potential hard negatives.
        """
        with Timer("Workflow: Train Scouts"):
            # 1. Load Data
            df_train = self.factory.load_and_process_data(
                split="train", load_cached_data=load_cached_data
            )
            df_val = self.factory.load_and_process_data(
                split="val", load_cached_data=load_cached_data
            )

            # 2. Create Balanced Scout Dataset
            df_scout_train = self.factory.get_scout_dataset(df_train)

            # 3. Initialize and Train Scouts
            scouts = TriEnsemble()
            print("Training Scout Ensemble...")
            scouts.fit(df_scout_train, df_val)

            # 4. Save Scouts
            scout_dir = os.path.join(self.config.MODEL_DIR, "scouts")
            scouts.save(scout_dir)

            return scouts

    def mine_hard_negatives(self, scouts, load_cached_data=True):
        """
        Phase 2: Use Scouts to mine hard negatives from the full training pool.
        Hard Negative Definition: Ground Truth = 0 AND P(Contact) > Threshold by ANY Scout.
        """
        with Timer("Workflow: Mine Hard Negatives"):
            # 1. Load Full Training Data (Mining Pool)
            # This is the gated survivor set from feature engineering
            df_pool = self.factory.load_and_process_data(
                split="train", load_cached_data=load_cached_data
            )

            # 2. Identify Feature Columns
            # We need to ensure we use the same features as training
            feature_cols = scouts.feature_cols
            if feature_cols is None:
                # Fallback if not set (should be set during fit)
                feature_cols = scouts._get_feature_cols(df_pool)

            X_pool = df_pool[feature_cols]
            y_true = df_pool["contact"].values

            # 3. Get Predictions from EACH Scout Model (Union Logic)
            hard_negative_mask = np.zeros(len(df_pool), dtype=bool)

            print("Scanning mining pool with Scouts...")
            for name, model in scouts.models.items():
                # Predict probabilities
                probs = model.predict(X_pool)

                # Check condition: Is Negative AND Prob > Threshold
                # Note: y_true is binary 0/1 here.
                # We are looking for False Positives (Negatives predicted as high prob)
                mask = (y_true == 0) & (probs > self.config.SCOUT_PROB_THRESHOLD)

                # Union: If any model flags it, it's hard
                hard_negative_mask = hard_negative_mask | mask

            # 4. Extract Indices
            hard_indices = df_pool.index[hard_negative_mask].values
            print(
                f"Mined {len(hard_indices)} Hard Negatives from {len(df_pool)} samples."
            )

            # 5. Save Indices
            save_data(hard_indices, self.config.HARD_NEGATIVE_INDICES_PATH)

            return hard_indices

    def train_experts(self, hard_negative_indices=None, load_cached_data=True):
        """
        Phase 3: Train Expert models on the Anchored Dataset (Positives + Hard Negs + Anchors).
        Uses Soft Targets for training.
        """
        with Timer("Workflow: Train Experts"):
            # 1. Load Data
            df_train = self.factory.load_and_process_data(
                split="train", load_cached_data=load_cached_data
            )
            df_val = self.factory.load_and_process_data(
                split="val", load_cached_data=load_cached_data
            )

            # 2. Load Indices if not provided
            if hard_negative_indices is None:
                hard_negative_indices = load_data(
                    self.config.HARD_NEGATIVE_INDICES_PATH
                )
                if hard_negative_indices is None:
                    raise FileNotFoundError(
                        "Hard negative indices not found. Run mining first."
                    )

            # 3. Construct Expert Dataset (Anchored + Soft Labels)
            df_expert_train = self.factory.construct_expert_dataset(
                df_train, hard_negative_indices
            )

            # 4. Initialize and Train Experts
            experts = TriEnsemble()
            print("Training Expert Ensemble...")
            # Note: df_expert_train['contact'] contains soft labels now
            experts.fit(df_expert_train, df_val)

            # 5. Optimize Threshold on Validation Set
            print("Optimizing Threshold...")
            val_probs = experts.predict(df_val)
            val_true = df_val["contact"].values

            best_threshold, best_mcc = optimize_threshold(val_true, val_probs)
            print(f"Best Threshold: {best_threshold}")
            print(f"Validation MCC: {best_mcc}")

            # 6. Save Experts and Threshold
            expert_dir = os.path.join(self.config.MODEL_DIR, "experts")
            experts.save(expert_dir)

            # Save threshold as npy
            thresh_path = os.path.join(self.config.MODEL_DIR, "best_threshold.npy")
            np.save(thresh_path, np.array([best_threshold]))

            return experts, best_threshold

    def predict_test(self, experts, best_threshold, load_cached_data=True):
        """
        Phase 4: Inference on Test Set.
        Handles re-alignment with sample_submission.csv (filling gated rows with 0).
        """
        with Timer("Workflow: Predict Test"):
            # 1. Load Processed Test Data (Gated Survivors)
            df_test_features = self.factory.load_and_process_data(
                split="test", load_cached_data=load_cached_data
            )

            # 2. Generate Predictions for Survivors
            print(f"Predicting on {len(df_test_features)} gated test samples...")
            probs = experts.predict(df_test_features)

            # Apply Threshold
            preds_binary = (probs >= best_threshold).astype(int)

            # 3. Prepare Partial Submission DataFrame
            df_preds = pd.DataFrame(
                {"contact_id": df_test_features["contact_id"], "contact": preds_binary}
            )

            # 4. Load Sample Submission (The Template)
            sample_sub = pd.read_csv(self.config.SAMPLE_SUBMISSION_PATH)
            print(f"Sample Submission Rows: {len(sample_sub)}")

            # 5. Merge Predictions into Template
            # Left join on contact_id.
            # Rows present in df_preds get the prediction.
            # Rows missing (gated out) get NaN, which we fill with 0.

            # Drop original contact col from sample_sub to avoid collision
            if "contact" in sample_sub.columns:
                sample_sub = sample_sub.drop(columns=["contact"])

            final_sub = sample_sub.merge(df_preds, on="contact_id", how="left")

            # Fill NaNs with 0 (No Contact)
            final_sub["contact"] = final_sub["contact"].fillna(0).astype(int)

            # 6. Save Submission
            print(f"Saving submission to {self.config.SUBMISSION_PATH}...")
            final_sub.to_csv(self.config.SUBMISSION_PATH, index=False)

            return final_sub

    def run_full_workflow(self, load_cached_data=True):
        """
        Executes the complete pipeline.
        """
        # 1. Train Scouts
        scouts = self.train_scouts(load_cached_data=load_cached_data)

        # 2. Mine Hard Negatives
        hard_neg_indices = self.mine_hard_negatives(
            scouts, load_cached_data=load_cached_data
        )

        # Free memory
        del scouts
        gc.collect()

        # 3. Train Experts
        experts, threshold = self.train_experts(
            hard_neg_indices, load_cached_data=load_cached_data
        )

        # 4. Predict Test
        self.predict_test(experts, threshold, load_cached_data=load_cached_data)

        print("Workflow Completed Successfully.")
