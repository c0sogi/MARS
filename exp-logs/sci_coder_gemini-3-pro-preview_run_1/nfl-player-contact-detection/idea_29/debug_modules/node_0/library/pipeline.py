import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import matthews_corrcoef

from library.config import Config
from library.utils import seed_everything
from library.data import NFLDataLoader
from library.models import TriEnsemble


class TrainingPipeline:
    """
    Orchestrates the Dual-Basis Kinematic-Spectral Anchored-Ensemble (DB-KSAE) pipeline.
    """

    def __init__(self):
        self.loader = NFLDataLoader()
        seed_everything(Config.SEED)

        # Paths for intermediate artifacts
        self.threshold_path = os.path.join(Config.MODEL_DIR, "best_threshold.npy")

    def train_scouts(self, load_cached_data=True):
        """
        Phase 1: Train the Scout models on a balanced dataset of Gated Survivors.
        """
        print("\n--- Phase 1: Training Scouts ---")

        # 1. Load Training Features
        df_train = self.loader.load_features(
            stage="train", load_cached_data=load_cached_data
        )

        # 2. Get Balanced Scout Data
        X_scout, y_scout = self.loader.get_scout_data(df_train)
        print(f"Scout Training Data Shape: {X_scout.shape}")

        # 3. Initialize and Train Scouts
        scouts = TriEnsemble()
        scouts.fit(X_scout, y_scout)

        # 4. Save Scout Models
        scouts.save_models(suffix="scout")
        print("Scout models trained and saved.")

    def mine_hard_negatives(self, load_cached_data=True):
        """
        Phase 2: Use Scouts to mine Hard Negatives from the full Gated Survivor pool.
        """
        print("\n--- Phase 2: Mining Hard Negatives ---")

        # Check cache
        if load_cached_data and os.path.exists(Config.CACHE_HARD_NEGATIVES):
            print(
                f"Loading cached hard negative indices from {Config.CACHE_HARD_NEGATIVES}..."
            )
            return np.load(Config.CACHE_HARD_NEGATIVES, allow_pickle=True)

        # 1. Load Data and Models
        df_train = self.loader.load_features(
            stage="train", load_cached_data=load_cached_data
        )

        scouts = TriEnsemble()
        scouts.load_models(suffix="scout")

        # 2. Get Candidates (All Gated Survivors)
        df_candidates, X_candidates = self.loader.get_mining_candidates(df_train)
        print(f"Mining candidates: {len(df_candidates)}")

        # 3. Predict with all Scouts (Union Logic)
        # We access the internal models directly to get individual probabilities
        p_lgbm = scouts.lgbm.predict_proba(X_candidates)[:, 1]
        p_xgb = scouts.xgb.predict_proba(X_candidates)[:, 1]
        p_hgb = scouts.hgb.predict_proba(X_candidates)[:, 1]

        # Max probability across ensemble (Union of detections)
        max_probs = np.maximum(np.maximum(p_lgbm, p_xgb), p_hgb)

        # 4. Identify Hard Negatives
        # Condition: Probability > Threshold AND Actual Label == 0
        # We rely on the index of df_candidates matching the original df_train index
        mask_hard = (max_probs > Config.SCOUT_THRESHOLD) & (
            df_candidates["contact"] == 0
        )
        hard_negative_indices = df_candidates[mask_hard].index.to_numpy()

        print(f"Found {len(hard_negative_indices)} hard negatives.")

        # 5. Save to Cache
        np.save(Config.CACHE_HARD_NEGATIVES, hard_negative_indices)
        print(f"Hard negative indices saved to {Config.CACHE_HARD_NEGATIVES}")

        return hard_negative_indices

    def train_expert(self, load_cached_data=True):
        """
        Phase 3: Train the Expert Ensemble on the Anchored Dataset with Soft Targets.
        """
        print("\n--- Phase 3: Training Expert Ensemble ---")

        # 1. Load Training Features
        df_train = self.loader.load_features(
            stage="train", load_cached_data=load_cached_data
        )

        # 2. Load Hard Negatives
        if os.path.exists(Config.CACHE_HARD_NEGATIVES):
            hard_negative_indices = np.load(
                Config.CACHE_HARD_NEGATIVES, allow_pickle=True
            )
        else:
            # Fallback if pipeline broken, though ideally should raise error
            print("Warning: Hard negative cache not found. Running mining...")
            hard_negative_indices = self.mine_hard_negatives(
                load_cached_data=load_cached_data
            )

        # 3. Construct Expert Dataset (Positives + Hard Negatives + Anchors)
        # This handles label smoothing internally
        X_expert, y_expert = self.loader.get_expert_data(
            df_train, hard_negative_indices, anchor_ratio=1.0
        )
        print(f"Expert Training Data Shape: {X_expert.shape}")

        # 4. Load Validation Data (for monitoring/early stopping)
        df_val = self.loader.load_features(
            stage="val", load_cached_data=load_cached_data
        )

        # Prepare Validation Set (Standard features, Binary targets)
        feature_cols = [
            "distance",
            "v_comp1",
            "v_comp2",
            "a_comp1",
            "a_comp2",
            "j_comp1",
            "j_comp2",
            "min_dist_pred",
        ]
        X_val = df_val[feature_cols]
        y_val = df_val["contact"]

        # 5. Train Expert Ensemble
        expert = TriEnsemble()
        expert.fit(X_expert, y_expert, X_val, y_val)

        # 6. Save Expert Models
        expert.save_models(suffix="expert")
        print("Expert models trained and saved.")

    def evaluate(self, load_cached_data=True):
        """
        Evaluates the Expert Ensemble on the validation set and optimizes the threshold.
        """
        print("\n--- Phase 4: Evaluation and Threshold Tuning ---")

        # 1. Load Validation Data
        df_val = self.loader.load_features(
            stage="val", load_cached_data=load_cached_data
        )
        feature_cols = [
            "distance",
            "v_comp1",
            "v_comp2",
            "a_comp1",
            "a_comp2",
            "j_comp1",
            "j_comp2",
            "min_dist_pred",
        ]
        X_val = df_val[feature_cols]
        y_val = df_val["contact"].values

        # 2. Load Expert Models
        expert = TriEnsemble()
        expert.load_models(suffix="expert")

        # 3. Predict Probabilities
        probs = expert.predict_proba(X_val)

        # 4. Grid Search for Best Threshold
        best_threshold = 0.5
        best_mcc = -1.0

        thresholds = np.arange(0.1, 0.91, 0.01)
        for t in thresholds:
            preds = (probs > t).astype(int)
            mcc = matthews_corrcoef(y_val, preds)
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = t

        print(f"Best Threshold: {best_threshold}")
        print(f"Best Validation MCC: {best_mcc}")

        # 5. Save Threshold
        np.save(self.threshold_path, np.array([best_threshold]))
        print(f"Best threshold saved to {self.threshold_path}")

        return best_mcc

    def inference(self, load_cached_data=True):
        """
        Generates the final submission for the test set.
        """
        print("\n--- Phase 5: Inference and Submission ---")

        # 1. Load Test Features
        df_test = self.loader.load_features(
            stage="test", load_cached_data=load_cached_data
        )
        feature_cols = [
            "distance",
            "v_comp1",
            "v_comp2",
            "a_comp1",
            "a_comp2",
            "j_comp1",
            "j_comp2",
            "min_dist_pred",
        ]
        X_test = df_test[feature_cols]

        # 2. Load Expert Models
        expert = TriEnsemble()
        expert.load_models(suffix="expert")

        # 3. Load Threshold
        if os.path.exists(self.threshold_path):
            best_threshold = float(np.load(self.threshold_path)[0])
            print(f"Loaded best threshold: {best_threshold}")
        else:
            best_threshold = 0.5
            print(f"Warning: Threshold file not found. Using default: {best_threshold}")

        # 4. Generate Predictions
        probs = expert.predict_proba(X_test)
        preds = (probs > best_threshold).astype(int)

        # 5. Format Submission
        submission = pd.DataFrame(
            {"contact_id": df_test["contact_id"], "contact": preds}
        )

        # 6. Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {submission.shape}")
