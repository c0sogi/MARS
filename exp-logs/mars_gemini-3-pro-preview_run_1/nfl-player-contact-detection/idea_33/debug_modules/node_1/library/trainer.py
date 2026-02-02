import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, log_loss
from library.config import Config
from library.utils import seed_everything, CacheManager
from library.feature_engine import FeatureEngine
from library.model_zoo import LGBMExpert, XGBExpert, HistGBExpert


class Trainer:
    """
    Orchestrates the DEIB-AME pipeline:
    1. Train Scouts on balanced data.
    2. Mine Hard Negatives using Scouts.
    3. Train Experts on Positives + Hard Negatives + Anchors.
    4. Optimize Threshold on Validation.
    5. Generate Submission.
    """

    def __init__(self):
        self.feature_engine = FeatureEngine()
        self.cache_manager = CacheManager()
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        # Subdirectories for model tiers
        self.scout_dir = os.path.join(self.models_dir, "scouts")
        self.expert_dir = os.path.join(self.models_dir, "experts")
        os.makedirs(self.scout_dir, exist_ok=True)
        os.makedirs(self.expert_dir, exist_ok=True)

    def _get_model_instance(self, model_type):
        if model_type == "lgbm":
            return LGBMExpert()
        elif model_type == "xgb":
            return XGBExpert()
        elif model_type == "hgb":
            return HistGBExpert()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train_scouts(self, df_train):
        """
        Phase 1: Train Scout models on a balanced subset of the gated data.
        """
        print("\n--- Phase 1: Training Scouts ---")

        # Prepare Balanced Dataset for Scouts
        positives = df_train[df_train["contact"] == 1]
        negatives = df_train[df_train["contact"] == 0]

        # Sample negatives to match positives 1:1 for initial scouting
        n_pos = len(positives)
        if len(negatives) > n_pos:
            negatives_sampled = negatives.sample(n=n_pos, random_state=Config.SEED)
        else:
            negatives_sampled = negatives

        df_scout = (
            pd.concat([positives, negatives_sampled])
            .sample(frac=1, random_state=Config.SEED)
            .reset_index(drop=True)
        )

        X = df_scout.drop(
            columns=[
                "contact",
                "contact_id",
                "game_play",
                "step",
                "nfl_player_id_1",
                "nfl_player_id_2",
            ],
            errors="ignore",
        )
        y = df_scout["contact"]

        scouts = {}
        for model_type in ["lgbm", "xgb", "hgb"]:
            model_path = os.path.join(self.scout_dir, f"{model_type}_model.joblib")

            # Check if model exists
            if os.path.exists(model_path):
                print(f"Loading existing Scout ({model_type})...")
                model = self._get_model_instance(model_type)
                model.load(model_path)
            else:
                print(f"Training Scout ({model_type})...")
                model = self._get_model_instance(model_type)
                model.fit(X, y)
                model.save(model_path)

            scouts[model_type] = model

        return scouts

    def mine_hard_negatives(self, df_train, scouts, load_cached_data=True):
        """
        Phase 2: Use Scouts to mine Hard Negatives from the entire negative pool.
        Hard Negative: Any negative sample where ANY scout predicts prob > Threshold.
        """
        print("\n--- Phase 2: Mining Hard Negatives ---")

        cache_file = "hard_negative_indices.npy"

        if load_cached_data and self.cache_manager.exists(cache_file):
            print("Loading cached hard negative indices...")
            return self.cache_manager.load(cache_file)

        # Filter for negatives only
        # We need to keep the original index to map back if we were subsetting,
        # but here we can just return the boolean mask or the subset of data.
        # Ideally, we return indices relative to df_train.

        neg_mask = df_train["contact"] == 0
        df_neg = df_train[neg_mask].copy()

        if len(df_neg) == 0:
            return np.array([])

        X_neg = df_neg.drop(
            columns=[
                "contact",
                "contact_id",
                "game_play",
                "step",
                "nfl_player_id_1",
                "nfl_player_id_2",
            ],
            errors="ignore",
        )

        # Ensemble Prediction (Union Logic)
        # Initialize max probability with zeros
        max_probs = np.zeros(len(df_neg))

        for name, model in scouts.items():
            print(f"Scout {name} scanning negatives...")
            probs = model.predict_proba(X_neg)
            max_probs = np.maximum(max_probs, probs)

        # Identify Hard Negatives
        hard_neg_local_mask = max_probs > Config.HARD_NEGATIVE_THRESHOLD

        # Get the global indices from df_train
        # df_neg is a slice, so we need its index
        hard_neg_indices = df_neg.index[hard_neg_local_mask].to_numpy()

        print(
            f"Mined {len(hard_neg_indices)} Hard Negatives out of {len(df_neg)} total negatives."
        )

        # Cache results
        self.cache_manager.save(hard_neg_indices, cache_file)

        return hard_neg_indices

    def train_experts(self, df_train, hard_neg_indices):
        """
        Phase 3: Train Expert models on Positives + Hard Negatives + Anchors.
        """
        print("\n--- Phase 3: Training Experts ---")

        # 1. Positives
        positives = df_train[df_train["contact"] == 1]

        # 2. Hard Negatives
        if len(hard_neg_indices) > 0:
            hard_negatives = df_train.loc[hard_neg_indices]
        else:
            hard_negatives = pd.DataFrame(columns=df_train.columns)

        # 3. Random Anchors (Easy Negatives)
        # Exclude hard negatives from the negative pool first
        neg_mask = df_train["contact"] == 0
        # Create a boolean mask for hard negatives
        is_hard = np.zeros(len(df_train), dtype=bool)
        if len(hard_neg_indices) > 0:
            is_hard[hard_neg_indices] = True

        easy_neg_candidates = df_train[neg_mask & ~is_hard]

        n_anchors = int(len(positives) * Config.ANCHOR_RATIO)
        # Clamp anchor count
        n_anchors = min(n_anchors, len(easy_neg_candidates))

        if n_anchors > 0:
            anchors = easy_neg_candidates.sample(n=n_anchors, random_state=Config.SEED)
        else:
            anchors = pd.DataFrame(columns=df_train.columns)

        print(f"Expert Dataset Composition:")
        print(f"  Positives: {len(positives)}")
        print(f"  Hard Negatives: {len(hard_negatives)}")
        print(f"  Anchors: {len(anchors)}")

        df_expert = (
            pd.concat([positives, hard_negatives, anchors])
            .sample(frac=1, random_state=Config.SEED)
            .reset_index(drop=True)
        )

        X = df_expert.drop(
            columns=[
                "contact",
                "contact_id",
                "game_play",
                "step",
                "nfl_player_id_1",
                "nfl_player_id_2",
            ],
            errors="ignore",
        )
        y = df_expert["contact"]

        experts = {}
        for model_type in ["lgbm", "xgb", "hgb"]:
            model_path = os.path.join(self.expert_dir, f"{model_type}_model.joblib")

            if os.path.exists(model_path):
                print(f"Loading existing Expert ({model_type})...")
                model = self._get_model_instance(model_type)
                model.load(model_path)
            else:
                print(f"Training Expert ({model_type})...")
                model = self._get_model_instance(model_type)
                model.fit(X, y)
                model.save(model_path)

            experts[model_type] = model

        return experts

    def evaluate_and_tune(self, df_val, experts):
        """
        Phase 4: Evaluate Experts on Validation set and optimize threshold.
        """
        print("\n--- Phase 4: Evaluation & Threshold Tuning ---")

        X_val = df_val.drop(
            columns=[
                "contact",
                "contact_id",
                "game_play",
                "step",
                "nfl_player_id_1",
                "nfl_player_id_2",
            ],
            errors="ignore",
        )
        y_val = df_val["contact"].values

        # Ensemble Prediction (Average)
        probs_sum = np.zeros(len(X_val))
        for name, model in experts.items():
            p = model.predict_proba(X_val)
            probs_sum += p

        y_prob = probs_sum / len(experts)

        # Calculate LogLoss
        ll = log_loss(y_val, y_prob)
        print(f"Ensemble Validation LogLoss: {ll}")

        # Threshold Optimization
        thresholds = np.arange(0.01, 1.00, 0.01)
        best_mcc = -1.0
        best_thresh = 0.5

        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            mcc = matthews_corrcoef(y_val, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = thresh

        print(f"Best Threshold: {best_thresh}")
        print(f"Best Validation MCC: {best_mcc}")

        # Save best threshold
        np.save(
            os.path.join(self.models_dir, "best_threshold.npy"), np.array([best_thresh])
        )

        return best_thresh

    def predict_test(self, df_test, experts, threshold):
        """
        Phase 5: Inference on Test Set and Submission Generation.
        """
        print("\n--- Phase 5: Inference & Submission ---")

        X_test = df_test.drop(
            columns=[
                "contact",
                "contact_id",
                "game_play",
                "step",
                "nfl_player_id_1",
                "nfl_player_id_2",
            ],
            errors="ignore",
        )

        # Ensemble Prediction
        probs_sum = np.zeros(len(X_test))
        for name, model in experts.items():
            p = model.predict_proba(X_test)
            probs_sum += p

        y_prob = probs_sum / len(experts)
        y_pred = (y_prob >= threshold).astype(int)

        # Create submission DataFrame
        # We need to map predictions back to contact_ids.
        # df_test contains the filtered survivors from gating.
        # The submission file expects ALL contact_ids.
        # Strategy: Load sample_submission, merge predictions, fill missing with 0.

        print("Loading sample submission template...")
        sample_sub = pd.read_csv(
            os.path.join(Config.INPUT_DIR, "sample_submission.csv")
        )

        # Create a dataframe of predictions
        pred_df = pd.DataFrame(
            {"contact_id": df_test["contact_id"], "contact_pred": y_pred}
        )

        # Merge
        submission = sample_sub.drop(columns=["contact"], errors="ignore").merge(
            pred_df, on="contact_id", how="left"
        )

        # Fill NaN (gated out rows) with 0
        submission["contact"] = submission["contact_pred"].fillna(0).astype(int)
        submission = submission.drop(columns=["contact_pred"])

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {submission.shape}")
        print(f"Predicted Contacts: {submission['contact'].sum()}")

    def run(self, load_cached_data=True, sample_size=None):
        """
        Main execution flow.
        """
        seed_everything(Config.SEED)

        # 1. Load and Process Data
        print("Loading and processing training data...")
        df_train = self.feature_engine.process_train(
            load_cached_data=load_cached_data, sample_size=sample_size
        )

        print("Loading and processing validation data...")
        df_val = self.feature_engine.process_val(
            load_cached_data=load_cached_data, sample_size=sample_size
        )

        # 2. Train Scouts
        scouts = self.train_scouts(df_train)

        # 3. Mine Hard Negatives
        hard_neg_indices = self.mine_hard_negatives(
            df_train, scouts, load_cached_data=load_cached_data
        )

        # 4. Train Experts
        experts = self.train_experts(df_train, hard_neg_indices)

        # 5. Evaluate
        best_threshold = self.evaluate_and_tune(df_val, experts)

        # 6. Inference
        print("Loading and processing test data...")
        df_test = self.feature_engine.process_test(
            load_cached_data=load_cached_data, sample_size=sample_size
        )

        self.predict_test(df_test, experts, best_threshold)
