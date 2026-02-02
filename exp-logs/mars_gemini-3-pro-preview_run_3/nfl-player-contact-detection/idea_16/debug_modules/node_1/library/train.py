import os
import gc
import json
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import setup_seed, calc_mcc, optimize_threshold
from library.data_loader import DataLoader
from library.feature_engineering import FeatureGenerator
from library.model import ContactXGB


class TrainPipeline:
    """
    Orchestrates the training of the Ego-Centric Dual-Stream GBDT architecture.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.thresholds_path = os.path.join(self.working_dir, "thresholds.json")
        setup_seed(Config.SEED)

    def _undersample(self, X, y, ids, neg_pos_ratio=Config.NEG_POS_RATIO):
        """
        Performs random undersampling on the negative class.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (np.ndarray): Labels.
            ids (np.ndarray): Contact IDs.
            neg_pos_ratio (float): Ratio of negatives to positives to keep.

        Returns:
            tuple: (X_resampled, y_resampled, ids_resampled)
        """
        print(f"Undersampling with Neg/Pos Ratio: {neg_pos_ratio}...")

        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]

        n_pos = len(pos_indices)
        n_neg = len(neg_indices)
        n_keep_neg = int(n_pos * neg_pos_ratio)

        if n_keep_neg < n_neg:
            # Randomly select negative indices
            np.random.seed(Config.SEED)
            keep_neg_indices = np.random.choice(neg_indices, n_keep_neg, replace=False)

            # Combine and shuffle
            combined_indices = np.concatenate([pos_indices, keep_neg_indices])
            np.random.shuffle(combined_indices)

            return (
                X.iloc[combined_indices].reset_index(drop=True),
                y[combined_indices],
                ids[combined_indices],
            )
        else:
            print(
                "Warning: Not enough negatives to satisfy ratio. Returning original data."
            )
            return X, y, ids

    def _train_stream(
        self, stream_name, df_train_merged, df_val_merged, load_cached_data
    ):
        """
        Handles feature generation, undersampling, training, and threshold optimization for a single stream.
        """
        print(f"\n=== Starting Pipeline for {stream_name} ===")

        # 1. Feature Generation
        fg = FeatureGenerator(run_mode="train")
        X_train, y_train, ids_train = fg.generate_features(
            df_train_merged, stream=stream_name, load_cached_data=load_cached_data
        )

        fg_val = FeatureGenerator(run_mode="validation")
        X_val, y_val, ids_val = fg_val.generate_features(
            df_val_merged, stream=stream_name, load_cached_data=load_cached_data
        )

        # 2. Undersampling (Train only)
        X_train_res, y_train_res, _ = self._undersample(X_train, y_train, ids_train)

        print(f"Train shape after sampling: {X_train_res.shape}")
        print(f"Validation shape: {X_val.shape}")

        # 3. Model Initialization
        if stream_name == "stream_a":
            params = Config.XGB_PARAMS_STREAM_A
        else:
            params = Config.XGB_PARAMS_STREAM_B

        model = ContactXGB(params)

        # 4. Training
        model.fit(X_train_res, y_train_res, X_val, y_val)

        # 5. Save Model
        model_path = os.path.join(self.working_dir, f"model_{stream_name}.json")
        model.save(model_path)

        # 6. Validation Prediction & Threshold Optimization
        print(f"Optimizing threshold for {stream_name}...")
        y_pred_proba = model.predict_proba(X_val)
        best_thresh, best_mcc = optimize_threshold(y_val, y_pred_proba)

        print(f"Best Threshold for {stream_name}: {best_thresh}")
        print(f"Best MCC for {stream_name}: {best_mcc}")

        # Store predictions for global evaluation
        val_preds_df = pd.DataFrame(
            {
                "contact_id": ids_val,
                "y_true": y_val,
                "y_prob": y_pred_proba,
                "stream": stream_name,
            }
        )

        # Cleanup to save memory
        del X_train, y_train, ids_train, X_train_res, y_train_res
        del X_val, y_val, ids_val, y_pred_proba
        gc.collect()

        return best_thresh, val_preds_df

    def run_training(self, load_cached_data=True):
        """
        Main execution method for the training pipeline.

        Args:
            load_cached_data (bool): Whether to use cached intermediate files.
        """
        print("Initializing Training Pipeline...")

        # --- Step 1: Load and Merge Data ---
        print("\n--- Loading Data ---")

        # Train Data
        loader_train = DataLoader(run_mode="train")
        meta_train = loader_train.load_metadata()
        track_train = loader_train.load_tracking(meta_train["game_play"].unique())
        helm_train = loader_train.load_helmets(meta_train["game_play"].unique())
        df_train_merged = loader_train.merge_data(
            meta_train, track_train, helm_train, load_cached_data=load_cached_data
        )

        # Validation Data
        loader_val = DataLoader(run_mode="validation")
        meta_val = loader_val.load_metadata()
        track_val = loader_val.load_tracking(meta_val["game_play"].unique())
        helm_val = loader_val.load_helmets(meta_val["game_play"].unique())
        df_val_merged = loader_val.merge_data(
            meta_val, track_val, helm_val, load_cached_data=load_cached_data
        )

        # Clear raw data to free memory for feature generation
        del meta_train, track_train, helm_train
        del meta_val, track_val, helm_val
        gc.collect()

        # --- Step 2: Stream A (Interaction) ---
        thresh_a, preds_a = self._train_stream(
            "stream_a", df_train_merged, df_val_merged, load_cached_data
        )

        # --- Step 3: Stream B (Impact) ---
        thresh_b, preds_b = self._train_stream(
            "stream_b", df_train_merged, df_val_merged, load_cached_data
        )

        # --- Step 4: Global Evaluation & Saving ---
        print("\n--- Global Evaluation ---")

        # Combine predictions
        all_preds = pd.concat([preds_a, preds_b], axis=0)

        # Apply specific thresholds
        all_preds["y_pred"] = 0

        mask_a = all_preds["stream"] == "stream_a"
        all_preds.loc[mask_a, "y_pred"] = (
            all_preds.loc[mask_a, "y_prob"] >= thresh_a
        ).astype(int)

        mask_b = all_preds["stream"] == "stream_b"
        all_preds.loc[mask_b, "y_pred"] = (
            all_preds.loc[mask_b, "y_prob"] >= thresh_b
        ).astype(int)

        # Calculate Global MCC
        global_mcc = calc_mcc(all_preds["y_true"], all_preds["y_pred"])
        print(f"Global Validation MCC: {global_mcc}")

        # Save Thresholds
        thresholds = {
            "stream_a": float(thresh_a),
            "stream_b": float(thresh_b),
            "global_mcc": float(global_mcc),
        }

        with open(self.thresholds_path, "w") as f:
            json.dump(thresholds, f, indent=4)

        print(f"Thresholds saved to {self.thresholds_path}")
        print("Training Pipeline Completed Successfully.")
