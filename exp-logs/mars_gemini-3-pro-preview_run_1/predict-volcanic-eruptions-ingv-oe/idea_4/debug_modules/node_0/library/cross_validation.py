import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, calculate_mae
from library.feature_engineering import TabularFeatureExtractor
from library.spectrogram_processing import SpectrogramGenerator
from library.data_loaders import SeismicCNNDataset
from library.training_engines import CNNTrainer, LGBMTrainer, RidgeStacker


class CrossValidator:
    """
    Orchestrates the 5-Fold Cross-Validation, Model Training (Branch A & B),
    Stacking, and Submission Generation.
    """

    def __init__(self):
        self.n_folds = Config.N_FOLDS
        self.seed = Config.SEED
        self.working_dir = Config.WORKING_DIR
        self.submission_path = Config.SUBMISSION_PATH

        # Ensure directories exist
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.submission_path), exist_ok=True)

    def _load_and_align_data(self):
        """
        Loads Train, Val, and Test data for both Tabular and Spectrogram branches.
        Concatenates Train and Val to form a full development set.
        Aligns data by sorting by segment_id.
        """
        print("Loading and aligning data...")

        # --- 1. Load Tabular Data ---
        tab_extractor = TabularFeatureExtractor()

        # Train & Val (to be merged)
        X_tab_train, y_tab_train, ids_tab_train = tab_extractor.get_features("train")
        X_tab_val, y_tab_val, ids_tab_val = tab_extractor.get_features("val")

        # Test
        X_tab_test, _, ids_tab_test = tab_extractor.get_features("test")

        # --- 2. Load Spectrogram Data ---
        spec_generator = SpectrogramGenerator()

        # Train & Val (to be merged)
        X_spec_train, y_spec_train, ids_spec_train = spec_generator.get_dataset("train")
        X_spec_val, y_spec_val, ids_spec_val = spec_generator.get_dataset("val")

        # Test
        X_spec_test, _, ids_spec_test = spec_generator.get_dataset("test")

        # --- 3. Merge Train and Val for CV ---
        # Tabular
        X_tab_full = pd.concat([X_tab_train, X_tab_val], axis=0)
        y_tab_full = pd.concat([y_tab_train, y_tab_val], axis=0)
        ids_tab_full = pd.concat([ids_tab_train, ids_tab_val], axis=0)

        # Spectrograms
        X_spec_full = np.concatenate([X_spec_train, X_spec_val], axis=0)
        y_spec_full = np.concatenate([y_spec_train, y_spec_val], axis=0)
        ids_spec_full = np.concatenate([ids_spec_train, ids_spec_val], axis=0)

        # --- 4. Align Development Data (Sort by Segment ID) ---
        # Create a sorter index based on IDs
        sort_idx = np.argsort(ids_spec_full)

        # Apply to Spectrograms
        ids_spec_full = ids_spec_full[sort_idx]
        X_spec_full = X_spec_full[sort_idx]
        y_spec_full = y_spec_full[sort_idx]

        # Apply to Tabular
        # Ensure tabular IDs match spectrogram IDs before sorting to be safe
        # We assume the set of IDs is identical (verified by metadata script)
        # We re-index the pandas objects to match the sorted order of spectrogram IDs
        X_tab_full = (
            X_tab_full.set_index(ids_tab_full).loc[ids_spec_full].reset_index(drop=True)
        )
        y_tab_full = (
            y_tab_full.set_index(ids_tab_full).loc[ids_spec_full].reset_index(drop=True)
        )
        ids_tab_full = (
            ids_tab_full.set_index(ids_tab_full)
            .loc[ids_spec_full]
            .reset_index(drop=True)
        )

        # Verify alignment
        if not np.all(ids_tab_full.values == ids_spec_full):
            raise ValueError(
                "Tabular and Spectrogram IDs do not match after alignment!"
            )

        # --- 5. Align Test Data ---
        sort_idx_test = np.argsort(ids_spec_test)

        ids_spec_test = ids_spec_test[sort_idx_test]
        X_spec_test = X_spec_test[sort_idx_test]

        X_tab_test = (
            X_tab_test.set_index(ids_tab_test).loc[ids_spec_test].reset_index(drop=True)
        )
        ids_tab_test = (
            ids_tab_test.set_index(ids_tab_test)
            .loc[ids_spec_test]
            .reset_index(drop=True)
        )

        return {
            "dev": {
                "ids": ids_spec_full,
                "X_tab": X_tab_full,
                "y_tab": y_tab_full,  # Real values
                "X_spec": X_spec_full,
                "y_spec": y_spec_full,  # Scaled values (if Config.TARGET_SCALING is set)
            },
            "test": {"ids": ids_spec_test, "X_tab": X_tab_test, "X_spec": X_spec_test},
        }

    def run(self):
        seed_everything(self.seed)

        # Load Data
        data = self._load_and_align_data()

        dev_ids = data["dev"]["ids"]
        X_tab = data["dev"]["X_tab"]
        y_tab = data["dev"]["y_tab"]  # Real
        X_spec = data["dev"]["X_spec"]
        y_spec = data["dev"]["y_spec"]  # Scaled

        test_ids = data["test"]["ids"]
        X_tab_test = data["test"]["X_tab"]
        X_spec_test = data["test"]["X_spec"]

        # Initialize OOF and Test Prediction Storage
        oof_preds_lgbm = np.zeros(len(dev_ids))
        oof_preds_cnn = np.zeros(len(dev_ids))

        test_preds_lgbm = np.zeros(len(test_ids))
        test_preds_cnn = np.zeros(len(test_ids))

        # K-Fold Cross Validation
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        print(f"\nStarting {self.n_folds}-Fold Cross-Validation...")

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_tab)):
            print(f"\n=== Fold {fold} ===")

            # --- 1. Branch A: LightGBM ---
            print("--- Training Branch A (LightGBM) ---")
            X_train_fold_tab, X_val_fold_tab = (
                X_tab.iloc[train_idx],
                X_tab.iloc[val_idx],
            )
            y_train_fold_tab, y_val_fold_tab = (
                y_tab.iloc[train_idx],
                y_tab.iloc[val_idx],
            )

            lgbm_trainer = LGBMTrainer()
            _, best_score_lgbm = lgbm_trainer.train(
                X_train_fold_tab,
                y_train_fold_tab,
                X_val_fold_tab,
                y_val_fold_tab,
                fold=fold,
            )

            # Predict OOF
            oof_preds_lgbm[val_idx] = lgbm_trainer.predict(X_val_fold_tab, fold=fold)

            # Predict Test (Accumulate)
            test_preds_lgbm += (
                lgbm_trainer.predict(X_tab_test, fold=fold) / self.n_folds
            )

            # --- 2. Branch B: CNN ---
            print("--- Training Branch B (CNN) ---")
            # Slice Spectrogram Data
            X_train_fold_spec, X_val_fold_spec = X_spec[train_idx], X_spec[val_idx]
            y_train_fold_spec, y_val_fold_spec = y_spec[train_idx], y_spec[val_idx]

            # Create DataLoaders
            # Note: y_spec is already scaled (e.g. log1p) from the generator
            train_ds = SeismicCNNDataset(
                X_train_fold_spec, y_train_fold_spec, is_train=True
            )
            val_ds = SeismicCNNDataset(X_val_fold_spec, y_val_fold_spec, is_train=False)

            train_loader = DataLoader(
                train_ds,
                batch_size=Config.CNN_BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=torch.cuda.is_available(),
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=Config.CNN_BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=torch.cuda.is_available(),
            )

            cnn_trainer = CNNTrainer()
            best_mae_cnn = cnn_trainer.train(train_loader, val_loader, fold=fold)

            # Predict OOF (Real Scale)
            # We create a loader for validation set just for prediction to ensure consistent batching
            # But we can reuse val_loader since it's not shuffled
            oof_preds_cnn[val_idx] = cnn_trainer.predict(val_loader, fold=fold)

            # Predict Test (Real Scale, Accumulate)
            test_ds = SeismicCNNDataset(X_spec_test, y=None, is_train=False)
            test_loader = DataLoader(
                test_ds,
                batch_size=Config.CNN_BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=torch.cuda.is_available(),
            )
            test_preds_cnn += cnn_trainer.predict(test_loader, fold=fold) / self.n_folds

        # --- 3. Meta-Learner (Stacking) ---
        print("\n=== Training Meta-Learner (Stacking) ===")

        # Prepare Meta-Features
        # Shape: (N_samples, 2)
        X_meta_train = np.column_stack([oof_preds_lgbm, oof_preds_cnn])
        X_meta_test = np.column_stack([test_preds_lgbm, test_preds_cnn])

        # Train Ridge Stacker
        stacker = RidgeStacker()
        # y_tab contains the real target values
        stacker.fit(X_meta_train, y_tab.values)

        # Evaluate Stacking Performance
        final_oof_preds = stacker.predict(X_meta_train)
        final_mae = calculate_mae(y_tab.values, final_oof_preds)
        print(f"Final OOF MAE (Stacked): {final_mae}")

        # --- 4. Generate Submission ---
        print("\nGenerating Submission...")
        final_test_preds = stacker.predict(X_meta_test)

        submission_df = pd.DataFrame(
            {"segment_id": test_ids, "time_to_eruption": final_test_preds}
        )

        # Ensure integer format if required, but float is usually safer for regression unless specified
        # The sample submission shows integers, but metric is MAE on time.
        # Usually time is continuous. However, sample submission has 0 (int).
        # We will keep as float or round? The sample submission snippet shows '0',
        # but the prompt says "time_to_eruption (int64)".
        # Given MAE metric, high precision is better.
        # However, if the target is strictly int, we might round.
        # Let's stick to the raw prediction for maximum precision,
        # but the sample submission implies int.
        # Let's round to nearest integer to match the likely ground truth format.
        # Actually, let's just save as is (or standard float) to be safe,
        # but the prompt says "time_to_eruption (int64)".
        # Let's cast to int to be strictly compliant with the sample format description.
        submission_df["time_to_eruption"] = (
            submission_df["time_to_eruption"].round().astype(int)
        )

        submission_df.to_csv(self.submission_path, index=False)
        print(f"Submission saved to {self.submission_path}")
