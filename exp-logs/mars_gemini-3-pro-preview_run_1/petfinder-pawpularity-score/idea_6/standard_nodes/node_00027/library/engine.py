import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from library.config import Config
from library.utils import seed_everything, compute_rmse
from library.processors import DataProcessor
from library.models import ModelFactory


class StackingTrainer:
    """
    Orchestrates the Tri-Paradigm Stacking workflow with Heterogeneous Interaction Experts.
    Manages Cross-Validation, Level-0 Expert training, caching, and Level-1 Meta-Learning.
    """

    def __init__(self):
        self.processor = DataProcessor()
        self.models = ModelFactory()

    def _run_cv_for_backbone(
        self, backbone_name, X_full, meta_full, y_full, X_test, meta_test
    ):
        """
        Executes 5-Fold Cross-Validation for a specific backbone.
        Trains 3 experts (Linear, Kernel, Partitioning) per fold.

        Args:
            backbone_name (str): Name of the backbone model.
            X_full (np.ndarray): Combined Train+Val embeddings.
            meta_full (np.ndarray): Combined Train+Val metadata.
            y_full (np.ndarray): Combined Train+Val targets.
            X_test (np.ndarray): Test set embeddings.
            meta_test (np.ndarray): Test set metadata.

        Returns:
            tuple: (oof_predictions, test_predictions)
                   oof_predictions: (N_full, 3)
                   test_predictions: (N_test, 3)
        """
        kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

        # Initialize containers
        # Columns: 0=Linear(Ridge), 1=Kernel(SVR), 2=Partitioning(ExtraTrees)
        oof_preds = np.zeros((len(X_full), 3), dtype=np.float32)
        test_preds_accum = np.zeros((len(X_test), 3), dtype=np.float32)

        print(f"[{backbone_name}] Starting {Config.N_FOLDS}-Fold CV...")

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_full, y_full)):
            # Split Data
            X_train, X_val = X_full[train_idx], X_full[val_idx]
            meta_train, meta_val = meta_full[train_idx], meta_full[val_idx]
            y_train, y_val = y_full[train_idx], y_full[val_idx]

            # -----------------------------------------------------------------
            # Pipeline A: Distance-Based Experts (Linear & Kernel)
            # Strategy: Concatenate Embeddings + Metadata -> StandardScaler
            # -----------------------------------------------------------------
            X_lin_train, scaler = self.processor.prepare_linear_features(
                X_train, meta_train, scaler=None
            )
            X_lin_val, _ = self.processor.prepare_linear_features(
                X_val, meta_val, scaler=scaler
            )
            X_lin_test, _ = self.processor.prepare_linear_features(
                X_test, meta_test, scaler=scaler
            )

            # 1. Linear Expert (Ridge Regression)
            model_linear = self.models.get_linear_expert()
            model_linear.fit(X_lin_train, y_train)
            oof_preds[val_idx, 0] = model_linear.predict(X_lin_val)
            test_preds_accum[:, 0] += model_linear.predict(X_lin_test)

            # 2. Kernel Expert (SVR)
            model_kernel = self.models.get_kernel_expert()
            model_kernel.fit(X_lin_train, y_train)
            oof_preds[val_idx, 1] = model_kernel.predict(X_lin_val)
            test_preds_accum[:, 1] += model_kernel.predict(X_lin_test)

            # -----------------------------------------------------------------
            # Pipeline B: Partitioning Expert (Tree-Based)
            # Strategy: PCA on Embeddings -> Concatenate with Raw Metadata
            # -----------------------------------------------------------------
            X_tree_train, pca = self.processor.prepare_tree_features(
                X_train, meta_train, pca=None
            )
            X_tree_val, _ = self.processor.prepare_tree_features(
                X_val, meta_val, pca=pca
            )
            X_tree_test, _ = self.processor.prepare_tree_features(
                X_test, meta_test, pca=pca
            )

            # 3. Partitioning Expert (ExtraTrees)
            model_tree = self.models.get_partitioning_expert()
            model_tree.fit(X_tree_train, y_train)
            oof_preds[val_idx, 2] = model_tree.predict(X_tree_val)
            test_preds_accum[:, 2] += model_tree.predict(X_tree_test)

        # Average test predictions across folds
        test_preds = test_preds_accum / Config.N_FOLDS

        return oof_preds, test_preds

    def process_backbone(self, backbone_name, load_cached_data=True, debug=False):
        """
        Manages the processing for a single backbone:
        1. Checks cache for existing OOF/Test predictions.
        2. If missing, loads features, runs CV, and caches results.

        Args:
            backbone_name (str): Hugging Face model ID.
            load_cached_data (bool): Whether to attempt loading from cache.
            debug (bool): Run in debug mode (fewer samples).

        Returns:
            tuple: (oof_preds, test_preds, targets)
        """
        # Construct cache prefix
        safe_name = backbone_name.replace("/", "_").replace("-", "_")
        prefix = f"{safe_name}_stacking"
        if debug:
            prefix += "_debug"

        # 1. Try Loading from Cache
        if load_cached_data:
            cached = self.processor.load_processed_data(
                ["oof", "test_pred", "targets"], prefix
            )
            if cached:
                print(f"[{backbone_name}] Loaded Level-0 predictions from cache.")
                return cached["oof"], cached["test_pred"], cached["targets"]

        # 2. Compute from Scratch
        print(
            f"[{backbone_name}] Computing Level-0 predictions (Cache miss or force reload)..."
        )

        # Load features for all splits
        # We combine 'train' and 'val' splits to maximize data for Cross-Validation
        f_train, _, m_train, y_train = self.processor.get_raw_features(
            backbone_name, "train", load_cached_data, debug
        )
        f_val, _, m_val, y_val = self.processor.get_raw_features(
            backbone_name, "val", load_cached_data, debug
        )
        f_test, _, m_test, _ = self.processor.get_raw_features(
            backbone_name, "test", load_cached_data, debug
        )

        # Concatenate Train and Validation sets
        X_full = np.concatenate([f_train, f_val], axis=0)
        meta_full = np.concatenate([m_train, m_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)

        # Run CV
        oof_preds, test_preds = self._run_cv_for_backbone(
            backbone_name, X_full, meta_full, y_full, f_test, m_test
        )

        # Cache the results
        data_to_save = {"oof": oof_preds, "test_pred": test_preds, "targets": y_full}
        self.processor.save_processed_data(data_to_save, prefix)

        # Clean up memory
        del f_train, f_val, f_test, X_full
        import gc

        gc.collect()

        return oof_preds, test_preds, y_full

    def run(self, load_cached_data=True, debug=False):
        """
        Main execution entry point.
        1. Generates/Loads Level-0 predictions for all backbones.
        2. Trains Level-1 Meta-Learner.
        3. Generates submission file.

        Args:
            load_cached_data (bool): Whether to use cached intermediate files.
            debug (bool): Whether to run in debug mode.
        """
        seed_everything(Config.SEED)

        level1_oof_features = []
        level1_test_features = []
        y_target = None

        # ---------------------------------------------------------------------
        # Step 1: Level-0 Experts (Feature Extraction & Base Models)
        # ---------------------------------------------------------------------
        for backbone in Config.BACKBONES:
            oof, test, y = self.process_backbone(backbone, load_cached_data, debug)

            level1_oof_features.append(oof)
            level1_test_features.append(test)

            # Ensure targets are consistent across backbones
            if y_target is None:
                y_target = y
            else:
                if not np.allclose(y_target, y):
                    raise ValueError(
                        f"Target mismatch detected for backbone {backbone}. Check data ordering."
                    )

        # Concatenate all expert predictions
        # Shape: (N_samples, 3_backbones * 3_experts) = (N, 9)
        X_meta_train = np.hstack(level1_oof_features)
        X_meta_test = np.hstack(level1_test_features)

        print("-" * 40)
        print(f"Level-1 Stacking Data Shapes:")
        print(f"Train (OOF): {X_meta_train.shape}")
        print(f"Test:        {X_meta_test.shape}")
        print("-" * 40)

        # ---------------------------------------------------------------------
        # Step 2: Level-1 Meta-Learner
        # ---------------------------------------------------------------------
        print("Training Level-1 Meta-Learner (Ridge Stacking)...")
        meta_learner = self.models.get_meta_learner()
        meta_learner.fit(X_meta_train, y_target)

        # Evaluation on OOF
        oof_final_preds = meta_learner.predict(X_meta_train)
        rmse = compute_rmse(y_target, oof_final_preds)
        print(f"CV RMSE: {rmse}")

        # ---------------------------------------------------------------------
        # Step 3: Submission Generation
        # ---------------------------------------------------------------------
        print("Generating final predictions for Test set...")
        final_test_preds = meta_learner.predict(X_meta_test)

        # Retrieve Test IDs
        # We read directly from metadata to ensure correct ordering and avoid reloading heavy features
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)
        if debug:
            # Match the limit logic in dataset.py
            test_df = test_df.iloc[:100]

        ids_test = test_df["Id"].values

        # Create Submission DataFrame
        submission = pd.DataFrame({"Id": ids_test, "Pawpularity": final_test_preds})

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
