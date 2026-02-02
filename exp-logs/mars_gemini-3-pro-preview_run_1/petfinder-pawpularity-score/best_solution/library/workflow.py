import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from library.config import Config
from library.utils import seed_everything, create_stratified_folds, calculate_rmse
from library.feature_extraction import DeepFeatureExtractor
from library.models import (
    get_ridge_expert,
    get_svr_expert,
    get_extratrees_expert,
    get_lgbm_expert,
    get_meta_learner,
)


class StratifiedStackingRunner:
    """
    Orchestrates the Stratified Tri-Paradigm Stacking Ensemble.

    Pipeline:
    1. Merge Train/Val metadata for full Cross-Validation.
    2. Extract features using 3 Backbones (SigLIP, DINOv2, ConvNeXt).
    3. Perform Stratified K-Fold CV.
    4. Train 12 Level-0 Experts per fold (3 Backbones x 4 Algorithms).
    5. Train Level-1 Meta-Learner on OOF predictions.
    6. Generate final submission.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.extractor = DeepFeatureExtractor()

    def run(self):
        # 1. Setup and Configuration
        Config.DEBUG = self.debug
        Config.setup()
        seed_everything(Config.SEED)

        print("Step 1: Loading and Merging Metadata...")
        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        val_df = pd.read_csv(Config.VAL_METADATA_PATH)

        # Merge train and val for full stratified CV to fix distribution mismatch
        full_train_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)

        # Handle Debug Mode
        if self.debug:
            print(
                f"DEBUG MODE: Truncating datasets to {Config.DEBUG_SAMPLE_SIZE} samples."
            )
            full_train_df = full_train_df.head(Config.DEBUG_SAMPLE_SIZE)
            test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

        # 2. Feature Extraction (Backbones)
        print("Step 2: Extracting Deep Features...")
        # We use specific cache keys to distinguish this merged run
        train_cache_key = "full_cv"
        test_cache_key = "test_final"

        data_cache = {}

        for backbone in Config.BACKBONES.keys():
            # Extract/Load features for the merged training set
            data_cache[f"{backbone}_train"] = self.extractor.extract_features(
                full_train_df, backbone, train_cache_key, load_cached_data=True
            )
            # Extract/Load features for the test set
            data_cache[f"{backbone}_test"] = self.extractor.extract_features(
                test_df, backbone, test_cache_key, load_cached_data=True
            )

        # 3. Prepare Stacking Structures
        n_samples = len(full_train_df)
        n_test = len(test_df)
        y = full_train_df["Pawpularity"].values

        # Define the 12 Experts (3 Backbones * 4 Algorithms)
        expert_algos = ["ridge", "svr", "et", "lgbm"]
        expert_cols = []
        for bb in Config.BACKBONES.keys():
            for algo in expert_algos:
                expert_cols.append(f"{bb}_{algo}")

        # DataFrame to store Out-of-Fold predictions (Level-1 Training Data)
        oof_preds = pd.DataFrame(0.0, index=np.arange(n_samples), columns=expert_cols)

        # DataFrame to accumulate Test predictions across folds (for averaging)
        test_preds_accum = pd.DataFrame(
            0.0, index=np.arange(n_test), columns=expert_cols
        )

        # 4. Stratified K-Fold Cross-Validation
        print(f"Step 3: Starting Stratified {Config.N_FOLDS}-Fold Cross-Validation...")
        folds = create_stratified_folds(
            full_train_df, n_folds=Config.N_FOLDS, seed=Config.SEED
        )

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"\n--- Fold {fold_idx + 1}/{Config.N_FOLDS} ---")

            # Iterate through each backbone to process features
            for backbone in Config.BACKBONES.keys():
                # Retrieve cached data
                train_data = data_cache[f"{backbone}_train"]
                test_data = data_cache[f"{backbone}_test"]

                # Get raw embeddings and metadata
                X_all = train_data["features"]
                M_all = train_data["meta"]
                X_test_raw = test_data["features"]
                M_test = test_data["meta"]

                # Split Train/Val for this fold
                X_train, X_val = X_all[train_idx], X_all[val_idx]
                M_train, M_val = M_all[train_idx], M_all[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # === Branch 1: Linear Experts (Ridge, SVR) ===
                # Logic: Concat [Embeddings, Metadata] -> StandardScaler

                scaler = StandardScaler()

                # Prepare matrices
                X_lin_train_raw = np.hstack([X_train, M_train])
                X_lin_val_raw = np.hstack([X_val, M_val])
                X_lin_test_raw = np.hstack([X_test_raw, M_test])

                # Fit Scaler on Train, Transform Val/Test
                X_lin_train = scaler.fit_transform(X_lin_train_raw)
                X_lin_val = scaler.transform(X_lin_val_raw)
                X_lin_test = scaler.transform(X_lin_test_raw)

                # 1. Ridge Regression Expert
                ridge = get_ridge_expert()
                ridge.fit(X_lin_train, y_train)
                oof_preds.loc[val_idx, f"{backbone}_ridge"] = ridge.predict(X_lin_val)
                test_preds_accum[f"{backbone}_ridge"] += ridge.predict(X_lin_test)

                # 2. SVR Expert
                svr = get_svr_expert()
                svr.fit(X_lin_train, y_train)
                oof_preds.loc[val_idx, f"{backbone}_svr"] = svr.predict(X_lin_val)
                test_preds_accum[f"{backbone}_svr"] += svr.predict(X_lin_test)

                # === Branch 2: Tree Experts (ExtraTrees, LightGBM) ===
                # Logic: PCA(Embeddings) -> Concat [PCA_Embeddings, Metadata]

                # Fit PCA on Train
                n_comps = min(X_train.shape[0], Config.PCA_COMPONENTS)
                pca = PCA(n_components=n_comps, random_state=Config.SEED)

                X_pca_train = pca.fit_transform(X_train)
                X_pca_val = pca.transform(X_val)
                X_pca_test = pca.transform(X_test_raw)

                # Concat with raw metadata (Tree models handle binary well)
                X_tree_train = np.hstack([X_pca_train, M_train])
                X_tree_val = np.hstack([X_pca_val, M_val])
                X_tree_test = np.hstack([X_pca_test, M_test])

                # 3. ExtraTrees Expert
                et = get_extratrees_expert()
                et.fit(X_tree_train, y_train)
                oof_preds.loc[val_idx, f"{backbone}_et"] = et.predict(X_tree_val)
                test_preds_accum[f"{backbone}_et"] += et.predict(X_tree_test)

                # 4. LightGBM Expert
                lgbm = get_lgbm_expert()
                lgbm.fit(X_tree_train, y_train)
                oof_preds.loc[val_idx, f"{backbone}_lgbm"] = lgbm.predict(X_tree_val)
                test_preds_accum[f"{backbone}_lgbm"] += lgbm.predict(X_tree_test)

        # Average test predictions across folds
        test_preds_avg = test_preds_accum / Config.N_FOLDS

        # 5. Evaluation and Meta-Learning
        print("\nStep 4: Level-0 Expert Performance (OOF RMSE)")
        for col in expert_cols:
            score = calculate_rmse(y, oof_preds[col].values)
            print(f"{col}: {score}")

        print("\nStep 5: Training Level-1 Meta-Learner...")
        # Train Meta-Learner on OOF predictions
        meta_model = get_meta_learner()
        meta_model.fit(oof_preds.values, y)

        # Calculate CV Score (Proxy via OOF)
        meta_oof_pred = meta_model.predict(oof_preds.values)
        final_cv_rmse = calculate_rmse(y, meta_oof_pred)
        print(f"Final Ensemble CV RMSE: {final_cv_rmse}")

        # 6. Submission Generation
        print("Step 6: Generating Submission...")
        final_test_pred = meta_model.predict(test_preds_avg.values)

        submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": final_test_pred})

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
